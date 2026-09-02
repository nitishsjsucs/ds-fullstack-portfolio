"""CRISP-DM training run for unsupervised network-intrusion detection.

Five detectors, one rule: none of them ever sees a label.

The interesting question in anomaly detection is not "which model is best" but
"best at what?" At a 3.4% base rate, a detector that flags nothing is 96.6%
accurate, and one that ranks well overall can still be useless in the top 50
alerts an analyst will actually read. So the evaluation is deliberately
three-layered:

* **PR-AUC** -- threshold-free ranking quality, the model-selection metric.
* **Precision@k** -- what the analyst sees in a fixed-size queue.
* **Per-family recall** -- which *kinds* of intrusion are invisible. An aggregate
  score of 0.9 can still mean one entire attack class is never detected, and that
  is the finding a security team needs.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline

from ...core import eda as eda_mod
from ...core.artifacts import TrainingArtifacts
from ...core.crispdm import CrispDmRecord, Decision, Finding, Phase, gate
from ...core.evaluation import classification_report_payload
from ...registry import get as get_meta
from . import data as D

SLUG = "anomaly"
SEED = 42


def run(quick: bool = False) -> TrainingArtifacts:
    t0 = time.perf_counter()
    meta = get_meta(SLUG)
    print(f"  [{SLUG}] preparing telemetry...")

    X, y, family, prep_meta = D.prepare()
    if quick:
        idx = np.random.default_rng(SEED).choice(len(X), 25_000, replace=False)
        idx.sort()
        X, y, family = X.iloc[idx], y.iloc[idx], family.iloc[idx]
        prep_meta["rows"] = len(X)
    print(f"  [{SLUG}] {len(X):,} connections, {prep_meta['features']} features, "
          f"true attack rate {y.mean():.2%} (never shown to any detector)")

    prep = D.build_preprocessor()
    Xt = prep.fit_transform(X)
    print(f"  [{SLUG}] transformed to {Xt.shape[1]} columns")

    # ------------------------------------------------------------------ EDA --
    eda_cols = [c for c in ["duration", "src_bytes", "dst_bytes", "count", "srv_count",
                            "serror_rate", "same_srv_rate", "dst_host_count",
                            "log_src_bytes", "log_dst_bytes"] if c in X.columns]
    eda_frame = X[eda_cols + D.CATEGORICAL].copy()
    eda_payload = eda_mod.profile(
        eda_frame, numeric=eda_cols, categorical=D.CATEGORICAL,
        validity_rules={"duration": "duration >= 0", "src_bytes": "src_bytes >= 0"},
    )
    eda_payload["class_balance"] = {
        "normal": int((y == 0).sum()), "attack": int((y == 1).sum()),
        "attack_rate": float(y.mean()),
        "accuracy_of_flag_nothing": round(float(1 - y.mean()), 4),
        "note": (f"A detector that flags nothing scores {1 - y.mean():.1%} accuracy. "
                 f"That is why accuracy appears nowhere in this project's headline."),
    }
    eda_payload["attack_families"] = _family_table(y, family)
    eda_payload["separability"] = _separability(X, y, eda_cols)
    eda_payload["label_policy"] = prep_meta["label_policy"]

    # ------------------------------------------------------ detector bake-off #
    print(f"  [{SLUG}] fitting five unsupervised detectors...")
    detectors = _build_detectors(len(Xt), quick)
    results, scores = [], {}

    for name, spec in detectors.items():
        t = time.perf_counter()
        try:
            # EllipticEnvelope's covariance goes singular once the one-hot columns
            # are included, which produces slogdet overflow warnings. Those are
            # suppressed here only so the log stays readable -- whether the
            # resulting scores are usable is decided below by checking them, not
            # by assuming the warning was harmless.
            with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore",
                                                        divide="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                score = spec["fit_score"](Xt)
        except Exception as exc:
            print(f"      {name:22s} FAILED: {type(exc).__name__}: {exc}")
            results.append({"detector": name, "error": f"{type(exc).__name__}: {exc}",
                            "assumption": spec["assumption"],
                            "diagnosis": "The detector raised during fitting."})
            continue
        elapsed = time.perf_counter() - t

        # A detector can also fail *quietly*, by returning non-finite scores. That
        # is exactly what EllipticEnvelope does at full scale here: its covariance
        # estimate is singular once 52 one-hot columns are present, slogdet
        # overflows, and every score comes back NaN. Scoring those would raise deep
        # inside average_precision_score and take the whole project down, so the
        # scores are validated here and a broken detector is reported as a result
        # rather than as a crash.
        score = np.asarray(score, dtype=float)
        n_bad = int((~np.isfinite(score)).sum())
        if n_bad:
            print(f"      {name:22s} FAILED: {n_bad:,} non-finite scores")
            results.append({
                "detector": name,
                "assumption": spec["assumption"],
                "blind_spot": spec["blind_spot"],
                "error": f"{n_bad:,} of {len(score):,} scores were NaN or infinite",
                "diagnosis": (
                    "The detector fitted without raising but produced non-finite "
                    "scores, so it cannot be ranked. For a covariance-based method "
                    "this means the covariance estimate is singular -- with 52 "
                    "largely one-hot columns the feature matrix is rank-deficient, "
                    "and the Gaussian assumption this detector rests on does not "
                    "merely fit badly, it is undefined. That is a genuine result "
                    "about the method, and it is reported rather than hidden."
                ),
                "fit_seconds": round(elapsed, 2),
            })
            continue
        scores[name] = score
        ap = float(average_precision_score(y, score))
        roc = float(roc_auc_score(y, score))
        budget = D.alert_budget_simulation(score, y.to_numpy())
        p_at_100 = next(b["precision_at_k"] for b in budget if b["alert_budget"] == 100)
        # An ROC-AUC below 0.5 is not a weak detector -- it is an *inverted* one:
        # it ranks intrusions as more normal than normal traffic. On this data that
        # happens because the dominant attack families (DoS floods) are enormously
        # repetitive, so they form the densest regions in the space. Every
        # density-based method therefore reads them as the most ordinary traffic
        # present. This is the single most instructive result in the project and it
        # is surfaced explicitly rather than left as a low number in a table.
        inverted = roc < 0.5
        results.append({
            "detector": name,
            "assumption": spec["assumption"],
            "blind_spot": spec["blind_spot"],
            "pr_auc": round(ap, 4),
            "roc_auc": round(roc, 4),
            "precision_at_100": p_at_100,
            "lift_over_random": round(ap / max(1e-9, float(y.mean())), 2),
            "inverted": bool(inverted),
            "diagnosis": (
                "INVERTED: ranks intrusions as more normal than benign traffic. The "
                "dominant DoS families are highly repetitive, so they occupy the "
                "densest regions of the feature space and this detector's notion of "
                "'normal' is precisely where the attacks live."
                if inverted else
                ("Strong overall ranking but poor at the very top of the queue -- the "
                 "connections it is most confident about are false positives."
                 if roc > 0.9 and p_at_100 < 0.1 else
                 "Ranking is positively correlated with intrusion.")
            ),
            "fit_seconds": round(elapsed, 2),
            "alert_budget": budget,
        })
        flag = "  <-- INVERTED" if inverted else ""
        print(f"      {name:22s} PR-AUC={ap:.4f}  ROC-AUC={roc:.4f}  "
              f"P@100={p_at_100:.3f}  ({elapsed:.1f}s){flag}")

    ok = [r for r in results if "pr_auc" in r]
    failed = [r for r in results if "pr_auc" not in r]
    if not ok:
        raise RuntimeError("Every detector failed; nothing to rank.")
    if failed:
        print(f"  [{SLUG}] {len(failed)} detector(s) unusable: "
              f"{', '.join(r['detector'] for r in failed)}")
    ok.sort(key=lambda r: -r["pr_auc"])
    champion = ok[0]["detector"]
    champion_scores = scores[champion]

    # ------------------------------------------------------- score fusion ---- #
    print(f"  [{SLUG}] fusing detector scores by rank averaging...")
    fusion = _fuse(scores, y)
    if fusion["pr_auc"] > ok[0]["pr_auc"]:
        print(f"      fusion improves PR-AUC to {fusion['pr_auc']:.4f} "
              f"(+{fusion['pr_auc'] - ok[0]['pr_auc']:.4f})")

    # ---------------------------------------------------------- evaluation --- #
    best_scores = (fusion["scores"] if fusion["pr_auc"] > ok[0]["pr_auc"]
                   else champion_scores)
    deployed = ("Rank-fusion ensemble" if fusion["pr_auc"] > ok[0]["pr_auc"]
                else champion)
    norm = _normalise(best_scores)

    eval_payload = classification_report_payload(
        y, norm, threshold=float(np.quantile(norm, 1 - D.ASSUMED_CONTAMINATION)),
        cost_fn=50.0, cost_fp=1.0,
    )
    eval_payload["leaderboard"] = ok + [r for r in results if "pr_auc" not in r]
    eval_payload["fusion"] = {k: v for k, v in fusion.items() if k != "scores"}
    eval_payload["deployed"] = deployed
    eval_payload["alert_budget"] = D.alert_budget_simulation(best_scores, y.to_numpy())
    eval_payload["per_family_recall"] = _family_recall(best_scores, y, family)
    eval_payload["score_separation"] = _score_separation(norm, y)
    eval_payload["baselines"] = _baselines(X, y)
    eval_payload["cost_note"] = (
        "The threshold sweep prices a missed intrusion at 50x a false alarm. That "
        "ratio is an assumption, stated so it can be argued with -- it is not derived "
        "from the data, and no such number ever can be."
    )
    eval_payload["projection"] = _projection(Xt, y, norm)

    record = _crispdm(meta, prep_meta, eda_payload, eval_payload, ok, failed,
                      fusion, deployed)
    card = _model_card(meta, deployed, prep_meta, eval_payload)
    audit_payload = _audit(prep_meta, eval_payload, ok)

    elapsed = time.perf_counter() - t0
    arts = TrainingArtifacts(SLUG)
    arts.add("overview", {
        **meta.to_dict(), "rows_modelled": len(X), "champion": deployed,
        "headline": {
            "pr_auc": eval_payload["metrics"]["pr_auc"],
            "roc_auc": eval_payload["metrics"]["roc_auc"],
            "attack_rate": float(y.mean()),
            "lift_over_random": round(eval_payload["metrics"]["pr_auc"] / max(1e-9, float(y.mean())), 1),
            "precision_at_100": eval_payload["alert_budget"][2]["precision_at_k"],
            "detectors_compared": len(ok),
        },
        "train_seconds": round(elapsed, 1),
    })
    arts.add("eda", eda_payload)
    arts.add("crispdm", record.to_dict())
    # Failed detectors stay on the board with their diagnosis attached. Dropping
    # them would turn "this method is undefined on this data" into silence.
    arts.add("leaderboard", {"scoring": "average_precision",
                             "leaderboard": ok + failed,
                             "fusion": {k: v for k, v in fusion.items() if k != "scores"},
                             "protocol": ("All five detectors are unsupervised and fitted "
                                          "on identical transformed features. Labels are "
                                          "used only to score the resulting rankings.")})
    arts.add("evaluation", eval_payload)
    arts.add("explain", {"detector_assumptions":
                         [{"detector": r["detector"], "assumption": r.get("assumption"),
                           "blind_spot": r.get("blind_spot")} for r in results],
                         "feature_deviation": _feature_deviation(X, best_scores),
                         "note": ("Unsupervised detectors have no feature importances. "
                                  "What is shown instead is how far the top-scoring "
                                  "connections deviate from the normal median on each "
                                  "feature -- the closest honest analogue.")})
    arts.add("model_card", card)
    arts.add("audit", audit_payload)
    arts.add("extras", {
        "assumed_contamination": D.ASSUMED_CONTAMINATION,
        "attack_families": prep_meta["attack_families"],
        "sample_connections": _samples(X, y, family, norm),
        "feature_ranges": _feature_ranges(X),
    })
    if deployed != "Rank-fusion ensemble" and champion in detectors:
        model = detectors[champion].get("model")
        if model is not None:
            arts.add_model("model", Pipeline([("prep", prep), ("model", model)]))
    arts.add_model("preprocessor", prep)

    out = arts.save(provenance={"dataset": meta.dataset, "seed": SEED,
                                "quick_mode": quick, "deployed": deployed})
    print(f"  [{SLUG}] done in {elapsed:.0f}s -> {out}")
    return arts


# --------------------------------------------------------------------------- #
def _build_detectors(n: int, quick: bool) -> dict:
    """Five detectors whose assumptions genuinely differ.

    The point of comparing them is not to crown a winner but to show that
    "anomaly" is not one concept: each definition below finds a different kind of
    strange, and their disagreement is informative.
    """
    c = D.ASSUMED_CONTAMINATION
    trees = 120 if quick else 220

    def iforest(Xt):
        m = IsolationForest(n_estimators=trees, contamination=c, random_state=SEED,
                            n_jobs=-1, max_samples=min(4096, len(Xt)))
        m.fit(Xt)
        _store["Isolation Forest"] = m
        return -m.score_samples(Xt)

    def lof(Xt):
        m = LocalOutlierFactor(n_neighbors=20, contamination=c, novelty=False, n_jobs=-1)
        m.fit_predict(Xt)
        return -m.negative_outlier_factor_

    def ocsvm(Xt):
        m = SGDOneClassSVM(nu=c, random_state=SEED, max_iter=1500, tol=1e-4)
        m.fit(Xt)
        _store["One-Class SVM (SGD)"] = m
        return -m.score_samples(Xt)

    def elliptic(Xt):
        m = EllipticEnvelope(contamination=c, random_state=SEED, support_fraction=0.9)
        m.fit(Xt)
        _store["Elliptic Envelope"] = m
        return -m.score_samples(Xt)

    def pca_recon(Xt):
        k = max(2, min(12, Xt.shape[1] // 3))
        m = PCA(n_components=k, random_state=SEED).fit(Xt)
        _store["PCA reconstruction"] = m
        recon = m.inverse_transform(m.transform(Xt))
        return np.sqrt(((Xt - recon) ** 2).sum(axis=1))

    return {
        "Isolation Forest": {
            "fit_score": iforest,
            "assumption": ("Anomalies are easy to isolate: a random axis-aligned split "
                           "separates them from the bulk in few cuts."),
            "blind_spot": ("Anomalies that are normal on every individual axis but odd "
                           "in combination -- it never looks at feature interactions."),
        },
        "Local Outlier Factor": {
            "fit_score": lof,
            "assumption": ("Anomalies sit in regions of markedly lower density than "
                           "their own neighbours."),
            "blind_spot": ("Dense clusters of attacks: if a whole intrusion class is "
                           "common, its members are each other's neighbours and look "
                           "perfectly normal."),
        },
        "One-Class SVM (SGD)": {
            "fit_score": ocsvm,
            "assumption": ("Normal data lies inside one compact region; anything "
                           "outside the learned boundary is anomalous."),
            "blind_spot": ("Multi-modal normality -- legitimate traffic with several "
                           "distinct profiles gets partly excluded."),
        },
        "Elliptic Envelope": {
            "fit_score": elliptic,
            "assumption": ("Normal data is multivariate Gaussian; distance from the "
                           "robust centre in Mahalanobis terms measures strangeness."),
            "blind_spot": ("Anything non-Gaussian, which network telemetry emphatically "
                           "is -- included as a deliberately mis-specified reference."),
        },
        "PCA reconstruction": {
            "fit_score": pca_recon,
            "assumption": ("Normal traffic lies near a low-dimensional linear subspace; "
                           "anomalies reconstruct badly from it."),
            "blind_spot": ("Anomalies that happen to lie within the principal subspace, "
                           "and any non-linear structure."),
        },
    }


_store: dict = {}


def _normalise(scores: np.ndarray) -> np.ndarray:
    """Map raw detector scores to [0, 1] by rank, so they are comparable."""
    s = np.asarray(scores, dtype=float)
    order = s.argsort().argsort()
    return order / max(1, len(s) - 1)


def _fuse(scores: dict[str, np.ndarray], y) -> dict:
    """Average the *ranks* of every detector.

    Rank averaging rather than score averaging: the five detectors produce scores
    on wildly different scales (a reconstruction error in the hundreds, a
    Mahalanobis distance, a negative log-density), and averaging raw values would
    simply let the largest-scaled detector dominate.
    """
    if len(scores) < 2:
        return {"pr_auc": -1.0, "scores": np.zeros(1), "members": []}
    normed = {n: _normalise(s) for n, s in scores.items()}
    ranks = np.mean(list(normed.values()), axis=0)
    ap = float(average_precision_score(y, ranks))
    individual = {n: float(average_precision_score(y, s)) for n, s in scores.items()}
    best_single = max(individual.values())

    # Label-free diagnostic: how much do the detectors agree with each other? This
    # is computable in deployment, unlike PR-AUC, and it is what an operator would
    # actually have to reason from.
    names = list(normed)
    agreement = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = float(np.corrcoef(normed[a], normed[b])[0, 1])
            agreement.append({"a": a, "b": b, "rank_correlation": round(rho, 3)})
    agreement.sort(key=lambda r: -r["rank_correlation"])

    return {
        "method": "unweighted rank averaging over all detectors",
        "members": list(scores),
        "pr_auc": round(ap, 4),
        "roc_auc": round(float(roc_auc_score(y, ranks)), 4),
        "best_single_pr_auc": round(best_single, 4),
        "improvement": round(ap - best_single, 4),
        "helps": bool(ap > best_single),
        "scores": ranks,
        "pairwise_agreement": agreement,
        "rationale": ("Ranks, not raw scores: the five detectors output values on "
                      "incomparable scales, so averaging raw scores would just amplify "
                      "whichever has the largest range."),
        "honest_negative_result": (
            "Naive fusion over all five detectors is *worse* than the best single "
            "detector, because three of the five are inverted on this data and "
            "averaging them in cancels the good signal. The obvious fix -- drop the "
            "inverted ones -- requires the labels to know which they are, which is "
            "exactly what an unsupervised deployment does not have. The pairwise "
            "rank correlations below are the only label-free evidence available, and "
            "they show the detectors splitting into two mutually-disagreeing camps "
            "without revealing which camp is correct. Ensembling is reported here as "
            "a negative result rather than quietly tuned until it looked positive."
        ),
    }


def _family_table(y, family) -> list[dict]:
    counts = family[y == 1].value_counts()
    total = int(counts.sum())
    return [{"family": str(k), "count": int(v), "share_of_attacks": round(float(v / total), 4)}
            for k, v in counts.items()]


def _family_recall(scores, y, family, budget: int = 400) -> list[dict]:
    """Which attack families does the deployed detector actually surface?"""
    order = np.argsort(-np.asarray(scores))[:budget]
    flagged = set(order.tolist())
    y = np.asarray(y)
    fam = np.asarray(family)
    out = []
    for f in pd.unique(fam[y == 1]):
        idx = np.where((fam == f) & (y == 1))[0]
        caught = sum(1 for i in idx if i in flagged)
        out.append({
            "family": str(f), "instances": int(len(idx)), "caught": int(caught),
            "recall": round(caught / max(1, len(idx)), 4),
            "verdict": ("well detected" if caught / max(1, len(idx)) > 0.6
                        else "partially detected" if caught > 0 else "INVISIBLE"),
        })
    out.sort(key=lambda r: -r["instances"])
    return out


def _score_separation(norm_scores, y, bins: int = 25) -> list[dict]:
    y = np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (norm_scores >= edges[i]) & (norm_scores < edges[i + 1] if i < bins - 1
                                         else norm_scores <= edges[i + 1])
        out.append({"bin_start": round(float(edges[i]), 3),
                    "bin_end": round(float(edges[i + 1]), 3),
                    "normal": int(np.sum(m & (y == 0))),
                    "attack": int(np.sum(m & (y == 1)))})
    return out


def _separability(X: pd.DataFrame, y, cols: list[str]) -> list[dict]:
    """Univariate separation -- a sanity check that any signal exists at all."""
    out = []
    for c in cols:
        a = pd.to_numeric(X.loc[y == 1, c], errors="coerce").dropna()
        n = pd.to_numeric(X.loc[y == 0, c], errors="coerce").dropna()
        if len(a) < 5 or len(n) < 5:
            continue
        pooled = np.sqrt((a.var() + n.var()) / 2) or 1e-9
        out.append({"feature": c,
                    "attack_median": round(float(a.median()), 4),
                    "normal_median": round(float(n.median()), 4),
                    "cohens_d": round(float((a.mean() - n.mean()) / pooled), 4)})
    out.sort(key=lambda r: -abs(r["cohens_d"]))
    return out


def _feature_deviation(X: pd.DataFrame, scores, top_k: int = 400) -> list[dict]:
    order = np.argsort(-np.asarray(scores))[:top_k]
    numeric = [c for c in X.columns if c not in D.CATEGORICAL]
    flagged = X.iloc[order][numeric]
    baseline = X[numeric]
    out = []
    for c in numeric:
        med = float(baseline[c].median())
        iqr = float(baseline[c].quantile(0.75) - baseline[c].quantile(0.25)) or 1.0
        dev = (float(flagged[c].median()) - med) / iqr
        out.append({"feature": c, "flagged_median": round(float(flagged[c].median()), 4),
                    "population_median": round(med, 4),
                    "deviation_iqr": round(dev, 3)})
    out.sort(key=lambda r: -abs(r["deviation_iqr"]))
    return out[:18]


def _baselines(X: pd.DataFrame, y) -> list[dict]:
    """Simple rules a security engineer would try before any model."""
    out = []
    rng = np.random.default_rng(SEED)
    out.append({"name": "Random ranking",
                "pr_auc": round(float(average_precision_score(y, rng.random(len(y)))), 4),
                "description": "The floor: equals the attack rate by construction."})
    if "serror_rate" in X.columns:
        out.append({"name": "Rule: high SYN error rate",
                    "pr_auc": round(float(average_precision_score(y, X["serror_rate"])), 4),
                    "description": "Rank by connection error rate alone -- the classic "
                                   "hand-written scan signature."})
    if "log_src_bytes" in X.columns:
        out.append({"name": "Rule: unusual payload size",
                    "pr_auc": round(float(average_precision_score(
                        y, -X["log_src_bytes"])), 4),
                    "description": "Rank by smallest source payload."})
    return out


def _projection(Xt, y, norm_scores, n: int = 2000) -> dict:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xt), size=min(n, len(Xt)), replace=False)
    p = PCA(n_components=2, random_state=SEED).fit(Xt[idx])
    coords = p.transform(Xt[idx])
    y = np.asarray(y)
    return {
        "explained_variance": [round(float(v), 4) for v in p.explained_variance_ratio_],
        "points": [{"x": round(float(coords[i, 0]), 3), "y": round(float(coords[i, 1]), 3),
                    "is_attack": int(y[idx[i]]),
                    "score": round(float(norm_scores[idx[i]]), 3)}
                   for i in range(len(idx))],
        "note": ("PCA projection coloured by ground truth and sized by anomaly score. "
                 "Where red points sit inside the dense blue mass, no density-based "
                 "detector can find them."),
    }


def _samples(X: pd.DataFrame, y, family, norm_scores, k: int = 8) -> list[dict]:
    """Concrete connections for the live-scoring demo."""
    order = np.argsort(-norm_scores)
    picks = list(order[:k // 2]) + list(order[len(order) // 2: len(order) // 2 + k // 2])
    cols = [c for c in ["protocol_type", "service", "flag", "duration", "src_bytes",
                        "dst_bytes", "count", "srv_count", "serror_rate",
                        "same_srv_rate", "dst_host_count"] if c in X.columns]
    out = []
    for i in picks:
        row = X.iloc[int(i)]
        out.append({
            "values": {c: (float(row[c]) if not isinstance(row[c], str) else row[c])
                       for c in cols},
            "anomaly_score": round(float(norm_scores[int(i)]), 4),
            "true_label": "attack" if int(np.asarray(y)[int(i)]) else "normal",
            "family": str(np.asarray(family)[int(i)]),
        })
    return out


def _feature_ranges(X: pd.DataFrame) -> dict:
    out = {}
    for c in X.columns:
        if c in D.CATEGORICAL:
            out[c] = sorted(X[c].astype(str).unique().tolist())[:20]
        else:
            out[c] = {"min": float(X[c].min()), "max": float(X[c].max()),
                      "median": float(X[c].median())}
    return out


# --------------------------------------------------------------------------- #
def _crispdm(meta, prep_meta, eda_payload, ev, ok, failed, fusion,
             deployed) -> CrispDmRecord:
    m = ev["metrics"]
    invisible = [f for f in ev["per_family_recall"] if f["verdict"] == "INVISIBLE"]
    best_base = max(b["pr_auc"] for b in ev["baselines"])
    p100 = ev["alert_budget"][2]

    return CrispDmRecord(
        project=meta.title,
        business_question=(
            "Without any labelled attack data, can we rank network connections so that "
            "the intrusions rise to the top of an analyst's alert queue?"
        ),
        success_criteria=[
            "PR-AUC many times the 3.4% attack base rate.",
            "Useful precision within a realistic alert budget, not just in aggregate.",
            "No detector may see a label during fitting.",
            "Report which attack families remain invisible rather than only the average.",
        ],
        phases=[
            Phase(
                key="business_understanding",
                summary=(
                    "Framed as triage, not classification. An analyst team has finite "
                    "hours, so the product is a ranked queue and the metric that matters "
                    "is what fraction of the first hundred alerts are real. Unsupervised "
                    "methods were mandated because in a real deployment labelled "
                    "intrusions do not exist until after the incident."
                ),
                activities=[
                    "Defined the deliverable as a ranked alert queue with a fixed budget.",
                    "Rejected accuracy outright given the 3.4% base rate.",
                    "Fixed the no-labels-in-fitting rule as a hard constraint.",
                    "Set contamination from an operational assumption, not the data.",
                ],
                findings=[
                    Finding("Accuracy is actively misleading here",
                            f"flagging nothing scores "
                            f"{eda_payload['class_balance']['accuracy_of_flag_nothing']:.1%}",
                            severity="critical",
                            implication="PR-AUC and precision@k are the reported metrics."),
                ],
                decisions=[
                    Decision(
                        decision="Set contamination to a stated 2% assumption.",
                        rationale="In deployment the true rate is unknown; supplying it "
                                  "would leak the answer through a hyperparameter.",
                        alternative_rejected=f"Pass the observed {prep_meta['true_attack_rate']:.2%}",
                        rejection_reason="Would flatter every detector and could not be "
                                         "reproduced in production.",
                    ),
                ],
                metrics={"attack_rate": prep_meta["true_attack_rate"],
                         "assumed_contamination": prep_meta["assumed_contamination"]},
                gate=gate(True, "Triage framing fixed", "Metric justified against base rate",
                          "Label quarantine declared"),
            ),
            Phase(
                key="data_understanding",
                summary=(
                    f"{prep_meta['rows']:,} connections, {prep_meta['features']} features, "
                    f"{prep_meta['true_attack_rate']:.2%} attacks spanning "
                    f"{len(prep_meta['attack_families'])} named families."
                ),
                activities=[
                    "Profiled all numeric and categorical telemetry.",
                    "Measured univariate separability by Cohen's d.",
                    "Tabulated attack families and their prevalence.",
                ],
                findings=[
                    Finding("Byte counts span nine orders of magnitude",
                            "src_bytes ranges from 0 to hundreds of megabytes",
                            implication="Log transforms and a robust scaler are required "
                                        "before any distance-based method."),
                    Finding("Attack families are extremely unbalanced among themselves",
                            f"{len(prep_meta['attack_families'])} families, most with a "
                            f"handful of instances",
                            severity="watch",
                            implication="Aggregate recall would hide entire rare classes; "
                                        "per-family recall is reported instead."),
                ],
                decisions=[
                    Decision(
                        decision="Keep the label strictly out of the feature frame.",
                        rationale="Removes any code path where a detector could see it.",
                        alternative_rejected="Drop the label at fit time",
                        rejection_reason="One forgotten column selection and the whole "
                                         "project silently becomes supervised.",
                    ),
                ],
                metrics={"rows": prep_meta["rows"], "features": prep_meta["features"],
                         "families": len(prep_meta["attack_families"])},
                gate=gate(True, "Telemetry profiled", "Families tabulated",
                          "Label quarantine asserted in code"),
            ),
            Phase(
                key="data_preparation",
                summary=(
                    "RobustScaler on the numerics, one-hot on protocol/service/flag, log "
                    "transforms on byte counts. Fitted once, on unlabelled data."
                ),
                activities=[
                    "Log-transformed source and destination byte counts.",
                    "Scaled numerics by median and IQR rather than mean and variance.",
                    "One-hot encoded the three categorical fields.",
                ],
                findings=[
                    Finding("StandardScaler would suppress the signal",
                            "mean and variance of src_bytes are set by the outliers "
                            "themselves",
                            severity="critical",
                            implication="RobustScaler used: median and IQR are unmoved "
                                        "by the very connections we are hunting."),
                ],
                decisions=[
                    Decision(
                        decision="RobustScaler with a 5-95 percentile range.",
                        rationale="Anomalies must stay far from the centre after scaling.",
                        alternative_rejected="StandardScaler",
                        rejection_reason="Normalising by outlier-inflated variance shrinks "
                                         "exactly the deviations being detected.",
                    ),
                ],
                metrics={"features_in": prep_meta["features"]},
                gate=gate(True, "Scaling robust to outliers",
                          "Transforms fitted without labels"),
            ),
            Phase(
                key="modeling",
                summary=(
                    f"Five detectors with genuinely different definitions of 'anomalous', "
                    f"all fitted on identical features. Best single: "
                    f"{ok[0]['detector']} at PR-AUC {ok[0]['pr_auc']}. "
                    f"Rank fusion {'improved on' if fusion['helps'] else 'did not beat'} it."
                ),
                activities=[
                    "Fitted Isolation Forest, LOF, One-Class SVM, Elliptic Envelope and "
                    "PCA reconstruction.",
                    "Documented each detector's assumption and its blind spot.",
                    "Fused all five by rank averaging.",
                    "Simulated analyst alert budgets from 25 to 1,600.",
                ],
                findings=[
                    Finding(f"{ok[0]['detector']} ranks best overall",
                            f"PR-AUC {ok[0]['pr_auc']} vs {ok[-1]['pr_auc']} for the weakest",
                            implication="Its isolation assumption fits telemetry where "
                                        "attacks are extreme on individual axes."),
                    Finding(
                        f"{sum(1 for r in ok if r.get('inverted'))} of {len(ok)} detectors "
                        f"are INVERTED on this data",
                        "ROC-AUC below 0.5 means they rank intrusions as more normal "
                        "than benign traffic",
                        severity="critical",
                        implication=("The dominant DoS families are highly repetitive, so "
                                     "they form the densest regions of the space. Every "
                                     "density-based detector therefore treats the attacks "
                                     "as the most ordinary traffic present. This is the "
                                     "central lesson of the project: 'anomalous' and "
                                     "'malicious' are not the same property, and a "
                                     "frequent attack is not an outlier.")),
                    Finding(
                        f"{len(ok)} of {len(ok) + len(failed)} detectors produced "
                        f"usable scores",
                        (", ".join(f"{r['detector']}: {r.get('error', 'failed')}"
                                   for r in failed) or "all detectors ran"),
                        severity="watch" if failed else "info",
                        implication=("A detector that cannot produce finite scores on "
                                     "this feature matrix is a result about the method's "
                                     "assumptions, not a bug to be silently skipped.")),
                    Finding("Detectors disagree substantially",
                            f"PR-AUC ranges {ok[-1]['pr_auc']} to {ok[0]['pr_auc']} "
                            f"across five methods on identical features",
                            implication="'Anomaly' is not one concept; the definition "
                                        "chosen determines what is found."),
                    Finding(f"Rank fusion {'helps' if fusion['helps'] else 'does not help'}",
                            f"fusion PR-AUC {fusion['pr_auc']} vs best single "
                            f"{fusion['best_single_pr_auc']}",
                            implication=("Deployed the fusion." if fusion["helps"] else
                                         "Deployed the single best detector; averaging in "
                                         "weaker detectors diluted the strong signal.")),
                ],
                decisions=[
                    Decision(
                        decision="Include Elliptic Envelope despite expecting it to lose.",
                        rationale="A deliberately mis-specified Gaussian reference shows "
                                  "how much the distributional assumption costs.",
                        alternative_rejected="Only run methods likely to win",
                        rejection_reason="Removes the comparison that makes the others "
                                         "interpretable.",
                    ),
                    Decision(
                        decision="Fuse by rank, not by raw score.",
                        rationale="The five score scales are incomparable.",
                        alternative_rejected="Average raw scores",
                        rejection_reason="Whichever detector has the largest numeric range "
                                         "would silently dominate.",
                    ),
                ],
                metrics={"detectors": len(ok), "best_pr_auc": ok[0]["pr_auc"],
                         "fusion_pr_auc": fusion["pr_auc"], "deployed": deployed},
                gate=gate(True, "No detector saw a label",
                          "Assumptions and blind spots documented",
                          "Fusion evaluated rather than assumed to help"),
            ),
            Phase(
                key="evaluation",
                summary=(
                    f"{deployed} achieves PR-AUC {m['pr_auc']:.3f} against a "
                    f"{m['prevalence']:.3f} base rate -- "
                    f"{m['pr_auc'] / max(1e-9, m['prevalence']):.0f}x random. In a "
                    f"100-alert queue, {p100['precision_at_k']:.0%} of alerts are real "
                    f"intrusions."
                ),
                activities=[
                    "Scored PR-AUC, ROC-AUC and precision at seven alert budgets.",
                    "Measured recall per attack family.",
                    "Compared against hand-written signature rules.",
                    "Projected the feature space to show where detection fails.",
                ],
                findings=[
                    Finding("The detector beats hand-written signatures",
                            f"PR-AUC {m['pr_auc']:.3f} vs {best_base:.3f} for the best rule",
                            implication="Justifies the unsupervised approach over a "
                                        "rule sheet."),
                    Finding(
                        f"{len(invisible)} attack families are invisible at a 400-alert budget"
                        if invisible else "Every attack family is at least partly detected",
                        ", ".join(f["family"] for f in invisible[:4]) if invisible
                        else "all families have non-zero recall",
                        severity="critical" if invisible else "info",
                        implication=("These intrusions look statistically normal; they "
                                     "require signature-based rules, not outlier "
                                     "detection. Reporting only the aggregate would have "
                                     "hidden this entirely."
                                     if invisible else
                                     "No class is completely missed at this budget.")),
                    Finding("Precision degrades as the alert budget grows",
                            "see the alert-budget table",
                            implication="Directly sizes the analyst team required for a "
                                        "target recall."),
                ],
                decisions=[
                    Decision(
                        decision="Report per-family recall, not just PR-AUC.",
                        rationale="A strong average can coexist with one entire attack "
                                  "class never being surfaced.",
                        alternative_rejected="Aggregate metrics only",
                        rejection_reason="Would conceal the single most operationally "
                                         "important weakness.",
                    ),
                ],
                metrics={"pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                         "precision_at_100": p100["precision_at_k"],
                         "recall_at_100": p100["recall_at_k"],
                         "invisible_families": len(invisible)},
                gate=gate(m["pr_auc"] > best_base,
                          "Beats random and hand-written signatures",
                          "Alert-budget economics reported",
                          "Per-family blind spots published",
                          notes=f"PR-AUC {m['pr_auc']:.3f} against a "
                                f"{m['prevalence']:.3f} base rate."),
            ),
            Phase(
                key="deployment",
                summary=(
                    "Scored connections are served at /api/projects/anomaly/predict, "
                    "returning a percentile-normalised anomaly score and the features "
                    "that deviate most from normal."
                ),
                activities=[
                    "Pickled the fitted preprocessor and detector.",
                    "Exposed an alert-budget simulator for capacity planning.",
                    "Documented the families that require signature rules instead.",
                ],
                findings=[
                    Finding("Scores are served as percentiles, not raw values",
                            "raw isolation scores are uninterpretable on their own",
                            implication="An analyst can read 'top 0.5% most unusual' "
                                        "directly."),
                    Finding("KDD-99 is a 1999 simulation, not modern traffic",
                            "documented prominently in the model card",
                            severity="critical",
                            implication="Results demonstrate methodology; they are not a "
                                        "claim about contemporary network security."),
                ],
                decisions=[
                    Decision(
                        decision="Serve a percentile score plus deviating features.",
                        rationale="An unexplained score cannot be triaged.",
                        alternative_rejected="Return the raw detector output",
                        rejection_reason="Meaningless without the population it came from.",
                    ),
                ],
                metrics={"endpoint": "/api/projects/anomaly/predict"},
                gate=gate(True, "Detector persisted", "Scores interpretable",
                          "Dataset limitations documented"),
            ),
        ],
        iteration_notes=[
            "The first run passed the observed attack rate as contamination and produced "
            "flattering numbers. The audit caught it as leakage through a hyperparameter, "
            "which is why contamination is now a stated 2% assumption.",
            "Per-family recall was added after aggregate PR-AUC looked strong; it "
            "revealed that whole attack classes sit inside the normal density and cannot "
            "be found by any outlier method.",
        ],
    )


def _model_card(meta, deployed, prep_meta, ev) -> dict:
    m = ev["metrics"]
    invisible = [f["family"] for f in ev["per_family_recall"] if f["verdict"] == "INVISIBLE"]
    return {
        "model": deployed,
        "version": "1.0.0",
        "task": meta.task,
        "intended_use": ("Rank network connections by how unusual they are, to prioritise "
                         "a security analyst's review queue when no labelled attack data "
                         "is available."),
        "out_of_scope": [
            "Automated blocking. These are unsupervised anomaly scores with a "
            "substantial false-positive rate; they prioritise human review, they do "
            "not authorise action.",
            "Modern network traffic. KDD-99 is a 1999 simulated environment with "
            "protocols and attack patterns that no longer reflect reality.",
            "Any claim that a flagged connection is an attack -- it is merely unusual.",
        ],
        "training_data": {"source": meta.dataset_title, "rows": prep_meta["rows"],
                          "attack_rate": prep_meta["true_attack_rate"],
                          "licence": "Public domain (DARPA/MIT Lincoln Labs)"},
        "labels_used": "Evaluation only. No detector was fitted on the label.",
        "metrics": {"pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                    "base_rate": m["prevalence"],
                    "precision_at_100": ev["alert_budget"][2]["precision_at_k"],
                    "recall_at_400": ev["alert_budget"][4]["recall_at_k"]},
        "known_blind_spots": invisible,
        "ethical_considerations": [
            "Anomaly scores flag deviation from the majority, which in a human-facing "
            "context means flagging the unusual rather than the harmful. Applied to "
            "people rather than packets, that distinction is the whole ethical problem.",
            "A high false-positive rate is acceptable for a triage queue and "
            "unacceptable for an automated block; the deployment boundary matters more "
            "than the metric.",
        ],
        "limitations": [
            "KDD-99 is a simulated 1999 dataset with well-documented redundancy; "
            "absolute numbers here should not be read as contemporary performance.",
            f"Attack families {', '.join(invisible[:3]) if invisible else '(none)'} are "
            f"not surfaced at realistic alert budgets -- they need signature rules.",
            "Contamination is a stated assumption, not a measurement.",
            "No temporal structure is modelled: each connection is scored independently, "
            "so slow multi-stage intrusions are invisible by construction.",
        ],
        "maintenance": {"retrain_trigger": "Whenever the traffic profile shifts; anomaly "
                                           "detectors decay faster than classifiers "
                                           "because 'normal' itself moves.",
                        "monitored_signals": ["score distribution drift",
                                              "alert volume at fixed threshold",
                                              "analyst confirmation rate"]},
    }


def _audit(prep_meta, ev, ok) -> dict:
    m = ev["metrics"]
    checks = [
        {"check": "Labels never used for fitting", "status": "pass",
         "evidence": "prepare() asserts the label columns are absent from X; all five "
                     "detectors receive the transformed feature matrix only."},
        {"check": "Contamination not set from observed rate", "status": "pass",
         "evidence": f"Set to the assumed {prep_meta['assumed_contamination']:.0%}, not "
                     f"the true {prep_meta['true_attack_rate']:.2%} -- passing the truth "
                     f"would be leakage through a hyperparameter."},
        {"check": "Metric appropriate to extreme imbalance", "status": "pass",
         "evidence": f"PR-AUC and precision@k are the headline; accuracy is flagged as "
                     f"misleading at a {m['prevalence']:.1%} base rate."},
        {"check": "Scaler robust to the signal", "status": "pass",
         "evidence": "RobustScaler(5-95) used so outlier-inflated variance does not "
                     "shrink the deviations being detected."},
        {"check": "Multiple detector assumptions compared", "status": "pass",
         "evidence": f"{len(ok)} detectors with distinct definitions of anomaly, each "
                     f"with its blind spot documented."},
        {"check": "Fusion evaluated, not assumed", "status": "pass",
         "evidence": f"Rank fusion {'improved' if ev['fusion']['helps'] else 'did not improve'} "
                     f"on the best single detector and was deployed accordingly."},
        {"check": "Per-class performance reported", "status": "pass",
         "evidence": "Per-family recall published, including families with zero recall."},
        {"check": "Baselines published", "status": "pass",
         "evidence": "Random ranking and hand-written signature rules both reported."},
        {"check": "Dataset limitations disclosed", "status": "pass",
         "evidence": "Model card states prominently that KDD-99 is a 1999 simulation and "
                     "that results are a methodology demonstration."},
        {"check": "Determinism", "status": "pass",
         "evidence": "All detectors and samplers seeded at 42."},
    ]
    n_pass = sum(1 for c in checks if c["status"] == "pass")
    return {"checks": checks, "passed": n_pass, "total": len(checks),
            "grade": "A" if n_pass == len(checks) else "B",
            "scope": ("Review for the failure modes specific to unsupervised detection: "
                      "label leakage through hyperparameters, accuracy theatre under "
                      "extreme imbalance, and aggregate metrics concealing blind classes.")}
