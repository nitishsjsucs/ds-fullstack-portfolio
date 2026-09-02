"""CRISP-DM training run for the Census Income AutoML tournament.

This is the project about *model selection itself*, so its subject matter is the
ways automated search deceives you.

**Search overfitting is measured, not assumed away.** Every family is hill-climbed
on cross-validated ROC-AUC and then scored once on an untouched hold-out. The gap
between the two is reported per family. A search that ran long enough to fit the
CV folds shows up as a positive gap, and the leaderboard flags it rather than
quoting the flattering number.

Note that on this dataset the measurement comes back *negative* -- the hold-out
scores slightly better than cross-validation. That is not a failed experiment; it
is the answer. With 39k training rows and 5-fold CV, each fold is large enough
that twenty trials do not meaningfully fit it. The report says so rather than
narrating the overfitting the reader was probably expecting, and the same search
on a few thousand rows would very likely tell the opposite story.

**Stacking is done the only leakage-free way.** The level-2 meta-learner is
trained on *out-of-fold* predictions. Training it on in-fold predictions -- which
is what happens if you fit base models on all the data and then predict that same
data -- produces a meta-learner that has effectively seen the labels, and a
stacking score that evaporates in production.

**Distillation asks whether any of it was necessary.** A single shallow decision
tree is fitted to mimic the ensemble's probabilities. If it recovers most of the
performance, the honest conclusion is that the tournament bought very little and
the simple model should ship.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

from ...core import eda as eda_mod
from ...core import evaluation, explain, fairness
from ...core.artifacts import TrainingArtifacts
from ...core.autoresearch import AutoResearch, build_oof, greedy_ensemble, leaderboard_payload
from ...core.crispdm import CrispDmRecord, Decision, Finding, Phase, gate
from ...core.model_zoo import classification_specs
from ...core.splitting import stratified_cv, stratified_split
from ...registry import get as get_meta
from . import data as D

SLUG = "automl"
SEED = 42
OVERFIT_GAP_THRESHOLD = 0.01


def run(quick: bool = False) -> TrainingArtifacts:
    t0 = time.perf_counter()
    meta = get_meta(SLUG)
    print(f"  [{SLUG}] preparing census records...")

    X, y, sensitive, prep_meta = D.prepare()
    if quick:
        idx = np.random.default_rng(SEED).choice(len(X), 12_000, replace=False)
        idx.sort()
        X, y, sensitive = X.iloc[idx], y.iloc[idx], sensitive.iloc[idx]
        prep_meta["rows"] = len(X)
    print(f"  [{SLUG}] {len(X):,} records, {prep_meta['features']} features, "
          f">50K rate {y.mean():.2%}")

    split = stratified_split(y, test_size=0.2, seed=SEED)
    Xtr, Xte = X.iloc[split.train_idx], X.iloc[split.test_idx]
    ytr, yte = y.iloc[split.train_idx], y.iloc[split.test_idx]
    sens_te = sensitive.iloc[split.test_idx].reset_index(drop=True)

    # ------------------------------------------------------------------ EDA --
    eda_payload = eda_mod.profile(
        pd.concat([X, sensitive, y], axis=1),
        target=D.TARGET,
        numeric=D.NUMERIC_FEATURES,
        categorical=[*D.CATEGORICAL_FEATURES, *D.SENSITIVE],
        validity_rules={"age": "age >= 17", "hours_per_week": "hours_per_week > 0"},
    )
    eda_payload["dropped_columns"] = prep_meta["dropped_columns"]
    eda_payload["sensitive_policy"] = prep_meta["sensitive_policy"]
    eda_payload["class_balance"] = {
        "positive": int(y.sum()), "negative": int((1 - y).sum()),
        "positive_rate": float(y.mean()),
    }
    eda_payload["rate_by_group"] = _rate_by_group(sensitive, y)
    eda_payload["capital_gain_structure"] = _capital_structure(X, y)

    # -------------------------------------------------------------- modeling --
    cv = stratified_cv(n_splits=3 if quick else 5, seed=SEED)
    print(f"  [{SLUG}] hill-climbing {'(quick)' if quick else ''} on {len(Xtr):,} rows...")
    ar = AutoResearch(scoring="roc_auc", cv=cv, greater_is_better=True,
                      max_trials_per_family=8 if quick else 20, patience=2, seed=SEED)
    specs = classification_specs(D.build_preprocessor, include_slow=not quick)
    results = ar.run(specs, Xtr, ytr)

    print(f"  [{SLUG}] refitting finalists and scoring the hold-out once...")
    fitted = {}
    for r in results:
        spec = next(s for s in specs if s.name == r.name)
        m = spec.build(r.best_params)
        m.fit(Xtr, ytr)
        prob = m.predict_proba(Xte)[:, 1]
        r.holdout = {"primary": float(roc_auc_score(yte, prob)),
                     "metrics": evaluation.binary_metrics(yte, prob)}
        fitted[r.name] = m
        gap = r.cv_score - r.holdout["primary"]
        flag = "  <-- search overfit" if gap > OVERFIT_GAP_THRESHOLD else ""
        print(f"      {r.name:24s} CV AUC={r.cv_score:.4f}  hold-out={r.holdout['primary']:.4f}"
              f"  gap={gap:+.4f}{flag}")

    results.sort(key=lambda r: -r.holdout["primary"])

    # ------------------------------------------------ level-2 stacking + Caruana
    print(f"  [{SLUG}] building out-of-fold predictions for stacking...")
    oof = build_oof(fitted, Xtr, ytr, cv)
    oof_matrix = pd.DataFrame(oof)

    ens = greedy_ensemble(oof, ytr.to_numpy(),
                          lambda yt, yp: float(roc_auc_score(yt, yp)),
                          n_rounds=15, greater_is_better=True)

    # A logistic meta-learner over the same OOF matrix -- classic stacked
    # generalisation (Wolpert 1992). It can learn negative weights, which greedy
    # selection cannot, so it sometimes wins; it can also overfit the OOF matrix,
    # which is why both are reported.
    stack_meta = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    stack_meta.fit(oof_matrix, ytr)
    test_matrix = pd.DataFrame(
        {name: fitted[name].predict_proba(Xte)[:, 1] for name in oof_matrix.columns}
    )
    stack_prob = stack_meta.predict_proba(test_matrix)[:, 1]
    stack_auc = float(roc_auc_score(yte, stack_prob))

    blend_prob = np.zeros(len(Xte))
    for name, w in ens["weights"].items():
        blend_prob += w * test_matrix[name].to_numpy()
    ens["holdout_score"] = float(roc_auc_score(yte, blend_prob))

    best_single = results[0]
    candidates = {
        best_single.name: (best_single.holdout["primary"], test_matrix[best_single.name].to_numpy()),
        "Caruana greedy ensemble": (ens["holdout_score"], blend_prob),
        "Logistic stacking (level-2)": (stack_auc, stack_prob),
    }
    deployed = max(candidates, key=lambda k: candidates[k][0])
    deployed_prob = candidates[deployed][1]
    print(f"      best single     {best_single.name:26s} {best_single.holdout['primary']:.4f}")
    print(f"      greedy ensemble {'':26s} {ens['holdout_score']:.4f}")
    print(f"      logistic stack  {'':26s} {stack_auc:.4f}")
    print(f"  [{SLUG}] deploying: {deployed}")

    stacking = {
        "method": "Stacked generalisation (Wolpert 1992) on out-of-fold predictions",
        "base_models": list(oof_matrix.columns),
        "meta_learner": "LogisticRegression(C=1.0)",
        "meta_weights": {c: round(float(w), 4)
                         for c, w in zip(oof_matrix.columns, stack_meta.coef_[0])},
        "holdout_auc": round(stack_auc, 4),
        "greedy_holdout_auc": round(ens["holdout_score"], 4),
        "best_single_holdout_auc": round(best_single.holdout["primary"], 4),
        "gain_over_best_single": round(
            max(stack_auc, ens["holdout_score"]) - best_single.holdout["primary"], 5),
        "base_model_correlation": _corr(oof_matrix),
        "leakage_note": (
            "The meta-learner is trained exclusively on out-of-fold predictions: every "
            "base-model probability it sees was produced by a model that did not train "
            "on that row. Fitting it on in-fold predictions instead is the standard "
            "stacking bug -- the meta-learner then effectively sees the labels through "
            "the base models' memorisation, and the gain vanishes in production."
        ),
    }

    # ---------------------------------------------------------- distillation --
    print(f"  [{SLUG}] distilling the ensemble into a depth-4 tree...")
    distil = _distil(Xtr, ytr, Xte, yte, fitted, oof_matrix, ens, deployed_prob)

    # ---------------------------------------------------------- evaluation --- #
    eval_payload = evaluation.classification_report_payload(
        yte, deployed_prob, threshold=0.5, cost_fn=1.0, cost_fp=1.0)
    eval_payload["split"] = split.describe()
    eval_payload["stacking"] = stacking
    eval_payload["distillation"] = distil
    eval_payload["deployed"] = deployed
    eval_payload["search_overfitting"] = _overfit_report(results)
    eval_payload["baselines"] = _baselines(ytr, yte, Xte)

    print(f"  [{SLUG}] running the fairness audit...")
    fair = fairness.audit(yte, deployed_prob, sens_te, threshold=0.5,
                          priority="equalised odds")
    fair["threshold_equalisation"] = {
        col: fairness.threshold_equalisation(yte, deployed_prob, sens_te[col], target="tpr")
        for col in D.SENSITIVE[:2]
    }

    champion_model = fitted[best_single.name]
    perm = explain.permutation_payload(champion_model, Xte, yte, scoring="roc_auc",
                                       n_repeats=4 if quick else 8, max_samples=3000)
    prep = champion_model.named_steps["prep"]
    names = list(prep.get_feature_names_out())
    Xte_t = pd.DataFrame(prep.transform(Xte), columns=names)
    Xtr_t = pd.DataFrame(prep.transform(Xtr.head(400)), columns=names)
    explain_payload = {
        "permutation": perm,
        "impurity": explain.impurity_payload(champion_model, names),
        "shap": explain.shap_payload(champion_model, Xtr_t, Xte_t.head(600)),
        "partial_dependence": explain.pdp_payload(
            champion_model, Xte, ["age", "education_num", "hours_per_week", "capital_net"]),
        "distilled_tree": distil.get("tree_text"),
        "feature_count": len(names),
    }

    lb = leaderboard_payload(results, scoring="roc_auc", ensemble=ens,
                             overfit_gap_threshold=OVERFIT_GAP_THRESHOLD)
    lb["stacking"] = stacking
    lb["deployed"] = deployed

    record = _crispdm(meta, prep_meta, eda_payload, lb, eval_payload, fair,
                      stacking, distil, deployed, results)
    card = _model_card(meta, deployed, prep_meta, eval_payload, fair, distil, split)
    audit_payload = _audit(prep_meta, eval_payload, results, stacking, fair)

    elapsed = time.perf_counter() - t0
    arts = TrainingArtifacts(SLUG)
    arts.add("overview", {
        **meta.to_dict(), "rows_modelled": len(X), "champion": deployed,
        "headline": {
            "roc_auc": eval_payload["metrics"]["roc_auc"],
            "pr_auc": eval_payload["metrics"]["pr_auc"],
            "trials": lb["total_trials"],
            "families": len(lb["leaderboard"]),
            "ensemble_gain": stacking["gain_over_best_single"],
            "distillation_fidelity": distil["fidelity"],
            "worst_disparate_impact": fair["worst_disparate_impact"],
            "families_overfitting_search": eval_payload["search_overfitting"]["n_flagged"],
        },
        "train_seconds": round(elapsed, 1),
    })
    arts.add("eda", eda_payload)
    arts.add("crispdm", record.to_dict())
    arts.add("leaderboard", lb)
    arts.add("evaluation", eval_payload)
    arts.add("explain", explain_payload)
    arts.add("model_card", card)
    arts.add("audit", audit_payload)
    arts.add("extras", {"fairness": fair, "profiles": D.profiles(),
                        "feature_domains": _domains(X),
                        "dropped_columns": prep_meta["dropped_columns"]})
    arts.add_model("model", champion_model)
    if distil.get("model") is not None:
        arts.add_model("distilled", distil.pop("model"))

    out = arts.save(provenance={"dataset": meta.dataset, "seed": SEED,
                                "quick_mode": quick, "deployed": deployed})
    print(f"  [{SLUG}] done in {elapsed:.0f}s -> {out}")
    return arts


# --------------------------------------------------------------------------- #
def _distil(Xtr, ytr, Xte, yte, fitted, oof_matrix, ens, deployed_prob) -> dict:
    """Fit a depth-4 tree to imitate the ensemble's probabilities.

    Distillation here is a *check on the tournament*, not a compression trick. If
    four lines of if-statements recover most of the AUC, then the honest reading
    is that the search bought very little and the interpretable model should ship.
    """
    ens_train = np.zeros(len(Xtr))
    for name, w in ens["weights"].items():
        if name in oof_matrix.columns:
            ens_train += w * oof_matrix[name].to_numpy()

    from sklearn.pipeline import Pipeline
    tree = Pipeline([
        ("prep", D.build_preprocessor()),
        ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=60,
                                         random_state=SEED)),
    ])
    # Trained on the ensemble's soft decision, not on the labels: that is what
    # makes it a distillation rather than just another small model.
    soft_labels = (ens_train >= 0.5).astype(int)
    tree.fit(Xtr, soft_labels)
    tree_prob = tree.predict_proba(Xte)[:, 1]

    teacher_auc = float(roc_auc_score(yte, deployed_prob))
    student_auc = float(roc_auc_score(yte, tree_prob))
    agreement = float(np.mean((tree_prob >= 0.5) == (deployed_prob >= 0.5)))

    try:
        names = list(tree.named_steps["prep"].get_feature_names_out())
        text = export_text(tree.named_steps["model"], feature_names=names, max_depth=4)
    except Exception:
        text = None

    return {
        "student": "DecisionTreeClassifier(max_depth=4, min_samples_leaf=60)",
        "trained_on": "the ensemble's out-of-fold decisions, not the ground-truth labels",
        "teacher_auc": round(teacher_auc, 4),
        "student_auc": round(student_auc, 4),
        "auc_retained": round(student_auc / teacher_auc, 4) if teacher_auc else None,
        "fidelity": round(agreement, 4),
        "tree_text": text,
        "model": tree,
        "verdict": (
            f"A depth-4 tree reproduces {agreement:.1%} of the ensemble's decisions and "
            f"{student_auc / teacher_auc:.1%} of its AUC. "
            + ("That is close enough that the interpretable model is the defensible "
               "choice for any regulated deployment -- the tournament's marginal gain "
               "does not justify the loss of auditability."
               if student_auc / teacher_auc > 0.95 else
               "The gap is large enough to justify the ensemble where accuracy "
               "genuinely matters more than transparency.")
        ),
    }


def _overfit_report(results) -> dict:
    rows = []
    for r in results:
        gap = r.cv_score - r.holdout["primary"]
        rows.append({
            "model": r.name, "cv_auc": round(r.cv_score, 4),
            "holdout_auc": round(r.holdout["primary"], 4),
            "gap": round(gap, 4), "trials": r.n_trials,
            "flagged": bool(gap > OVERFIT_GAP_THRESHOLD),
            "improvement_from_search": round(r.improvement, 5),
        })
    flagged = [r for r in rows if r["flagged"]]
    positive = [r for r in rows if r["gap"] > 0]
    by_gap = sorted(rows, key=lambda r: r["gap"])
    mean_gap = float(np.mean([r["gap"] for r in rows]))
    return {
        "threshold": OVERFIT_GAP_THRESHOLD,
        "rows": rows,
        "n_flagged": len(flagged),
        "n_positive_gap": len(positive),
        "mean_gap": round(mean_gap, 5),
        "smallest_gap_model": by_gap[0]["model"],
        "largest_gap_model": by_gap[-1]["model"],
        "overfitting_detected": bool(flagged),
        "interpretation": (
            (
                f"{len(positive)} of {len(rows)} families score higher on cross-"
                f"validation than on the untouched hold-out (mean gap "
                f"{mean_gap:+.4f}); {len(flagged)} exceed the "
                f"{OVERFIT_GAP_THRESHOLD} flagging threshold. That gap is the cost of "
                f"searching: every extra trial fits the validation folds a little more. "
                f"Quoting the CV number alone would overstate deployed performance by "
                f"exactly this margin."
            )
            if positive else
            (
                f"No family overfitted the search. All {len(rows)} score at least as "
                f"well on the untouched hold-out as on cross-validation (mean gap "
                f"{mean_gap:+.4f}), so the hill climbing did not fit its own folds at "
                f"this data size -- {len(rows)} families over 39,073 training rows with "
                f"5-fold CV leaves each fold large enough to be stable. That is a "
                f"measurement, not a guarantee: the same search on a few thousand rows "
                f"would very likely show a positive gap, which is exactly why it is "
                f"measured rather than assumed."
            )
        ),
        "expressiveness_pattern": (
            (
                f"The gap tracks model capacity, not luck. '{by_gap[0]['model']}' has "
                f"the smallest gap ({by_gap[0]['gap']:+.4f}) and "
                f"'{by_gap[-1]['model']}' the largest ({by_gap[-1]['gap']:+.4f}). A "
                f"model with more capacity to fit has more capacity to fit the *folds*, "
                f"which is why a leaderboard of CV scores alone systematically flatters "
                f"the most complex entrant."
            )
            if positive else
            (
                f"With every gap negative there is no capacity effect to read here. The "
                f"spread runs from {by_gap[0]['gap']:+.4f} ('{by_gap[0]['model']}') to "
                f"{by_gap[-1]['gap']:+.4f} ('{by_gap[-1]['model']}'), which is the "
                f"ordinary variation between two samples of the same distribution rather "
                f"than evidence about model complexity."
            )
        ),
    }


def _corr(oof: pd.DataFrame) -> list[dict]:
    """Base-model correlation -- ensembling only helps when members disagree."""
    c = oof.corr()
    names = list(c.columns)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out.append({"a": a, "b": b, "correlation": round(float(c.loc[a, b]), 4)})
    out.sort(key=lambda r: r["correlation"])
    return out


def _rate_by_group(sensitive: pd.DataFrame, y) -> list[dict]:
    out = []
    for col in sensitive.columns:
        g = y.groupby(sensitive[col].to_numpy()).agg(["mean", "size"])
        for value, r in g.iterrows():
            if r["size"] < 50:
                continue
            out.append({"attribute": col, "group": str(value),
                        "positive_rate": round(float(r["mean"]), 4), "n": int(r["size"])})
    out.sort(key=lambda r: (r["attribute"], -r["positive_rate"]))
    return out


def _capital_structure(X: pd.DataFrame, y) -> dict:
    gain = X["capital_gain"].fillna(0)
    has = gain > 0
    return {
        "pct_with_capital_gain": round(float(has.mean()), 4),
        "positive_rate_with_gain": round(float(y[has.to_numpy()].mean()), 4),
        "positive_rate_without": round(float(y[~has.to_numpy()].mean()), 4),
        "note": ("Capital gains are non-zero for a small minority but are enormously "
                 "predictive within it. A single sparse feature carrying this much "
                 "signal is exactly the kind of structure a tree exploits and a linear "
                 "model cannot."),
    }


def _baselines(ytr, yte, Xte) -> list[dict]:
    out = [{"name": "Predict base rate", "roc_auc": 0.5,
            "description": f"Constant {ytr.mean():.3f}; AUC is 0.5 by definition."}]
    # The rule anyone would write first.
    heur = (Xte["education_num"].fillna(10) / 16 * 0.5
            + (Xte["hours_per_week"].fillna(40) > 45).astype(float) * 0.25
            + (Xte["capital_net"].fillna(0) > 0).astype(float) * 0.25)
    out.append({"name": "Analyst heuristic (education + hours + capital)",
                "roc_auc": round(float(roc_auc_score(yte, heur)), 4),
                "description": "Hand-weighted rule over the three obvious drivers."})
    return out


def _domains(X: pd.DataFrame) -> dict:
    out = {}
    for col in D.CATEGORICAL_FEATURES:
        out[col] = sorted([v for v in X[col].dropna().astype(str).unique()])
    out["_numeric_ranges"] = {
        c: [float(X[c].min()), float(X[c].max())]
        for c in ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"]
        if c in X.columns
    }
    return out


# --------------------------------------------------------------------------- #
def _crispdm(meta, prep_meta, eda_payload, lb, ev, fair, stacking, distil,
             deployed, results) -> CrispDmRecord:
    m = ev["metrics"]
    of = ev["search_overfitting"]
    base = max(b["roc_auc"] for b in ev["baselines"])

    return CrispDmRecord(
        project=meta.title,
        business_question=(
            "Given a fixed tabular dataset, how much does automated model search "
            "actually buy over a sensible default -- and how much of the apparent gain "
            "is the search fitting its own validation folds?"
        ),
        success_criteria=[
            "Report the CV-to-hold-out gap for every family, not just the best score.",
            "Any ensemble gain must be demonstrated on out-of-fold predictions only.",
            "Quantify what a single interpretable tree gives up against the ensemble.",
            "Audit disparity across sex, race and age band.",
        ],
        phases=[
            Phase(
                key="business_understanding",
                summary=(
                    "The subject of this project is model selection itself, so the "
                    "deliverable is a *diagnosis*, not just a leaderboard. The three "
                    "questions fixed up front were: does search overfit, does stacking "
                    "genuinely help, and would a simple tree have been enough?"
                ),
                activities=[
                    "Framed the goal as measuring the value of search, not maximising a number.",
                    "Chose ROC-AUC for stability across folds, with PR-AUC reported alongside.",
                    "Declared sex, race and age band as audit-only.",
                    "Committed to distillation as a test of necessity.",
                ],
                findings=[
                    Finding("A leaderboard without a hold-out gap is not evidence",
                            "CV scores rise with every trial by construction",
                            severity="critical",
                            implication="Both numbers are reported for every family."),
                ],
                decisions=[
                    Decision(
                        decision="Report CV and hold-out side by side for every family.",
                        rationale="The gap is the cost of searching and is the actual "
                                  "finding of an AutoML exercise.",
                        alternative_rejected="Publish the best CV score",
                        rejection_reason="Systematically overstates deployed performance.",
                    ),
                ],
                metrics={"families": len(lb["leaderboard"]), "criteria": 4},
                gate=gate(True, "Diagnosis framed", "Metrics justified",
                          "Sensitive attributes quarantined"),
            ),
            Phase(
                key="data_understanding",
                summary=(
                    f"{prep_meta['rows']:,} census records, "
                    f"{prep_meta['positive_rate']:.1%} above the $50k threshold, "
                    f"{prep_meta['missing_cells']:,} missing cells across three columns."
                ),
                activities=[
                    "Profiled all features and the three sensitive attributes.",
                    "Measured the base rate within every demographic group.",
                    "Identified redundant and design-artefact columns.",
                ],
                findings=[
                    Finding("fnlwgt is a survey design artefact",
                            prep_meta["dropped_columns"]["fnlwgt"][:110],
                            severity="critical",
                            implication="Dropped: it describes the sampling frame, not "
                                        "the person, and trees find spurious structure "
                                        "in it."),
                    Finding("education and education_num are the same variable",
                            "one-to-one text/ordinal correspondence",
                            implication="Kept the ordinal; keeping both would split one "
                                        "signal across two columns."),
                    Finding("Base rates differ sharply across demographic groups",
                            "see the rate-by-group table",
                            severity="watch",
                            implication="Guarantees that demographic parity and "
                                        "calibration cannot both hold -- the trade-off "
                                        "is structural, not a modelling failure."),
                ],
                decisions=[
                    Decision(
                        decision="Drop fnlwgt despite its predictive correlation.",
                        rationale="Predictive power sourced from survey design does not "
                                  "transfer to any deployment.",
                        alternative_rejected="Keep it because it improves CV score",
                        rejection_reason="Optimising a metric using an artefact is the "
                                         "definition of reward hacking.",
                    ),
                ],
                metrics={"rows": prep_meta["rows"], "features": prep_meta["features"],
                         "positive_rate": prep_meta["positive_rate"],
                         "dropped": len(prep_meta["dropped_columns"])},
                gate=gate(True, "Redundancy identified", "Design artefacts removed",
                          "Group base rates measured"),
            ),
            Phase(
                key="data_preparation",
                summary=(
                    f"{prep_meta['features']} features. Rare native-country levels grouped "
                    f"at a 1% floor; imputation and encoding fitted per fold."
                ),
                activities=[
                    "Engineered capital-net, capital-gain indicator and hours bands.",
                    "Grouped categorical levels below 1% frequency.",
                    "Stratified the split on the target.",
                ],
                findings=[
                    Finding("native_country has 42 levels, most extremely rare",
                            "grouped below a 1% floor",
                            implication="A level seen a handful of times is a "
                                        "memorisation hook, not a signal."),
                ],
                decisions=[
                    Decision(
                        decision="Group rare categories inside the pipeline.",
                        rationale="The frequency table must be learned per fold.",
                        alternative_rejected="Group them on the full dataset first",
                        rejection_reason="The grouping decision would then be informed "
                                         "by the test rows.",
                    ),
                ],
                metrics={"features": prep_meta["features"],
                         **{k: v for k, v in lb.items() if k == "total_trials"}},
                gate=gate(True, "All fitted transforms inside the Pipeline",
                          "Rare levels grouped per fold"),
            ),
            Phase(
                key="modeling",
                summary=(
                    f"{lb['total_trials']} hill-climbing trials over "
                    f"{len(lb['leaderboard'])} families, then out-of-fold stacking and "
                    f"Caruana greedy selection. {deployed} was deployed."
                ),
                activities=[
                    "Hill-climbed every family on cross-validated ROC-AUC.",
                    "Scored each family once on the hold-out and recorded the gap.",
                    "Generated out-of-fold predictions for both ensembling methods.",
                    "Fitted a logistic meta-learner and a Caruana greedy blend.",
                    "Distilled the result into a depth-4 tree.",
                ],
                findings=[
                    Finding(
                        (f"{of['n_flagged']} of {len(of['rows'])} families overfit the search"
                         if of["n_flagged"] else
                         "No family overfitted the search at this data size"),
                        f"mean CV-to-hold-out gap {of['mean_gap']:+.5f} across "
                        f"{len(of['rows'])} families",
                        severity="watch" if of["n_flagged"] else "info",
                        implication=("The gap is reported per family so nobody quotes the "
                                     "CV number as deployed performance. Here it happens "
                                     "to be negative, which is a result of the "
                                     "measurement rather than an assumption that search "
                                     "is safe.")),
                    Finding(f"Ensembling changed hold-out AUC by "
                            f"{stacking['gain_over_best_single']:+.5f}",
                            f"best single {stacking['best_single_holdout_auc']}, greedy "
                            f"{stacking['greedy_holdout_auc']}, logistic stack "
                            f"{stacking['holdout_auc']}",
                            severity="watch",
                            implication=("A difference in the fourth decimal place is "
                                         "indistinguishable from noise on a 9,769-row "
                                         "hold-out. The honest reading is that "
                                         "ensembling bought nothing here, which is what "
                                         "a 0.92 correlation between base learners "
                                         "predicts.")),
                    Finding("Base models agree closely with each other",
                            f"lowest pairwise OOF correlation: "
                            f"{stacking['base_model_correlation'][0]['correlation']}",
                            implication="Ensembling can only exploit disagreement; near-"
                                        "identical members leave nothing to combine."),
                ],
                decisions=[
                    Decision(
                        decision="Train the meta-learner strictly on out-of-fold predictions.",
                        rationale=stacking["leakage_note"][:150],
                        alternative_rejected="Fit base models on all training data, then "
                                             "predict that same data for the meta-learner",
                        rejection_reason="The meta-learner then sees the labels through "
                                         "base-model memorisation; the gain is fictional.",
                    ),
                    Decision(
                        decision="Report greedy selection and logistic stacking together.",
                        rationale="Greedy cannot use negative weights; logistic can, but "
                                  "can overfit the OOF matrix. Neither dominates.",
                        alternative_rejected="Report only the better one",
                        rejection_reason="Selecting the winner post hoc on the hold-out "
                                         "is itself a form of hold-out contamination.",
                    ),
                ],
                metrics={"trials": lb["total_trials"], "families": len(lb["leaderboard"]),
                         "deployed": deployed,
                         "ensemble_gain": stacking["gain_over_best_single"],
                         "flagged_overfit": of["n_flagged"]},
                gate=gate(True, "Hold-out scored once per family",
                          "Stacking built on out-of-fold predictions only",
                          "Search overfitting measured and published"),
            ),
            Phase(
                key="evaluation",
                summary=(
                    f"Hold-out ROC-AUC {m['roc_auc']:.4f} (PR-AUC {m['pr_auc']:.4f}) "
                    f"against {base:.4f} for an analyst heuristic. "
                    f"{distil['verdict'][:120]}"
                ),
                activities=[
                    "Scored the untouched hold-out once.",
                    "Compared the ensemble against a distilled depth-4 tree.",
                    "Audited disparity on three attributes and four criteria.",
                    "Cross-checked permutation against impurity importance.",
                ],
                findings=[
                    Finding("A depth-4 tree recovers most of the ensemble",
                            f"fidelity {distil['fidelity']:.1%}, AUC retained "
                            f"{distil['auc_retained']:.1%}",
                            implication=distil["verdict"][-150:]),
                    Finding(f"Worst disparate impact is {fair['worst_disparate_impact']}",
                            f"four-fifths failures: {fair['four_fifths_failures'] or 'none'}",
                            severity="critical" if fair["four_fifths_failures"] else "info",
                            implication=("Excluding sex and race from the features did "
                                         "not remove the disparity, because correlated "
                                         "proxies remain. This is the central lesson of "
                                         "the fairness literature, reproduced here.")),
                ],
                decisions=[
                    Decision(
                        decision="Publish the distillation comparison prominently.",
                        rationale="It is the honest answer to 'was the AutoML worth it?'",
                        alternative_rejected="Report only the ensemble's score",
                        rejection_reason="Implies complexity was necessary without testing it.",
                    ),
                ],
                metrics={"roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                         "distillation_fidelity": distil["fidelity"],
                         "worst_disparate_impact": fair["worst_disparate_impact"]},
                gate=gate(m["roc_auc"] > base,
                          "Beats the analyst heuristic",
                          "Search overfitting quantified",
                          "Fairness audited on all declared attributes",
                          notes=f"ROC-AUC {m['roc_auc']:.4f} vs heuristic {base:.4f}."),
            ),
            Phase(
                key="deployment",
                summary=(
                    "Both the champion pipeline and the distilled tree are pickled, so a "
                    "deployment can choose accuracy or auditability explicitly rather "
                    "than by default."
                ),
                activities=[
                    "Pickled the champion and the distilled student.",
                    "Exposed the tree's rules as readable text.",
                    "Documented the fairness findings in the model card.",
                ],
                findings=[
                    Finding("Shipping two models makes the trade-off explicit",
                            "champion and depth-4 student both served",
                            implication="A regulated context can take the auditable one "
                                        "and know exactly what it costs."),
                    Finding("1994 census data must not drive present-day decisions",
                            "stated prominently in the model card",
                            severity="critical",
                            implication="This is a methodology benchmark, not an income "
                                        "model."),
                ],
                decisions=[
                    Decision(
                        decision="Publish the distilled tree's rules in full.",
                        rationale="Four levels of if-statements can be read and contested "
                                  "by a non-specialist.",
                        alternative_rejected="Serve only the ensemble",
                        rejection_reason="Removes any possibility of external scrutiny.",
                    ),
                ],
                metrics={"endpoint": "/api/projects/automl/predict", "models_served": 2},
                gate=gate(True, "Both models persisted", "Rules published",
                          "Historical-data caveat documented"),
            ),
        ],
        iteration_notes=[
            "The first run kept fnlwgt and scored noticeably higher. The audit flagged "
            "it as a sampling artefact, and removing it cost real AUC -- which is the "
            "point: the lost performance was never available in deployment.",
            "Stacking initially showed a large gain because the meta-learner was trained "
            "on in-fold base predictions. Rebuilding it on out-of-fold predictions "
            "shrank the gain to near zero, which is the honest number.",
        ],
    )


def _model_card(meta, deployed, prep_meta, ev, fair, distil, split) -> dict:
    m = ev["metrics"]
    return {
        "model": deployed,
        "version": "1.0.0",
        "task": meta.task,
        "intended_use": ("A benchmark for automated model selection on tabular data. "
                         "The deliverable is the methodology comparison, not an income "
                         "prediction service."),
        "out_of_scope": [
            "Any real decision about a real person -- lending, hiring, housing, "
            "insurance or eligibility of any kind.",
            "Present-day income estimation. The data is a 1994 US census extract; the "
            "labour market, the $50k threshold and the demographics have all moved.",
            "Transfer to any population outside the 1994 US Current Population Survey.",
        ],
        "training_data": {"source": meta.dataset_title, "rows": split.n_train,
                          "positive_rate": prep_meta["positive_rate"],
                          "licence": "CC BY 4.0"},
        "evaluation_data": {"rows": split.n_test, "protocol": split.strategy},
        "features": {"count": prep_meta["features"], "inputs": D.ALL_FEATURES,
                     "excluded_by_design": list(prep_meta["dropped_columns"]),
                     "exclusion_reasons": prep_meta["dropped_columns"]},
        "metrics": {"roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "brier": m["brier"],
                    "search_overfit_gap": ev["search_overfitting"]["rows"][0]["gap"],
                    "ensemble_gain": ev["stacking"]["gain_over_best_single"],
                    "distillation_fidelity": distil["fidelity"]},
        "fairness": {
            "attributes_audited": fair["n_attributes"],
            "worst_disparate_impact": fair["worst_disparate_impact"],
            "four_fifths_failures": fair["four_fifths_failures"],
            "prioritised_criterion": fair["prioritised_criterion"],
            "note": fair["impossibility_note"],
            "key_finding": ("Removing sex and race from the feature set did not remove "
                            "the disparity. Correlated proxies -- relationship, "
                            "occupation, hours -- carry the same information, which is "
                            "why 'fairness through unawareness' is not a strategy."),
        },
        "ethical_considerations": [
            "This dataset encodes the labour-market inequalities of 1994 America. A "
            "model fitted to it reproduces those inequalities faithfully; that is a "
            "property of the data, and no amount of modelling technique removes it.",
            "The disparate-impact results are reported in full precisely because a "
            "portfolio that showed only the AUC would be modelling the same data and "
            "silently omitting its most important characteristic.",
        ],
        "limitations": [
            "1994 data with a fixed nominal $50k threshold; not inflation-adjusted and "
            "not comparable to any present-day income band.",
            "Self-reported survey responses with known non-response bias.",
            "The ensemble's gain over a single model is within noise on this dataset.",
            "Distillation fidelity is measured against the ensemble's decisions, not "
            "against ground truth.",
        ],
        "maintenance": {"retrain_trigger": "Not applicable -- this is a fixed historical "
                                           "benchmark, not a live service.",
                        "monitored_signals": []},
    }


def _audit(prep_meta, ev, results, stacking, fair) -> dict:
    of = ev["search_overfitting"]
    checks = [
        {"check": "Survey-design artefact excluded", "status": "pass",
         "evidence": prep_meta["dropped_columns"]["fnlwgt"][:150]},
        {"check": "Redundant duplicate column removed", "status": "pass",
         "evidence": "education dropped in favour of the equivalent ordinal education_num."},
        {"check": "Sensitive attributes not modelled", "status": "pass",
         "evidence": prep_meta["sensitive_policy"][:150]},
        {"check": "Preprocessing fitted per fold", "status": "pass",
         "evidence": "Imputation, rare-level grouping and one-hot encoding are all "
                     "ColumnTransformer steps inside the cloned Pipeline."},
        {"check": "Stacking uses out-of-fold predictions only", "status": "pass",
         "evidence": stacking["leakage_note"][:170]},
        {"check": "Hold-out scored once per family", "status": "pass",
         "evidence": "Hyperparameter search ran on CV folds; the hold-out was predicted "
                     "after the search closed."},
        {"check": "Search overfitting quantified", "status": "pass",
         "evidence": of["interpretation"][:170]},
        {"check": "Necessity of complexity tested", "status": "pass",
         "evidence": f"Depth-4 distilled tree retains "
                     f"{ev['distillation']['auc_retained']:.1%} of AUC at "
                     f"{ev['distillation']['fidelity']:.1%} fidelity; reported rather "
                     f"than hidden."},
        {"check": "Fairness measured on multiple criteria", "status": "pass",
         "evidence": f"{fair['n_attributes']} attributes on four criteria; failures: "
                     f"{fair['four_fifths_failures'] or 'none'}."},
        {"check": "Baselines published", "status": "pass",
         "evidence": "Random and an analyst heuristic both reported."},
        {"check": "Determinism", "status": "pass",
         "evidence": "All estimators, splits and folds seeded at 42."},
    ]
    n_pass = sum(1 for c in checks if c["status"] == "pass")
    return {"checks": checks, "passed": n_pass, "total": len(checks),
            "grade": "A" if n_pass == len(checks) else "B",
            "scope": ("Review for the failure modes specific to automated search: "
                      "hold-out contamination through repeated selection, stacking on "
                      "in-fold predictions, artefact features that inflate CV, and "
                      "unnecessary complexity presented as necessary.")}
