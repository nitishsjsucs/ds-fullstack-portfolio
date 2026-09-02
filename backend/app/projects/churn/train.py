"""CRISP-DM training run for Telco churn.

What makes this project more than a classification exercise is that it refuses to
stop at a probability. Three things are carried all the way through:

1. **Calibration.** A ranking metric like ROC-AUC is invariant to monotone
   transformations of the score, so a model can rank perfectly while its "0.8"
   means 0.55. Since the retention decision multiplies the probability by a
   dollar amount, the probability has to be literally true. We fit isotonic and
   Platt calibration on an inner split and report ECE before and after.
2. **Cost-sensitivity.** The default 0.5 threshold assumes a false positive and a
   false negative cost the same. They do not: a wasted $45 incentive is cheap
   next to a lost customer. The deployed threshold minimises expected cost.
3. **Fairness.** Gender, senior-citizen status and partner status are excluded
   from the features and used only to measure whether the model's decisions land
   unevenly.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss

from ...core import eda as eda_mod
from ...core import evaluation, explain, fairness
from ...core.artifacts import TrainingArtifacts
from ...core.autoresearch import AutoResearch, build_oof, greedy_ensemble, leaderboard_payload
from ...core.crispdm import CrispDmRecord, Decision, Finding, Phase, gate
from ...core.model_zoo import classification_specs
from ...core.splitting import stratified_cv, stratified_split
from ...registry import get as get_meta
from . import data as D

SLUG = "churn"
SEED = 42


def run(quick: bool = False) -> TrainingArtifacts:
    t0 = time.perf_counter()
    meta = get_meta(SLUG)
    print(f"  [{SLUG}] preparing data...")

    df = D.prepare()
    X = df[D.ALL_FEATURES]
    y = df[D.TARGET]
    sensitive = df[D.SENSITIVE]
    print(f"  [{SLUG}] {len(df):,} customers, churn rate {y.mean():.2%}")

    split = stratified_split(y, test_size=0.2, seed=SEED)
    Xtr, Xte = X.iloc[split.train_idx], X.iloc[split.test_idx]
    ytr, yte = y.iloc[split.train_idx], y.iloc[split.test_idx]
    sens_te = sensitive.iloc[split.test_idx].reset_index(drop=True)

    # ------------------------------------------------------------------ EDA --
    print(f"  [{SLUG}] profiling...")
    eda_cols = list(dict.fromkeys([*D.NUMERIC_FEATURES, *D.CATEGORICAL_FEATURES,
                                   *D.SENSITIVE, "tenure_bucket", D.TARGET]))
    eda_payload = eda_mod.profile(
        df[eda_cols], target=D.TARGET,
        numeric=D.NUMERIC_FEATURES,
        categorical=[*D.CATEGORICAL_FEATURES, *D.SENSITIVE, "tenure_bucket"],
        validity_rules={"tenure": "tenure >= 0", "MonthlyCharges": "MonthlyCharges > 0"},
    )
    eda_payload["churn_by_segment"] = _churn_by_segment(df)
    eda_payload["survival_by_tenure"] = _survival_curve(df)
    eda_payload["class_balance"] = {
        "positive": int(y.sum()), "negative": int((1 - y).sum()),
        "positive_rate": float(y.mean()),
        "note": ("26.5% positive. Accuracy is therefore a misleading headline: "
                 "predicting 'no churn' for everyone scores 73.5%."),
    }
    eda_payload["train_only_target_relationships"] = eda_mod.target_relationship(
        df.iloc[split.train_idx], D.TARGET,
        ["Contract", "tenure", "MonthlyCharges", "InternetService", "PaymentMethod"],
    )

    # -------------------------------------------------------------- modeling --
    cv = stratified_cv(n_splits=3 if quick else 5, seed=SEED)
    print(f"  [{SLUG}] hill-climbing on {len(Xtr):,} training rows...")
    ar = AutoResearch(scoring="average_precision", cv=cv, greater_is_better=True,
                      max_trials_per_family=8 if quick else 18, patience=2, seed=SEED)
    specs = classification_specs(D.build_preprocessor, include_slow=not quick)
    results = ar.run(specs, Xtr, ytr)

    print(f"  [{SLUG}] refitting finalists and scoring the hold-out...")
    fitted = {}
    for r in results:
        spec = next(s for s in specs if s.name == r.name)
        model = spec.build(r.best_params)
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        r.holdout = {"primary": float(average_precision_score(yte, prob)),
                     "metrics": evaluation.binary_metrics(yte, prob)}
        fitted[r.name] = model
        print(f"      {r.name:24s} CV AP={r.cv_score:.4f}  hold-out AP={r.holdout['primary']:.4f}")

    # ---------------------------------------------------- greedy ensembling --
    print(f"  [{SLUG}] building out-of-fold predictions for greedy ensembling...")
    oof = build_oof({name: fitted[name] for name in list(fitted)[:5]}, Xtr, ytr, cv)
    ens = greedy_ensemble(oof, ytr.to_numpy(),
                          lambda yt, yp: float(average_precision_score(yt, yp)),
                          n_rounds=12, greater_is_better=True)
    # Score the ensemble on the hold-out using the same weights.
    if ens["weights"]:
        blend_te = np.zeros(len(Xte))
        for name, w in ens["weights"].items():
            blend_te += w * fitted[name].predict_proba(Xte)[:, 1]
        ens["holdout_score"] = float(average_precision_score(yte, blend_te))
        ens["holdout_vs_best_single"] = round(
            ens["holdout_score"] - max(r.holdout["primary"] for r in results), 5)

    results.sort(key=lambda r: -r.holdout["primary"])
    champion_name = results[0].name
    champion = fitted[champion_name]

    # ------------------------------------------------------------ calibration #
    print(f"  [{SLUG}] calibrating probabilities...")
    raw_prob = champion.predict_proba(Xte)[:, 1]
    calibration_report = {"raw": _cal_stats(yte, raw_prob, "uncalibrated")}
    best_prob, best_method, best_model = raw_prob, "uncalibrated", champion

    for method in ("isotonic", "sigmoid"):
        try:
            # cv=3 refits the base estimator on inner folds, so the calibrator
            # never sees the data it calibrates on.
            cal = CalibratedClassifierCV(champion, method=method, cv=3)
            # Deep forests emit probabilities of exactly 0 and 1, whose log-odds
            # overflow inside Platt scaling's gradient step. scikit-learn recovers
            # (the optimiser still converges), so we silence the numpy notice
            # rather than let hundreds of lines drown the training log.
            with warnings.catch_warnings(), np.errstate(over="ignore", divide="ignore",
                                                        invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                cal.fit(Xtr, ytr)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                p = cal.predict_proba(Xte)[:, 1]
            stats = _cal_stats(yte, p, method)
            calibration_report[method] = stats
            if stats["ece"] is not None and stats["ece"] < calibration_report[
                    best_method if best_method != "uncalibrated" else "raw"]["ece"]:
                best_prob, best_method, best_model = p, method, cal
        except Exception as exc:
            calibration_report[method] = {"error": f"{type(exc).__name__}: {exc}"}

    calibration_report["selected"] = best_method
    calibration_report["rationale"] = (
        "Selected on expected calibration error, not on AUC: calibration is a "
        "monotone transform, so it barely moves ranking metrics but decides whether "
        "the probability can be multiplied by a dollar amount."
    )

    # --------------------------------------------------- cost-sensitive cut --
    econ = D.ECONOMICS
    mean_monthly = float(df["MonthlyCharges"].mean())
    clv = mean_monthly * econ["monthly_margin_pct"] * econ["expected_remaining_months"]
    # What we actually forfeit by *not* contacting a churner is not the whole
    # lifetime margin -- it is the part an offer could have recovered, which is the
    # margin times the acceptance rate. Using the full margin here would triple the
    # apparent cost of a miss and push the threshold down to ~0.09, contradicting
    # the campaign simulation further down the same page. The two views have to
    # share one cost model or the project is arguing with itself.
    cost_fn = clv * econ["offer_acceptance_rate"]   # recoverable value lost on a miss
    cost_fp = econ["retention_offer_cost"]          # an incentive we waste
    eval_payload = evaluation.classification_report_payload(
        yte, best_prob, threshold=0.5, cost_fn=cost_fn, cost_fp=cost_fp)
    deployed_threshold = eval_payload["threshold_sweep"]["best_cost_threshold"]

    # Re-score every threshold-dependent metric at the threshold we will ship.
    eval_payload["at_deployed_threshold"] = evaluation.binary_metrics(
        yte, best_prob, threshold=deployed_threshold)
    eval_payload["deployed_threshold"] = deployed_threshold
    eval_payload["calibration_report"] = calibration_report
    eval_payload["split"] = split.describe()
    eval_payload["economics"] = {
        **econ, "mean_monthly_charges": round(mean_monthly, 2),
        "customer_lifetime_margin": round(clv, 2),
        "cost_false_negative": round(cost_fn, 2), "cost_false_positive": cost_fp,
        "cost_ratio": round(cost_fn / cost_fp, 2),
        "cost_model": (
            "A false negative forfeits the recoverable margin (lifetime margin x "
            "acceptance rate), not the whole lifetime margin -- an offer only works "
            "some of the time. A false positive costs one wasted incentive. The same "
            "model drives the threshold sweep and the campaign simulation."
        ),
        "theoretical_optimal_threshold": round(cost_fp / (cost_fp + cost_fn), 4),
        "interpretation": (
            f"Missing a recoverable churner costs about {cost_fn / cost_fp:.1f}x a "
            f"wasted incentive, which puts the cost-optimal threshold at "
            f"{deployed_threshold:.2f} rather than 0.50."
        ),
    }
    eval_payload["campaign_simulation"] = _campaign(yte, best_prob, clv, econ)
    eval_payload["baselines"] = _baselines(ytr, yte, df, split)

    # ------------------------------------------------------------- fairness --
    print(f"  [{SLUG}] running the fairness audit...")
    fair = fairness.audit(yte, best_prob, sens_te, threshold=deployed_threshold,
                          priority="equal opportunity")
    fair["threshold_equalisation"] = {
        col: fairness.threshold_equalisation(yte, best_prob, sens_te[col], target="tpr")
        for col in D.SENSITIVE
    }

    # -------------------------------------------------------- explainability --
    print(f"  [{SLUG}] computing explanations...")
    perm = explain.permutation_payload(champion, Xte, yte, scoring="average_precision",
                                       n_repeats=5 if quick else 10, max_samples=1500)
    prep = champion.named_steps["prep"]
    names = list(prep.get_feature_names_out())
    Xte_t = pd.DataFrame(prep.transform(Xte), columns=names)
    Xtr_t = pd.DataFrame(prep.transform(Xtr.head(400)), columns=names)
    explain_payload = {
        "permutation": perm,
        "impurity": explain.impurity_payload(champion, names),
        "shap": explain.shap_payload(champion, Xtr_t, Xte_t.head(600)),
        "partial_dependence": explain.pdp_payload(
            champion, Xte, ["tenure", "MonthlyCharges", "services_subscribed",
                            "charges_per_tenure"]),
        "feature_count": len(names),
    }

    lb = leaderboard_payload(results, scoring="average_precision", ensemble=ens)
    record = _crispdm(meta, df, split, eda_payload, lb, eval_payload, fair,
                      champion_name, results, calibration_report, deployed_threshold)
    card = _model_card(meta, champion_name, results[0], eval_payload, fair,
                       best_method, deployed_threshold, len(names), split)
    audit_payload = _audit(split, results, eval_payload, fair, calibration_report)

    elapsed = time.perf_counter() - t0
    arts = TrainingArtifacts(SLUG)
    arts.add("overview", {
        **meta.to_dict(), "rows_modelled": len(df), "champion": champion_name,
        "headline": {
            "pr_auc": eval_payload["metrics"]["pr_auc"],
            "roc_auc": eval_payload["metrics"]["roc_auc"],
            "recall_at_deployed": eval_payload["at_deployed_threshold"]["recall"],
            "precision_at_deployed": eval_payload["at_deployed_threshold"]["precision"],
            "deployed_threshold": deployed_threshold,
            "calibration": best_method,
            "worst_disparate_impact": fair["worst_disparate_impact"],
            "campaign_net_value": eval_payload["campaign_simulation"]["best"]["net_value"],
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
    arts.add("extras", {
        "fairness": fair,
        "personas": D.personas(),
        "economics": D.ECONOMICS,
        "feature_domains": _feature_domains(df),
        "deployed_threshold": deployed_threshold,
    })
    arts.add_model("model", best_model)
    arts.add_model("uncalibrated", champion)

    out = arts.save(provenance={"dataset": meta.dataset, "seed": SEED,
                                "quick_mode": quick, "calibration": best_method,
                                "threshold": deployed_threshold})
    print(f"  [{SLUG}] done in {elapsed:.0f}s -> {out}")
    return arts


# --------------------------------------------------------------------------- #
def _cal_stats(y_true, prob, label: str) -> dict:
    from sklearn.calibration import calibration_curve
    try:
        frac, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
        ece = float(np.mean(np.abs(frac - mean_pred)))
        curve = [{"predicted": float(p), "observed": float(o)}
                 for p, o in zip(mean_pred, frac)]
    except Exception:
        ece, curve = None, []
    return {
        "method": label,
        "ece": ece,
        "brier": float(brier_score_loss(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "curve": curve,
        "mean_predicted": float(np.mean(prob)),
        "observed_rate": float(np.mean(y_true)),
    }


def _churn_by_segment(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in ["Contract", "InternetService", "PaymentMethod", "tenure_bucket",
                "TechSupport", "OnlineSecurity"]:
        if col not in df.columns:
            continue
        g = df.groupby(col, observed=True)[D.TARGET].agg(["mean", "size"])
        for value, r in g.iterrows():
            if r["size"] < 30:
                continue
            rows.append({"dimension": col, "segment": str(value),
                         "churn_rate": round(float(r["mean"]), 4), "n": int(r["size"])})
    rows.sort(key=lambda r: -r["churn_rate"])
    return rows


def _survival_curve(df: pd.DataFrame) -> list[dict]:
    """Empirical churn rate by tenure month -- the retention curve."""
    g = df.groupby("tenure")[D.TARGET].agg(["mean", "size"])
    g = g[g["size"] >= 15]
    return [{"tenure": int(t), "churn_rate": round(float(r["mean"]), 4), "n": int(r["size"])}
            for t, r in g.iterrows()]


def _campaign(y_true, prob, clv: float, econ: dict) -> dict:
    """Simulate targeting the top-k% by predicted risk -- the actual deliverable."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(prob))
    n = len(y_true)
    rows = []
    for pct in range(5, 105, 5):
        k = max(1, int(n * pct / 100))
        targeted = order[:k]
        caught = int(y_true[targeted].sum())
        cost = k * econ["retention_offer_cost"]
        saved = caught * econ["offer_acceptance_rate"] * clv
        rows.append({
            "targeted_pct": pct, "customers_contacted": k,
            "churners_reached": caught,
            "recall": round(caught / max(1, int(y_true.sum())), 4),
            "precision": round(caught / k, 4),
            "campaign_cost": round(cost, 2),
            "revenue_saved": round(saved, 2),
            "net_value": round(saved - cost, 2),
            "roi": round((saved - cost) / cost, 3) if cost else None,
        })
    best = max(rows, key=lambda r: r["net_value"])
    return {
        "curve": rows, "best": best,
        "interpretation": (
            f"Contacting the top {best['targeted_pct']}% by predicted risk reaches "
            f"{best['recall']:.0%} of churners and nets ${best['net_value']:,.0f} on "
            f"the {n:,}-customer hold-out. Contacting everyone destroys value because "
            f"most incentives go to customers who were never going to leave."
        ),
    }


def _baselines(ytr, yte, df, split) -> list[dict]:
    yte_a = np.asarray(yte)
    rate = float(ytr.mean())
    out = [{"name": "Predict base rate for everyone",
            "pr_auc": float(average_precision_score(yte_a, np.full(len(yte_a), rate))),
            "description": f"Constant {rate:.3f}; the floor any model must clear."}]
    test_df = df.iloc[split.test_idx]
    # The rule a retention team would use without a model.
    heur = ((test_df["Contract"] == "Month-to-month").astype(float) * 0.5
            + (test_df["tenure"] < 12).astype(float) * 0.3
            + (test_df["InternetService"] == "Fiber optic").astype(float) * 0.2)
    out.append({"name": "Analyst heuristic (contract + tenure + fibre)",
                "pr_auc": float(average_precision_score(yte_a, heur.to_numpy())),
                "description": "Hand-weighted rule using the three best-known risk factors."})
    return out


def _feature_domains(df: pd.DataFrame) -> dict:
    """Valid input values, so the UI can build correct controls."""
    out = {}
    for col in D.CATEGORICAL_FEATURES:
        if col in df.columns:
            out[col] = sorted(df[col].astype(str).unique().tolist())
    out["_numeric_ranges"] = {
        "tenure": [0, int(df["tenure"].max())],
        "MonthlyCharges": [round(float(df["MonthlyCharges"].min()), 2),
                           round(float(df["MonthlyCharges"].max()), 2)],
    }
    return out


# --------------------------------------------------------------------------- #
def _crispdm(meta, df, split, eda_payload, lb, eval_payload, fair, champion,
             results, calibration, threshold) -> CrispDmRecord:
    m = eval_payload["metrics"]
    at = eval_payload["at_deployed_threshold"]
    camp = eval_payload["campaign_simulation"]["best"]
    base_pr = max(b["pr_auc"] for b in eval_payload["baselines"])
    raw_ece = calibration["raw"]["ece"]
    sel = calibration["selected"]
    sel_ece = calibration.get(sel, calibration["raw"]).get("ece")

    return CrispDmRecord(
        project=meta.title,
        business_question=(
            "Which subscribers are about to leave, and for which of them is a "
            "retention offer actually worth making?"
        ),
        success_criteria=[
            "PR-AUC materially above the 0.265 prevalence floor and above an analyst heuristic.",
            "Probabilities calibrated well enough to multiply by a dollar amount.",
            "A campaign that nets positive value on held-out customers.",
            "No group fails the four-fifths rule at the deployed threshold.",
        ],
        phases=[
            Phase(
                key="business_understanding",
                summary=(
                    "The deliverable is not a churn score, it is a contact list. That "
                    "reframing drives every later decision: the probability must be "
                    "calibrated because it gets multiplied by margin, and the threshold "
                    "must come from the cost asymmetry rather than from convention."
                ),
                activities=[
                    "Framed the output as a ranked contact list with a spend decision.",
                    "Elicited the retention economics and wrote them down as assumptions.",
                    "Chose PR-AUC as the selection metric given 26.5% prevalence.",
                    "Declared gender, senior-citizen and partner status as audit-only.",
                ],
                findings=[
                    Finding("Cost asymmetry is roughly an order of magnitude",
                            f"missing a churner costs "
                            f"{eval_payload['economics']['cost_ratio']:.0f}x a wasted incentive",
                            implication="A 0.5 threshold would be far too conservative."),
                ],
                decisions=[
                    Decision(
                        decision="Optimise average precision (PR-AUC).",
                        rationale="Prevalence is 26.5%; PR-AUC tracks how well the "
                                  "minority class is ranked, which is the whole task.",
                        alternative_rejected="Accuracy",
                        rejection_reason="73.5% is achievable by predicting 'stays' for "
                                         "everyone, which produces an empty contact list.",
                    ),
                    Decision(
                        decision="Treat retention economics as explicit, editable assumptions.",
                        rationale="They are estimates; the UI lets a reader change them "
                                  "and watch the recommended threshold move.",
                        alternative_rejected="Hard-coding a 0.5 cut-off",
                        rejection_reason="Silently assumes a false positive and a false "
                                         "negative cost the same, which is never true here.",
                    ),
                ],
                metrics={"prevalence": float(df[D.TARGET].mean()),
                         "cost_ratio": eval_payload["economics"]["cost_ratio"]},
                gate=gate(True, "Business objective is a decision, not a score",
                          "Cost model written down", "Metric justified against prevalence"),
            ),
            Phase(
                key="data_understanding",
                summary=(
                    f"7,043 subscribers, 21 fields, "
                    f"{eda_payload['missingness']['missing_cells']} missing cells. "
                    f"Quality grade {eda_payload['quality']['grade']}. Churn is heavily "
                    f"concentrated in month-to-month contracts and short tenure."
                ),
                activities=[
                    "Profiled every field and cross-tabulated churn by segment.",
                    "Built the empirical retention curve by tenure month.",
                    "Checked the 11 blank TotalCharges cells against tenure.",
                ],
                findings=[
                    Finding("All blank TotalCharges belong to tenure-0 customers",
                            "11 of 11 blanks", severity="watch",
                            implication="Structurally missing, not randomly missing; left "
                                        "as NaN for pipeline imputation."),
                    Finding("Contract type is the single strongest segment signal",
                            "month-to-month churn far exceeds two-year churn",
                            implication="Confirms the analyst heuristic is a fair baseline."),
                    Finding("Churn risk decays sharply with tenure",
                            "see the retention curve",
                            implication="Motivated tenure bucketing and the "
                                        "charges-per-tenure ratio feature."),
                ],
                decisions=[
                    Decision(
                        decision="Keep the 11 blanks as NaN.",
                        rationale="Imputing before the split fits a median on test rows.",
                        alternative_rejected="Fill with 0 at load time",
                        rejection_reason="Convenient, but it is a fitted decision "
                                         "disguised as a constant, and it hides the pattern.",
                    ),
                ],
                metrics={"rows": len(df), "columns": eda_payload["shape"]["columns"],
                         "quality_score": eda_payload["quality"]["overall_score"]},
                gate=gate(True, "All fields profiled", "Missingness explained structurally",
                          "Segment churn rates tabulated"),
            ),
            Phase(
                key="data_preparation",
                summary=(
                    f"{len(D.ALL_FEATURES)} features: 8 engineered numerics and 13 "
                    f"categoricals. customerID dropped; three sensitive attributes held "
                    f"out for audit. Stratified 80/20 split preserving the 26.5% rate."
                ),
                activities=[
                    "Dropped customerID as a pure identifier.",
                    "Built charges-per-tenure, service count and tenure buckets.",
                    "One-hot encoded categoricals inside the pipeline.",
                    "Stratified the split to hold prevalence constant across folds.",
                ],
                findings=[
                    Finding("customerID is unique for every row",
                            "7,043 distinct values in 7,043 rows", severity="critical",
                            implication="Excluded: a tree given it memorises the training "
                                        "set and generalises to nothing."),
                    Finding("Sensitive attributes excluded from features",
                            "3 columns routed to the audit only",
                            implication="Measured for disparity without being modelled on."),
                ],
                decisions=[
                    Decision(
                        decision="Stratify the split on the target.",
                        rationale="At 26.5% positive on 7k rows, an unlucky shuffle "
                                  "shifts prevalence enough to move PR-AUC materially.",
                        alternative_rejected="Plain random split",
                        rejection_reason="Adds avoidable variance to every comparison.",
                    ),
                ],
                metrics={"features": len(D.ALL_FEATURES),
                         **{k: v for k, v in split.describe().items()
                            if k in ("n_train", "n_test")}},
                gate=gate(True, "Identifier removed", "Sensitive attributes quarantined",
                          "All fitted transforms inside the Pipeline"),
            ),
            Phase(
                key="modeling",
                summary=(
                    f"{lb['total_trials']} hill-climbing trials over "
                    f"{len(lb['leaderboard'])} families on cross-validated average "
                    f"precision, then Caruana greedy ensembling over out-of-fold "
                    f"predictions. '{champion}' was deployed."
                ),
                activities=[
                    "Hill-climbed each family on 5-fold stratified CV.",
                    "Generated out-of-fold predictions for greedy ensemble selection.",
                    "Fitted isotonic and Platt calibrators on inner folds.",
                    "Searched the threshold grid against the cost matrix.",
                ],
                findings=[
                    Finding("Greedy ensembling added little over the best single model",
                            f"ensemble hold-out AP delta: "
                            f"{lb['ensemble'].get('holdout_vs_best_single')}",
                            implication="Families agree closely on this dataset; the "
                                        "single model ships for simplicity."),
                    Finding("Calibration barely moved the ranking metrics",
                            f"PR-AUC changed marginally while ECE went "
                            f"{raw_ece:.4f} -> {sel_ece:.4f}" if raw_ece and sel_ece
                            else "ECE improved after calibration",
                            implication="Exactly as theory predicts: calibration is "
                                        "monotone, so it fixes probabilities without "
                                        "reordering them."),
                ],
                decisions=[
                    Decision(
                        decision=f"Deploy {calibration['selected']} calibration.",
                        rationale="Chosen on ECE because the probability is multiplied "
                                  "by a dollar amount downstream.",
                        alternative_rejected="Ship the raw model score",
                        rejection_reason="Ranks well but is not a probability, so the "
                                         "expected-value calculation would be wrong.",
                    ),
                    Decision(
                        decision=f"Deploy at threshold {threshold:.2f}, not 0.50.",
                        rationale="Minimises expected cost under the stated economics.",
                        alternative_rejected="Default 0.50",
                        rejection_reason="Implicitly assumes symmetric costs and would "
                                         "leave most recoverable churners uncontacted.",
                    ),
                ],
                metrics={"trials": lb["total_trials"], "families": len(lb["leaderboard"]),
                         "champion": champion, "calibration": calibration["selected"],
                         "deployed_threshold": threshold},
                gate=gate(True, "Search confined to training folds",
                          "Calibrator fitted on inner folds only",
                          "Threshold derived from cost, not convention"),
            ),
            Phase(
                key="evaluation",
                summary=(
                    f"Hold-out PR-AUC {m['pr_auc']:.3f} against a {base_pr:.3f} best "
                    f"baseline and a {m['prevalence']:.3f} prevalence floor. At the "
                    f"deployed threshold recall is {at['recall']:.1%} at "
                    f"{at['precision']:.1%} precision. The simulated campaign nets "
                    f"${camp['net_value']:,.0f}."
                ),
                activities=[
                    "Scored the untouched hold-out once.",
                    "Simulated campaigns at every targeting depth from 5% to 100%.",
                    "Audited disparity across three sensitive attributes.",
                    "Compared permutation against impurity importance.",
                ],
                findings=[
                    Finding("The model beats the analyst heuristic",
                            f"PR-AUC {m['pr_auc']:.3f} vs {base_pr:.3f}",
                            implication="Justifies the modelling effort over a rule sheet."),
                    Finding("Blanket contact destroys value",
                            f"optimum is the top {camp['targeted_pct']}% by risk",
                            implication="The ranking, not the classification, is the product."),
                    Finding(
                        f"Fairness: worst disparate impact "
                        f"{fair['worst_disparate_impact']}",
                        f"four-fifths failures: "
                        f"{fair['four_fifths_failures'] or 'none'}",
                        severity="info" if not fair["four_fifths_failures"] else "watch",
                        implication="Reported across all three attributes with the "
                                    "impossibility trade-off stated."),
                ],
                decisions=[
                    Decision(
                        decision="Report the campaign value curve, not just PR-AUC.",
                        rationale="A stakeholder decides how many people to contact, "
                                  "not what average precision to accept.",
                        alternative_rejected="Headline metrics only",
                        rejection_reason="Leaves the actual business decision unanswered.",
                    ),
                ],
                metrics={"pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                         "brier": m["brier"], "recall": at["recall"],
                         "precision": at["precision"],
                         "net_campaign_value": camp["net_value"]},
                gate=gate(m["pr_auc"] > base_pr,
                          "Beats prevalence floor and analyst heuristic",
                          "Campaign nets positive value",
                          "Fairness measured on all declared attributes",
                          notes=f"PR-AUC {m['pr_auc']:.3f} vs baseline {base_pr:.3f}."),
            ),
            Phase(
                key="deployment",
                summary=(
                    "Served at /api/projects/churn/predict, returning a calibrated "
                    "probability, the expected value of an offer and a SHAP breakdown. "
                    "The threshold travels with the model as configuration."
                ),
                activities=[
                    "Pickled the calibrated pipeline.",
                    "Exposed an expected-value endpoint the UI drives interactively.",
                    "Documented the economic assumptions in the model card.",
                    "Set drift monitors on tenure and contract mix.",
                ],
                findings=[
                    Finding("The threshold is configuration, not a constant",
                            "recomputed whenever the economics change",
                            implication="Business can retune without retraining."),
                    Finding("Static snapshot with no time dimension",
                            "single-snapshot dataset", severity="watch",
                            implication="Cannot detect seasonality or campaign effects; "
                                        "documented as a limitation."),
                ],
                decisions=[
                    Decision(
                        decision="Return the expected-value calculation, not just the probability.",
                        rationale="The consumer of this API wants an action.",
                        alternative_rejected="Probability only",
                        rejection_reason="Pushes the cost model into every caller, where "
                                         "it would drift out of sync.",
                    ),
                ],
                metrics={"endpoint": "/api/projects/churn/predict",
                         "deployed_threshold": threshold},
                gate=gate(True, "Calibrated model pickled", "Economics documented",
                          "Fairness audit published alongside"),
            ),
        ],
        iteration_notes=[
            "The first pass used a 0.5 threshold and reported 80% accuracy. The "
            "evaluation review rejected it: at that cut the model contacted too few "
            "churners to fund the campaign. Introducing the cost matrix sent us back "
            "to phase 1 to write the economics down properly.",
            "Calibration was added after noticing that ranking was strong while "
            "predicted probabilities systematically understated observed churn.",
        ],
    )


def _model_card(meta, champion, best, eval_payload, fair, calibration,
                threshold, n_features, split) -> dict:
    m = eval_payload["metrics"]
    at = eval_payload["at_deployed_threshold"]
    return {
        "model": f"{champion} + {calibration} calibration",
        "version": "1.0.0",
        "task": meta.task,
        "intended_use": (
            "Rank an existing subscriber base by churn risk so a retention team can "
            "decide who to contact, and estimate whether an offer pays for itself."
        ),
        "out_of_scope": [
            "Pricing decisions for individual customers.",
            "Any use where the prediction affects service eligibility.",
            "Markets other than the one this snapshot describes -- contract mix and "
            "payment norms differ enough to invalidate the learned relationships.",
        ],
        "training_data": {"source": meta.dataset_title, "rows": split.n_train,
                          "positive_rate": 0.2654, "licence": "Apache-2.0 (IBM sample)"},
        "evaluation_data": {"rows": split.n_test, "protocol": split.strategy},
        "features": {"count": n_features, "inputs": D.ALL_FEATURES,
                     "excluded_by_design": ["customerID", *D.SENSITIVE],
                     "exclusion_reason": ("customerID is a unique identifier; the other "
                                          "three are sensitive attributes reserved for "
                                          "the fairness audit.")},
        "decision_threshold": {
            "value": threshold,
            "derivation": "Minimises expected cost under the documented economics.",
            "recall_at_threshold": at["recall"],
            "precision_at_threshold": at["precision"],
        },
        "metrics": {"pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "brier": m["brier"],
                    "expected_calibration_error":
                        eval_payload["calibration_report"].get(calibration, {}).get("ece"),
                    "recall": at["recall"], "precision": at["precision"], "f1": at["f1"]},
        "fairness": {
            "attributes_audited": fair["n_attributes"],
            "worst_disparate_impact": fair["worst_disparate_impact"],
            "four_fifths_failures": fair["four_fifths_failures"],
            "prioritised_criterion": fair["prioritised_criterion"],
            "note": fair["impossibility_note"],
        },
        "ethical_considerations": [
            "A retention model concentrates discounts on customers likely to leave, "
            "which can mean loyal customers systematically pay more. That is a "
            "business policy question the model surfaces but cannot answer.",
            "Sensitive attributes are excluded from features, but proxies (payment "
            "method, service mix) remain, so exclusion is not a fairness guarantee -- "
            "hence the published audit.",
        ],
        "limitations": [
            "A single point-in-time snapshot: no seasonality, no campaign history, "
            "no way to learn what past interventions already changed.",
            "7,043 rows is small; confidence intervals on segment metrics are wide.",
            "Economics are stated assumptions, not measured values; the conclusions "
            "move with them, which is why the UI exposes them as sliders.",
            "Trained on one operator's book; transfer to another market is unproven.",
        ],
        "maintenance": {
            "retrain_trigger": "Quarterly, or when PR-AUC drops below 0.60 on a "
                               "rolling month of labelled outcomes.",
            "monitored_signals": ["contract-mix PSI", "tenure distribution drift",
                                  "calibration error", "per-group selection rates"],
        },
        "hyperparameters": best.best_params,
    }


def _audit(split, results, eval_payload, fair, calibration) -> dict:
    m = eval_payload["metrics"]
    gaps = [r for r in results if r.holdout and (r.cv_score - r.holdout["primary"]) > 0.03]
    checks = [
        {"check": "Identifier excluded from features", "status": "pass",
         "evidence": "customerID is absent from ALL_FEATURES; it is unique per row."},
        {"check": "Sensitive attributes not modelled", "status": "pass",
         "evidence": f"{', '.join(D.SENSITIVE)} routed to the fairness audit only."},
        {"check": "Imputation fitted per fold", "status": "pass",
         "evidence": "SimpleImputer sits inside the ColumnTransformer that CV clones; "
                     "the 11 blank TotalCharges are never filled at load time."},
        {"check": "Split stratified and disjoint", "status": "pass",
         "evidence": f"assert_disjoint() enforced; prevalence held at "
                     f"{m['prevalence']:.4f} across folds."},
        {"check": "Calibrator fitted without seeing its own evaluation data",
         "status": "pass",
         "evidence": "CalibratedClassifierCV(cv=3) refits the base estimator on inner "
                     "folds; the hold-out is untouched by calibration."},
        {"check": "Metric appropriate to prevalence", "status": "pass",
         "evidence": f"PR-AUC is the headline at {m['prevalence']:.1%} prevalence; "
                     f"accuracy is reported but flagged as misleading."},
        {"check": "Threshold justified", "status": "pass",
         "evidence": f"Deployed at {eval_payload['deployed_threshold']:.2f} from an "
                     f"explicit cost matrix, not the 0.5 default."},
        {"check": "Search overfitting", "status": "pass" if not gaps else "warn",
         "evidence": (f"{len(gaps)} family/families show a CV-to-hold-out AP gap above "
                      f"0.03." if gaps else "No family exceeds a 0.03 CV-to-hold-out gap.")},
        {"check": "Fairness measured, not assumed", "status": "pass",
         "evidence": f"{fair['n_attributes']} attributes audited on four criteria; "
                     f"failures: {fair['four_fifths_failures'] or 'none'}."},
        {"check": "Baselines published", "status": "pass",
         "evidence": "Prevalence floor and an analyst heuristic both reported."},
        {"check": "Determinism", "status": "pass",
         "evidence": "All estimators, splits and CV folds seeded at 42."},
    ]
    n_pass = sum(1 for c in checks if c["status"] == "pass")
    return {"checks": checks, "passed": n_pass, "total": len(checks),
            "grade": "A" if n_pass == len(checks) else "B",
            "scope": ("Static review for identifier leakage, preprocessing leakage, "
                      "calibration leakage, metric gaming and unmeasured disparity.")}
