# Telco Churn & Retention Economics

*Cost-sensitive classification with calibration and a fairness audit on 7,043 subscribers.*

**Task** Supervised binary classification (imbalanced)  ·  **Domain** Subscription retention  ·  **Primary metric** PR-AUC (average precision)

> Generated from the training run of `2026-09-02T18:14:10+00:00` (commit `n/a`, seed 42). Every figure below is read from that run's artifacts.

## Abstract

Cost-sensitive classification with calibration and a fairness audit on 7,043 subscribers. The system follows the full CRISP-DM cycle on 7,043 rows of IBM Telco Customer Churn, selecting XGBoost by hill-climbing search on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: pr auc 0.6604, roc auc 0.848, recall at deployed 0.8209, precision at deployed 0.5142. Hold-out PR-AUC 0.660 against a 0.539 best baseline and a 0.265 prevalence floor. At the deployed threshold recall is 82.1% at 51.4% precision. The simulated campaign nets $11,597. A static leakage audit of the training path passes 11 of 11 checks, and 6 of 6 CRISP-DM phase gates are met.

## 1. Business understanding

**Question.** Which subscribers are about to leave, and for which of them is a retention offer actually worth making?

**Success criteria, fixed before modelling:**

- PR-AUC materially above the 0.265 prevalence floor and above an analyst heuristic.
- Probabilities calibrated well enough to multiply by a dollar amount.
- A campaign that nets positive value on held-out customers.
- No group fails the four-fifths rule at the deployed threshold.

**Why PR-AUC (average precision).** Churn prevalence is 26.5%. Accuracy rewards predicting 'stays' for everyone (73.5%); PR-AUC only rewards ranking real churners highly.

## 2. Data

- **Source** — IBM Telco Customer Churn
- **Rows** — 7,043 in the curated file; 7,043 after preparation
- **Licence** — Apache-2.0 (IBM sample)
- **Quality** — grade A at 99.9% across six dimensions
- **Missingness** — 11 of 183,118 cells (0.01%); 99.8% of rows complete

The curation rule and a SHA-256 for this file are recorded in [`data/MANIFEST.json`](../../../data/MANIFEST.json). Ingestion applies only column selection, dtype coercion, deterministic subsampling and stable sorting — no imputation, scaling or target-aware filtering.

## 3. Data Understanding

7,043 subscribers, 21 fields, 11 missing cells. Quality grade A. Churn is heavily concentrated in month-to-month contracts and short tenure.

**Findings.**

- *All blank TotalCharges belong to tenure-0 customers* — 11 of 11 blanks. Structurally missing, not randomly missing; left as NaN for pipeline imputation.
- *Contract type is the single strongest segment signal* — month-to-month churn far exceeds two-year churn. Confirms the analyst heuristic is a fair baseline.
- *Churn risk decays sharply with tenure* — see the retention curve. Motivated tenure bucketing and the charges-per-tenure ratio feature.

**Decisions.**

- **Keep the 11 blanks as NaN.** Imputing before the split fits a median on test rows. *Rejected:* Fill with 0 at load time — Convenient, but it is a fitted decision disguised as a constant, and it hides the pattern.

## 4. Data Preparation

21 features: 8 engineered numerics and 13 categoricals. customerID dropped; three sensitive attributes held out for audit. Stratified 80/20 split preserving the 26.5% rate.

**Findings.**

- *customerID is unique for every row* — 7,043 distinct values in 7,043 rows. Excluded: a tree given it memorises the training set and generalises to nothing.
- *Sensitive attributes excluded from features* — 3 columns routed to the audit only. Measured for disparity without being modelled on.

**Decisions.**

- **Stratify the split on the target.** At 26.5% positive on 7k rows, an unlucky shuffle shifts prevalence enough to move PR-AUC materially. *Rejected:* Plain random split — Adds avoidable variance to every comparison.

## 5. Modelling

95 hill-climbing trials over 6 families on cross-validated average precision, then Caruana greedy ensembling over out-of-fold predictions. 'XGBoost' was deployed.

**Leaderboard** — scored by average_precision.

| Model | CV | Hold-out | Gap | Trials |
|---|---|---|---|---|
| XGBoost | 0.6578 | 0.6652 | -0.0074 | 19 |
| Extra trees | 0.6564 | 0.6634 | -0.007 | 13 |
| Random forest | 0.6622 | 0.66 | 0.0022 | 19 |
| Logistic regression | 0.6556 | 0.6591 | -0.0034 | 8 |
| LightGBM | 0.651 | 0.659 | -0.008 | 18 |
| Hist gradient boosting | 0.6468 | 0.6474 | -0.0006 | 18 |

> Hyperparameters were selected by coordinate-ascent hill climbing on the cross-validated average_precision. The hold-out set was scored once per family after the search finished and never fed back into selection. 'generalisation_gap' = CV minus hold-out; a large positive gap means the search overfitted the folds.

## 6. Evaluation

Hold-out PR-AUC 0.660 against a 0.539 best baseline and a 0.265 prevalence floor. At the deployed threshold recall is 82.1% at 51.4% precision. The simulated campaign nets $11,597.

**Headline results**

| Measure | Value |
|---|---|
| Pr auc | 0.6604 |
| Roc auc | 0.848 |
| Recall at deployed | 0.8209 |
| Precision at deployed | 0.5142 |
| Deployed threshold | 0.2746 |
| Calibration | isotonic |
| Worst disparate impact | 0.5083 |
| Campaign net value | 11,597 |

**Baselines the model had to beat**

| Baseline | Score | What it does |
|---|---|---|
| Predict base rate for everyone | 0.2654 | Constant 0.265; the floor any model must clear. |
| Analyst heuristic (contract + tenure + fibre) | 0.5393 | Hand-weighted rule using the three best-known risk factors. |

## 7. Deployment

Served at /api/projects/churn/predict, returning a calibrated probability, the expected value of an offer and a SHAP breakdown. The threshold travels with the model as configuration.

Served at `POST /api/projects/churn/predict`. The fitted pipeline is pickled whole, so preprocessing travels with the model and training and serving cannot drift apart.

## 8. Limitations

- A single point-in-time snapshot: no seasonality, no campaign history, no way to learn what past interventions already changed.
- 7,043 rows is small; confidence intervals on segment metrics are wide.
- Economics are stated assumptions, not measured values; the conclusions move with them, which is why the UI exposes them as sliders.
- Trained on one operator's book; transfer to another market is unproven.

**Out of scope**

- Pricing decisions for individual customers.
- Any use where the prediction affects service eligibility.
- Markets other than the one this snapshot describes -- contract mix and payment norms differ enough to invalidate the learned relationships.

**Ethical considerations**

- A retention model concentrates discounts on customers likely to leave, which can mean loyal customers systematically pay more. That is a business policy question the model surfaces but cannot answer.
- Sensitive attributes are excluded from features, but proxies (payment method, service mix) remain, so exclusion is not a fairness guarantee -- hence the published audit.

## 9. Audit

11 of 11 checks pass (grade A). *Scope:* Static review for identifier leakage, preprocessing leakage, calibration leakage, metric gaming and unmeasured disparity.

| Check | Status | Evidence |
|---|---|---|
| Identifier excluded from features | pass | customerID is absent from ALL_FEATURES; it is unique per row. |
| Sensitive attributes not modelled | pass | gender, SeniorCitizen, Partner routed to the fairness audit only. |
| Imputation fitted per fold | pass | SimpleImputer sits inside the ColumnTransformer that CV clones; the 11 blank TotalCharges are never filled at load time. |
| Split stratified and disjoint | pass | assert_disjoint() enforced; prevalence held at 0.2654 across folds. |
| Calibrator fitted without seeing its own evaluation data | pass | CalibratedClassifierCV(cv=3) refits the base estimator on inner folds; the hold-out is untouched by calibration. |
| Metric appropriate to prevalence | pass | PR-AUC is the headline at 26.5% prevalence; accuracy is reported but flagged as misleading. |
| Threshold justified | pass | Deployed at 0.27 from an explicit cost matrix, not the 0.5 default. |
| Search overfitting | pass | No family exceeds a 0.03 CV-to-hold-out gap. |
| Fairness measured, not assumed | pass | 3 attributes audited on four criteria; failures: ['SeniorCitizen', 'Partner']. |
| Baselines published | pass | Prevalence floor and an analyst heuristic both reported. |
| Determinism | pass | All estimators, splits and CV folds seeded at 42. |

**Phase gates:** 6 of 6 passed.

- PASS — 1. Business Understanding
- PASS — 2. Data Understanding
- PASS — 3. Data Preparation
- PASS — 4. Modeling
- PASS — 5. Evaluation: PR-AUC 0.660 vs baseline 0.539.
- PASS — 6. Deployment

## 10. Where this project looped back

CRISP-DM is iterative. These are the points where a later phase sent the work back to an earlier one — normally the part deleted before publication.

- The first pass used a 0.5 threshold and reported 80% accuracy. The evaluation review rejected it: at that cut the model contacted too few churners to fund the campaign. Introducing the cost matrix sent us back to phase 1 to write the economics down properly.
- Calibration was added after noticing that ranking was strong while predicted probabilities systematically understated observed churn.

## Reproduction

```bash
python scripts/fetch_data.py
python scripts/train_all.py --only churn
```

Every estimator, split and sampler is seeded, so a rerun on the same data reproduces this leaderboard exactly.
