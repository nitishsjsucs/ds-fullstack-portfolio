# AutoML Stacking Tournament

*Hill-climbing search and Caruana ensembling over 48,842 census records.*

**Task** AutoML / model selection  ·  **Domain** Income prediction & fairness  ·  **Primary metric** ROC-AUC (CV) with hold-out gap

> Generated from the training run of `2026-09-02T21:16:26+00:00` (commit `n/a`, seed 42). Every figure below is read from that run's artifacts.

## Abstract

Hill-climbing search and Caruana ensembling over 48,842 census records. The system follows the full CRISP-DM cycle on 48,842 rows of UCI Adult / Census Income (1994 CPS), selecting Caruana greedy ensemble by hill-climbing search on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: roc auc 0.9314, pr auc 0.8363, trials 101, families 6. Hold-out ROC-AUC 0.9314 (PR-AUC 0.8363) against 0.7543 for an analyst heuristic. A depth-4 tree reproduces 92.5% of the ensemble's decisions and 93.1% of its AUC. The gap is large enough to justify the A static leakage audit of the training path passes 11 of 11 checks, and 6 of 6 CRISP-DM phase gates are met.

## 1. Business understanding

**Question.** Given a fixed tabular dataset, how much does automated model search actually buy over a sensible default -- and how much of the apparent gain is the search fitting its own validation folds?

**Success criteria, fixed before modelling:**

- Report the CV-to-hold-out gap for every family, not just the best score.
- Any ensemble gain must be demonstrated on out-of-fold predictions only.
- Quantify what a single interpretable tree gives up against the ensemble.
- Audit disparity across sex, race and age band.

**Why ROC-AUC (CV) with hold-out gap.** Model selection needs a ranking metric stable across folds; the CV-to-hold-out gap is reported alongside to expose search overfitting.

## 2. Data

- **Source** — UCI Adult / Census Income (1994 CPS)
- **Rows** — 48,842 in the curated file; 48,842 after preparation
- **Licence** — CC BY 4.0
- **Quality** — grade A at 97.7% across six dimensions
- **Missingness** — 6,465 of 830,314 cells (0.78%); 92.6% of rows complete

The curation rule and a SHA-256 for this file are recorded in [`data/MANIFEST.json`](../../../data/MANIFEST.json). Ingestion applies only column selection, dtype coercion, deterministic subsampling and stable sorting — no imputation, scaling or target-aware filtering.

## 3. Data Understanding

48,842 census records, 23.9% above the $50k threshold, 6,465 missing cells across three columns.

**Findings.**

- *fnlwgt is a survey design artefact* — Census sampling weight -- a survey-design artefact describing how many people the row represents, not a proper. Dropped: it describes the sampling frame, not the person, and trees find spurious structure in it.
- *education and education_num are the same variable* — one-to-one text/ordinal correspondence. Kept the ordinal; keeping both would split one signal across two columns.
- *Base rates differ sharply across demographic groups* — see the rate-by-group table. Guarantees that demographic parity and calibration cannot both hold -- the trade-off is structural, not a modelling failure.

**Decisions.**

- **Drop fnlwgt despite its predictive correlation.** Predictive power sourced from survey design does not transfer to any deployment. *Rejected:* Keep it because it improves CV score — Optimising a metric using an artefact is the definition of reward hacking.

## 4. Data Preparation

13 features. Rare native-country levels grouped at a 1% floor; imputation and encoding fitted per fold.

**Findings.**

- *native_country has 42 levels, most extremely rare* — grouped below a 1% floor. A level seen a handful of times is a memorisation hook, not a signal.

**Decisions.**

- **Group rare categories inside the pipeline.** The frequency table must be learned per fold. *Rejected:* Group them on the full dataset first — The grouping decision would then be informed by the test rows.

## 5. Modelling

101 hill-climbing trials over 6 families, then out-of-fold stacking and Caruana greedy selection. Caruana greedy ensemble was deployed.

**Leaderboard** — scored by roc_auc.

| Model | CV | Hold-out | Gap | Trials |
|---|---|---|---|---|
| LightGBM | 0.9283 | 0.9313 | -0.003 | 21 |
| Hist gradient boosting | 0.928 | 0.9313 | -0.0034 | 21 |
| XGBoost | 0.928 | 0.9308 | -0.0028 | 20 |
| Random forest | 0.92 | 0.9245 | -0.0045 | 17 |
| Extra trees | 0.9097 | 0.9138 | -0.0041 | 12 |
| Logistic regression | 0.9067 | 0.9079 | -0.0011 | 10 |

> Hyperparameters were selected by coordinate-ascent hill climbing on the cross-validated roc_auc. The hold-out set was scored once per family after the search finished and never fed back into selection. 'generalisation_gap' = CV minus hold-out; a large positive gap means the search overfitted the folds.

## 6. Evaluation

Hold-out ROC-AUC 0.9314 (PR-AUC 0.8363) against 0.7543 for an analyst heuristic. A depth-4 tree reproduces 92.5% of the ensemble's decisions and 93.1% of its AUC. The gap is large enough to justify the

**Headline results**

| Measure | Value |
|---|---|
| Roc auc | 0.9314 |
| Pr auc | 0.8363 |
| Trials | 101 |
| Families | 6 |
| Ensemble gain | 0.0001 |
| Distillation fidelity | 0.9249 |
| Worst disparate impact | 0.0203 |
| Families overfitting search | 0 |

**Baselines the model had to beat**

| Baseline | Score | What it does |
|---|---|---|
| Predict base rate | 0.5 | Constant 0.239; AUC is 0.5 by definition. |
| Analyst heuristic (education + hours + capital) | 0.7543 | Hand-weighted rule over the three obvious drivers. |

## 7. Deployment

Both the champion pipeline and the distilled tree are pickled, so a deployment can choose accuracy or auditability explicitly rather than by default.

Served at `POST /api/projects/automl/predict`. The fitted pipeline is pickled whole, so preprocessing travels with the model and training and serving cannot drift apart.

## 8. Limitations

- 1994 data with a fixed nominal $50k threshold; not inflation-adjusted and not comparable to any present-day income band.
- Self-reported survey responses with known non-response bias.
- The ensemble's gain over a single model is within noise on this dataset.
- Distillation fidelity is measured against the ensemble's decisions, not against ground truth.

**Out of scope**

- Any real decision about a real person -- lending, hiring, housing, insurance or eligibility of any kind.
- Present-day income estimation. The data is a 1994 US census extract; the labour market, the $50k threshold and the demographics have all moved.
- Transfer to any population outside the 1994 US Current Population Survey.

**Ethical considerations**

- This dataset encodes the labour-market inequalities of 1994 America. A model fitted to it reproduces those inequalities faithfully; that is a property of the data, and no amount of modelling technique removes it.
- The disparate-impact results are reported in full precisely because a portfolio that showed only the AUC would be modelling the same data and silently omitting its most important characteristic.

## 9. Audit

11 of 11 checks pass (grade A). *Scope:* Review for the failure modes specific to automated search: hold-out contamination through repeated selection, stacking on in-fold predictions, artefact features that inflate CV, and unnecessary complexity presented as necessary.

| Check | Status | Evidence |
|---|---|---|
| Survey-design artefact excluded | pass | Census sampling weight -- a survey-design artefact describing how many people the row represents, not a property of the person. Correlates with geogra |
| Redundant duplicate column removed | pass | education dropped in favour of the equivalent ordinal education_num. |
| Sensitive attributes not modelled | pass | sex, race and age band are excluded from the feature matrix and used only to measure disparity. Exclusion does not make the model fair -- occupation,  |
| Preprocessing fitted per fold | pass | Imputation, rare-level grouping and one-hot encoding are all ColumnTransformer steps inside the cloned Pipeline. |
| Stacking uses out-of-fold predictions only | pass | The meta-learner is trained exclusively on out-of-fold predictions: every base-model probability it sees was produced by a model that did not train on that row. Fitting i |
| Hold-out scored once per family | pass | Hyperparameter search ran on CV folds; the hold-out was predicted after the search closed. |
| Search overfitting quantified | pass | No family overfitted the search. All 6 score at least as well on the untouched hold-out as on cross-validation (mean gap -0.0032), so the hill climbing did not fit its ow |
| Necessity of complexity tested | pass | Depth-4 distilled tree retains 93.1% of AUC at 92.5% fidelity; reported rather than hidden. |
| Fairness measured on multiple criteria | pass | 3 attributes on four criteria; failures: ['sex', 'race', 'age_band']. |
| Baselines published | pass | Random and an analyst heuristic both reported. |
| Determinism | pass | All estimators, splits and folds seeded at 42. |

**Phase gates:** 6 of 6 passed.

- PASS — 1. Business Understanding
- PASS — 2. Data Understanding
- PASS — 3. Data Preparation
- PASS — 4. Modeling
- PASS — 5. Evaluation: ROC-AUC 0.9314 vs heuristic 0.7543.
- PASS — 6. Deployment

## 10. Where this project looped back

CRISP-DM is iterative. These are the points where a later phase sent the work back to an earlier one — normally the part deleted before publication.

- The first run kept fnlwgt and scored noticeably higher. The audit flagged it as a sampling artefact, and removing it cost real AUC -- which is the point: the lost performance was never available in deployment.
- Stacking initially showed a large gain because the meta-learner was trained on in-fold base predictions. Rebuilding it on out-of-fold predictions shrank the gain to near zero, which is the honest number.

## Reproduction

```bash
python scripts/fetch_data.py
python scripts/train_all.py --only automl
```

Every estimator, split and sampler is seeded, so a rerun on the same data reproduces this leaderboard exactly.
