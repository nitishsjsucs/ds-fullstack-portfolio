# NYC Taxi Fare & Duration

*Spatio-temporal gradient boosting on 150,000 real January-2024 yellow-cab trips.*

**Task** Supervised regression (dual target)  ·  **Domain** Urban mobility / pricing  ·  **Primary metric** MAE (minutes)

> Generated from the training run of `2026-09-02T18:06:42+00:00` (commit `n/a`, seed 42). Every figure below is read from that run's artifacts.

## Abstract

Spatio-temporal gradient boosting on 150,000 real January-2024 yellow-cab trips. The system follows the full CRISP-DM cycle on 150,000 rows of NYC TLC Yellow Taxi Trip Records, January 2024, selecting XGBoost by hill-climbing search on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: mae minutes 2.9284, rmse minutes 4.6072, r2 0.8356, fare mae usd 1.737. On the untouched final week the champion achieves 2.93 minutes MAE (R^2 0.836), against 6.29 for the best heuristic -- a 53% reduction in error. The 80% interval covers 77.3% of outcomes. A static leakage audit of the training path passes 10 of 10 checks, and 6 of 6 CRISP-DM phase gates are met.

## 1. Business understanding

**Question.** Can we tell a rider, at the moment they request a yellow cab, how long the trip will take and what it will cost -- accurately enough that they trust the number and plan around it?

**Success criteria, fixed before modelling:**

- Mean absolute duration error under 5 minutes on unseen future days.
- Beat the dispatcher's distance-over-speed heuristic by a wide margin.
- Publish an 80% interval whose empirical coverage is within 5 points of 80%.
- No feature may use information unavailable at the moment of booking.

**Why MAE (minutes).** Riders experience absolute lateness, not squared error, and MAE is robust to the genuine 4-hour outlier trips the TLC publishes.

## 2. Data

- **Source** — NYC TLC Yellow Taxi Trip Records, January 2024
- **Rows** — 150,000 in the curated file; 143,688 after preparation
- **Licence** — NYC Open Data / TLC public domain
- **Quality** — grade A at 100.0% across six dimensions
- **Missingness** — 0 of 3,017,448 cells (0.00%); 100.0% of rows complete

The curation rule and a SHA-256 for this file are recorded in [`data/MANIFEST.json`](../../../data/MANIFEST.json). Ingestion applies only column selection, dtype coercion, deterministic subsampling and stable sorting — no imputation, scaling or target-aware filtering.

## 3. Data Understanding

150,000 trips sampled from the 2.96M the TLC published for January 2024, joined to the official 265-zone dictionary. The data is real, and so are its defects: the quality scorecard grades it A at 100.0%.

**Findings.**

- *A visible minority of trips are physically impossible* — 6,312 rows (4.21%) violate at least one plausibility rule. Removed by stated rule, each exclusion counted and published.
- *Average speed collapses during the weekday peak* — peak-hour mean duration cell: Wed 4:00. Motivated explicit rush-hour and cyclical-time features.
- *Trip distance is strongly but non-linearly related to duration* — Spearman exceeds Pearson across the correlation matrix. Signals that tree ensembles should beat linear models.

**Decisions.**

- **Join the TLC zone lookup and model borough, not LocationID.** 265 arbitrary integers invite memorisation; 6 boroughs carry real geographic meaning. *Rejected:* Raw LocationID as a numeric feature — Implies an ordering between zones that does not exist.

## 4. Data Preparation

143,688 trips survive the eight plausibility rules. Fifteen booking-time features are constructed; every fitted transformation lives inside the model pipeline.

**Findings.**

- *No filter references the target distribution* — 8 of 8 rules are physical bounds. Hard-but-real cases stay in the test set.
- *Preprocessing is re-fitted per fold* — 3 fitted transformers, all inside the Pipeline. Cross-validated scores are not optimistically biased.

**Decisions.**

- **Split chronologically rather than randomly.** Deployment predicts future trips from past ones; the evaluation must mirror that. *Rejected:* Random 80/20 shuffle — Lets the model interpolate within a rush hour it has already seen, inflating the score.
- **Frequency-encode zone IDs instead of one-hot.** Captures 'busy origin' in one column instead of 530. *Rejected:* Target encoding — Target encoding needs careful nested CV to avoid leaking the label; the gain did not justify the risk.

## 5. Modelling

79 hill-climbing trials across 6 model families, scored by expanding-window cross-validated MAE. 'XGBoost' won and was deployed.

**Leaderboard** — scored by neg_mean_absolute_error.

| Model | CV | Hold-out | Gap | Trials |
|---|---|---|---|---|
| XGBoost | -3.3159 | -2.9284 | -0.3876 | 15 |
| LightGBM | -3.3318 | -2.9524 | -0.3795 | 14 |
| Hist gradient boosting | -3.3409 | -2.9902 | -0.3507 | 14 |
| Random forest | -3.5702 | -3.1033 | -0.4669 | 15 |
| Extra trees | -3.5959 | -3.1176 | -0.4783 | 9 |
| Ridge regression | -4.8561 | -4.4117 | -0.4444 | 12 |

> Hyperparameters were selected by coordinate-ascent hill climbing on the cross-validated neg_mean_absolute_error. The hold-out set was scored once per family after the search finished and never fed back into selection. 'generalisation_gap' = CV minus hold-out; a large positive gap means the search overfitted the folds.

## 6. Evaluation

On the untouched final week the champion achieves 2.93 minutes MAE (R^2 0.836), against 6.29 for the best heuristic -- a 53% reduction in error. The 80% interval covers 77.3% of outcomes.

**Headline results**

| Measure | Value |
|---|---|
| Mae minutes | 2.9284 |
| Rmse minutes | 4.6072 |
| R2 | 0.8356 |
| Fare mae usd | 1.737 |
| Interval coverage | 0.7727 |

**Baselines the model had to beat**

| Baseline | Score | What it does |
|---|---|---|
| Predict training mean | 8.0365 | Always answer 15.0 minutes. |
| Distance / median hourly speed | 7.7311 | The dispatcher's rule of thumb, calibrated per hour of day. |
| Borough-pair historical mean | 6.2909 | Lookup table of average duration for each origin/destination pair. |

## 7. Deployment

Served by FastAPI behind /api/projects/taxi/predict. The pipeline is pickled whole -- preprocessing travels with the model, so training and serving cannot drift apart.

Served at `POST /api/projects/taxi/predict`. The fitted pipeline is pickled whole, so preprocessing travels with the model and training and serving cannot drift apart.

## 8. Limitations

- Trained on a single month; no seasonal coverage.
- No live traffic, weather or incident feed -- the model learns the average January Tuesday, not today's Tuesday.
- Widest errors on airport trips; MAE there exceeds the aggregate by a visible margin (see the per-segment table).
- Predicted fare excludes tips, tolls and surcharges by construction.

**Out of scope**

- Green cabs, for-hire vehicles and app-dispatch services, which have different pricing rules and are not in the training data.
- Any month other than January: the model has never seen summer traffic, school holidays, or a different congestion-pricing regime.
- Individual driver or passenger assessment of any kind.

**Ethical considerations**

- Zone-level features could encode neighbourhood demographics. The model predicts travel time, not creditworthiness or risk, and is not used to accept or refuse service, which bounds the harm surface.
- Duration estimates that are systematically pessimistic for outer-borough trips could discourage drivers from accepting them; the per-borough bias table exists to make that measurable rather than invisible.

## 9. Audit

10 of 10 checks pass (grade A). *Scope:* Static review of the training path for target leakage, temporal leakage, preprocessing leakage, selection leakage and metric gaming.

| Check | Status | Evidence |
|---|---|---|
| Temporal ordering preserved | pass | assert_chronological() enforced at the split boundary 2024-01-26 01:45:58; no shuffle anywhere in the path. |
| Post-outcome columns excluded | pass | tip_amount, tolls_amount, total_amount, congestion_surcharge and dropoff_datetime are absent from ALL_FEATURES. |
| Preprocessing fitted per fold | pass | Imputers, one-hot vocabulary and the zone frequency table are ColumnTransformer steps inside the Pipeline that CV clones. |
| Row filters independent of the target distribution | pass | All 8 rules are physical bounds; none references a mean, standard deviation or quantile of the target. |
| Hold-out used once | pass | Hyperparameter search scored only cross-validated folds; the test half was predicted after the search closed. |
| Search overfitting | pass | No family's CV score exceeds its hold-out score by more than 0.5 min. |
| Metric appropriate to the task | pass | MAE reported as headline with RMSE, R^2, bias and p90 error alongside; no accuracy-style metric is claimed for a regression. |
| Baselines published | pass | Three heuristics reported; champion MAE 2.93 min beats the best of them. |
| Importance cross-validated across methods | pass | 14 features exceed twice their permutation noise; impurity importance shown alongside for contrast. |
| Determinism | pass | Every estimator and sampler seeded (seed=42); ingestion seeded separately at 20240115. |

**Phase gates:** 6 of 6 passed.

- PASS — 1. Business Understanding
- PASS — 2. Data Understanding: Overall data quality 100.0% (grade A).
- PASS — 3. Data Preparation
- PASS — 4. Modeling
- PASS — 5. Evaluation: MAE 2.93 min against a 5.00 min target.
- PASS — 6. Deployment

## 10. Where this project looped back

CRISP-DM is iterative. These are the points where a later phase sent the work back to an earlier one — normally the part deleted before publication.

- Phase 5 sent us back to phase 3: the first evaluation showed airport trips with double the error of city trips, which is why an explicit trip_type feature exists at all.
- The interval was added after phase 1's review concluded that a bare point estimate would overstate what the data supports.

## Reproduction

```bash
python scripts/fetch_data.py
python scripts/train_all.py --only taxi
```

Every estimator, split and sampler is seeded, so a rerun on the same data reproduces this leaderboard exactly.
