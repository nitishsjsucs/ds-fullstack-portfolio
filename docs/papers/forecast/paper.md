# Bike-Share Demand Forecasting

*Multi-horizon hourly forecasting over 17,379 real Capital Bikeshare observations.*

**Task** Time series forecasting  ·  **Domain** Demand planning  ·  **Primary metric** MASE

> Generated from the training run of `2026-09-02T18:16:32+00:00` (commit `n/a`, seed 42). Every figure below is read from that run's artifacts.

## Abstract

Multi-horizon hourly forecasting over 17,379 real Capital Bikeshare observations. The system follows the full CRISP-DM cycle on 17,379 rows of UCI Bike Sharing (hourly, 2011-2012), selecting Gradient boosting on lags by comparative evaluation on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: mase 0.3347, mae 28.7969, rmse 45.6546, r2 0.9571. Hold-out MAE 28.8 rides against 64.217 for seasonal naive -- a skill score of 55.2%. MASE is 0.335, but note the naive method itself scores 0.7463 on this hold-out, so the head-to-head skill score is the meaningful comparison. The 80% band covers 79.2% of hours. A static leakage audit of the training path passes 13 of 13 checks, and 6 of 6 CRISP-DM phase gates are met.

## 1. Business understanding

**Question.** How many bikes will be hired in the next hour, and how confident can an operations team be in that number when deciding where to rebalance stock?

**Success criteria, fixed before modelling:**

- Positive skill against seasonal naive on the same hold-out -- the model must genuinely beat 'same hour last week', not merely score MASE < 1.
- Accuracy verified across five rolling-origin folds, not one lucky split.
- An 80% interval whose empirical coverage is close to 80%.
- No feature may use information unavailable at forecast time.

**Why MASE.** MASE is scale-free and divides by the seasonal-naive error, so a value below 1 proves the model beats 'same hour last week' -- the only baseline that matters.

## 2. Data

- **Source** — UCI Bike Sharing (hourly, 2011-2012)
- **Rows** — 17,379 in the curated file; 16,631 after preparation
- **Licence** — CC BY 4.0
- **Quality** — grade A at 100.0% across six dimensions
- **Missingness** — 0 of 166,310 cells (0.00%); 100.0% of rows complete

The curation rule and a SHA-256 for this file are recorded in [`data/MANIFEST.json`](../../../data/MANIFEST.json). Ingestion applies only column selection, dtype coercion, deterministic subsampling and stable sorting — no imputation, scaling or target-aware filtering.

## 3. Data Understanding

17,379 observed hours over two years with 165 gaps. STL attributes 70% of variance to the daily cycle and 28% to residual noise.

**Findings.**

- *165 hours are missing from the archive* — 0.94% of the expected grid. Reindexed as explicit NaNs so lag distances stay true; never imputed.
- *ACF shows dominant peaks at lags 24 and 168* — both far outside the significance band. Directly determined the lag feature set.
- *Demand is bimodal on working days* — morning and evening commuting peaks; a single midday hump at weekends. Motivated the hour x working-day interaction that tree models capture and the linear model cannot.

**Decisions.**

- **Reindex gaps rather than dropping the rows.** Lags must be measured in hours, not in rows. *Rejected:* Treat consecutive rows as consecutive hours — Turns a 24-lag into 'whatever happened 24 rows ago', silently corrupting every lag feature.

## 4. Data Preparation

33 features: calendar, cyclical encodings, seven lags and rolling statistics -- all shifted so they look strictly backwards. 748 warm-up rows dropped where the 168-hour look-back is undefined.

**Findings.**

- *Rolling features are shifted before aggregation* — past = target.shift(1) precedes every .rolling() call. Without it each rolling window contains the value it is used to predict -- a leak that typically halves the reported error.
- *Warm-up rows cannot be scored fairly* — 748 rows dropped. The longest lag is undefined there; keeping them would mean imputing the predictor.

**Decisions.**

- **Shift before rolling, everywhere.** A window that includes the current hour is a leak. *Rejected:* rolling(24).mean() on the raw target — The single most common time-series bug; produces excellent scores that do not survive deployment.

## 5. Modelling

Three models on 5 purged rolling-origin folds. Gradient boosting on lags wins with CV MASE 0.4443.

**Leaderboard** — scored by MASE (lower is better).

| Model | CV MASE | Hold-out MASE | Gap |
|---|---|---|---|
| Gradient boosting on lags | 0.4443 | 0.3347 | -0.1096 |
| Ridge on lags + calendar | 0.5402 | 0.5169 | -0.0233 |
| Seasonal naive (t-168) | 0.7244 | 0.7463 | 0.0219 |

## 6. Evaluation

Hold-out MAE 28.8 rides against 64.217 for seasonal naive -- a skill score of 55.2%. MASE is 0.335, but note the naive method itself scores 0.7463 on this hold-out, so the head-to-head skill score is the meaningful comparison. The 80% band covers 79.2% of hours.

**Headline results**

| Measure | Value |
|---|---|
| Mase | 0.3347 |
| Mae | 28.7969 |
| Rmse | 45.6546 |
| R2 | 0.9571 |
| Skill vs seasonal naive | 0.5516 |
| Beats seasonal naive | yes |
| Interval coverage | 0.792 |
| Folds | 5 |

**Baselines the model had to beat**

| Baseline | Score | What it does |
|---|---|---|
| Overall mean | 174.853 | Always predict 182 rides. |
| Persistence (t-1) | 85.319 | Predict this hour equals last hour. |
| Seasonal naive (t-168) | 64.217 | Same hour last week -- the MASE denominator. |
| Hour-of-day climatology | 113.689 | Historical average for this hour of day. |

## 7. Deployment

Served at /api/projects/forecast/predict with a point forecast and an 80% band. The model needs the last 168 hours of history as input, which the endpoint documents explicitly.

Served at `POST /api/projects/forecast/predict`. The fitted pipeline is pickled whole, so preprocessing travels with the model and training and serving cannot drift apart.

## 8. Limitations

- Two years of history only; no multi-year trend can be learned.
- Weather features are observations, not forecasts. In deployment they would be replaced by predicted weather, adding error this evaluation does not capture.
- Requires 168 hours of contiguous history; gaps degrade it to calendar-only accuracy.
- System-wide totals only -- no spatial resolution.

**Out of scope**

- Long-horizon planning beyond about a week, where the model has only calendar information and degrades to climatology.
- Other cities: demand rhythm is a function of local geography, transit and climate that this model has no access to.
- Individual station-level forecasting -- this is a system-wide total.

**Ethical considerations**

- Demand forecasts drive where bikes are physically placed. Systematically under-forecasting low-income or peripheral areas would entrench unequal service; a station-level deployment would need a per-area equity audit that a system-wide total cannot provide.

## 9. Audit

13 of 13 checks pass (grade A). *Scope:* Review for the failure modes specific to time series: lookahead in rolling windows, unshuffled-but-unpurged CV, uneven spacing treated as even, and one-step accuracy quoted for multi-step use.

| Check | Status | Evidence |
|---|---|---|
| No shuffling anywhere | pass | temporal_split asserts monotonic timestamps; every CV fold is an expanding window in time order. |
| Rolling features shifted before aggregation | pass | engineer() computes past = target.shift(1) and derives every rolling statistic from it, so no window contains its own target. |
| Target-summing columns excluded | pass | casual + registered == cnt exactly. Either would give a perfect score and neither is known before the hour happens. |
| CV purged by the maximum look-back | pass | PurgedTimeSeriesSplit(window=168, embargo=24) -- training rows whose feature window reaches into validation are dropped. |
| Gaps made explicit, not silently skipped | pass | The archive omits 165 hours. They are reindexed in as explicit NaNs rather than left as absent rows, so a 24-lag is genuinely 24 hours back. Rows whose target is missing are exclud |
| Missing targets never imputed | pass | 165 hours with no observed demand are excluded from both training and scoring. |
| Baseline scored under the identical protocol | pass | Seasonal naive is an estimator inside the same backtest loop, not a separately-computed number. |
| Skill measured head-to-head, not inferred from MASE | pass | MASE 0.3346532242437272 is reported, but the deciding comparison is the skill score 0.5516 = 1 - MAE_model/MAE_naive, both on the same hold-out. MASE's denominator is in-sample error, so MASE < 1 alone would not establish skill. |
| Multi-horizon evaluated honestly | pass | At horizon h, every lag shorter than h is removed, because it would not have been observed yet. This is the difference between a real multi-step forecast and a  |
| Intervals conformally calibrated and coverage-tested | pass | Quantile models fitted on the first 85% of training and calibrated on the most recent 15% they never saw. Coverage went 67.1% -> 79.2% against 80% nominal. |
| Conformal calibration set disjoint from quantile fitting | pass | The calibration block is the last 15% of the training span and is excluded from the quantile models' fit, so conformity scores are genuinely out-of-sample. |
| Search overfitting | pass | No model's hold-out MASE exceeds its CV MASE by more than 0.15. |
| Determinism | pass | All estimators seeded at 42; folds are deterministic by position. |

**Phase gates:** 6 of 6 passed.

- PASS — 1. Business Understanding
- PASS — 2. Data Understanding
- PASS — 3. Data Preparation
- PASS — 4. Modeling
- PASS — 5. Evaluation: Skill 55.2%; model MAE 28.8 vs naive 64.217; best baseline MAE 64.217.
- PASS — 6. Deployment

## 10. Where this project looped back

CRISP-DM is iterative. These are the points where a later phase sent the work back to an earlier one — normally the part deleted before publication.

- The first version computed rolling means without shifting and reported a MASE around 0.3. That number was impossible, which is what prompted the audit that found the leak. The shift is now the first thing the preparation module does, and the leakage test suite asserts it.
- Plain TimeSeriesSplit was replaced by the purged variant after noticing that CV scores were consistently better than hold-out scores -- the signature of trailing windows crossing the fold boundary.

## Reproduction

```bash
python scripts/fetch_data.py
python scripts/train_all.py --only forecast
```

Every estimator, split and sampler is seeded, so a rerun on the same data reproduces this leaderboard exactly.
