# Bike-Share Demand Forecasting — Abstract

**Task** Time series forecasting  ·  **Data** UCI Bike Sharing (hourly, 2011-2012)  ·  **Metric** MASE

Multi-horizon hourly forecasting over 17,379 real Capital Bikeshare observations. The system follows the full CRISP-DM cycle on 17,379 rows of UCI Bike Sharing (hourly, 2011-2012), selecting Gradient boosting on lags by comparative evaluation on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: mase 0.3347, mae 28.7969, rmse 45.6546, r2 0.9571. Hold-out MAE 28.8 rides against 64.217 for seasonal naive -- a skill score of 55.2%. MASE is 0.335, but note the naive method itself scores 0.7463 on this hold-out, so the head-to-head skill score is the meaningful comparison. The 80% band covers 79.2% of hours. A static leakage audit of the training path passes 13 of 13 checks, and 6 of 6 CRISP-DM phase gates are met.

## Results

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

## What this model must not be used for

- Long-horizon planning beyond about a week, where the model has only calendar information and degrades to climatology.
- Other cities: demand rhythm is a function of local geography, transit and climate that this model has no access to.
- Individual station-level forecasting -- this is a system-wide total.

## Known limitations

- Two years of history only; no multi-year trend can be learned.
- Weather features are observations, not forecasts. In deployment they would be replaced by predicted weather, adding error this evaluation does not capture.
- Requires 168 hours of contiguous history; gaps degrade it to calendar-only accuracy.
- System-wide totals only -- no spatial resolution.

Full write-up: [`paper.md`](./paper.md).
