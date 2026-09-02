# NYC Taxi Fare & Duration — Abstract

**Task** Supervised regression (dual target)  ·  **Data** NYC TLC Yellow Taxi Trip Records, January 2024  ·  **Metric** MAE (minutes)

Spatio-temporal gradient boosting on 150,000 real January-2024 yellow-cab trips. The system follows the full CRISP-DM cycle on 150,000 rows of NYC TLC Yellow Taxi Trip Records, January 2024, selecting XGBoost by hill-climbing search on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: mae minutes 2.9284, rmse minutes 4.6072, r2 0.8356, fare mae usd 1.737. On the untouched final week the champion achieves 2.93 minutes MAE (R^2 0.836), against 6.29 for the best heuristic -- a 53% reduction in error. The 80% interval covers 77.3% of outcomes. A static leakage audit of the training path passes 10 of 10 checks, and 6 of 6 CRISP-DM phase gates are met.

## Results

| Measure | Value |
|---|---|
| Mae minutes | 2.9284 |
| Rmse minutes | 4.6072 |
| R2 | 0.8356 |
| Fare mae usd | 1.737 |
| Interval coverage | 0.7727 |

## What this model must not be used for

- Green cabs, for-hire vehicles and app-dispatch services, which have different pricing rules and are not in the training data.
- Any month other than January: the model has never seen summer traffic, school holidays, or a different congestion-pricing regime.
- Individual driver or passenger assessment of any kind.

## Known limitations

- Trained on a single month; no seasonal coverage.
- No live traffic, weather or incident feed -- the model learns the average January Tuesday, not today's Tuesday.
- Widest errors on airport trips; MAE there exceeds the aggregate by a visible margin (see the per-segment table).
- Predicted fare excludes tips, tolls and surcharges by construction.

Full write-up: [`paper.md`](./paper.md).
