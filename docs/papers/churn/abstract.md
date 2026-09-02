# Telco Churn & Retention Economics — Abstract

**Task** Supervised binary classification (imbalanced)  ·  **Data** IBM Telco Customer Churn  ·  **Metric** PR-AUC (average precision)

Cost-sensitive classification with calibration and a fairness audit on 7,043 subscribers. The system follows the full CRISP-DM cycle on 7,043 rows of IBM Telco Customer Churn, selecting XGBoost by hill-climbing search on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: pr auc 0.6604, roc auc 0.848, recall at deployed 0.8209, precision at deployed 0.5142. Hold-out PR-AUC 0.660 against a 0.539 best baseline and a 0.265 prevalence floor. At the deployed threshold recall is 82.1% at 51.4% precision. The simulated campaign nets $11,597. A static leakage audit of the training path passes 11 of 11 checks, and 6 of 6 CRISP-DM phase gates are met.

## Results

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

## What this model must not be used for

- Pricing decisions for individual customers.
- Any use where the prediction affects service eligibility.
- Markets other than the one this snapshot describes -- contract mix and payment norms differ enough to invalidate the learned relationships.

## Known limitations

- A single point-in-time snapshot: no seasonality, no campaign history, no way to learn what past interventions already changed.
- 7,043 rows is small; confidence intervals on segment metrics are wide.
- Economics are stated assumptions, not measured values; the conclusions move with them, which is why the UI exposes them as sliders.
- Trained on one operator's book; transfer to another market is unproven.

Full write-up: [`paper.md`](./paper.md).
