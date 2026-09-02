# AutoML Stacking Tournament — Abstract

**Task** AutoML / model selection  ·  **Data** UCI Adult / Census Income (1994 CPS)  ·  **Metric** ROC-AUC (CV) with hold-out gap

Hill-climbing search and Caruana ensembling over 48,842 census records. The system follows the full CRISP-DM cycle on 48,842 rows of UCI Adult / Census Income (1994 CPS), selecting Caruana greedy ensemble by hill-climbing search on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: roc auc 0.9314, pr auc 0.8363, trials 101, families 6. Hold-out ROC-AUC 0.9314 (PR-AUC 0.8363) against 0.7543 for an analyst heuristic. A depth-4 tree reproduces 92.5% of the ensemble's decisions and 93.1% of its AUC. The gap is large enough to justify the A static leakage audit of the training path passes 11 of 11 checks, and 6 of 6 CRISP-DM phase gates are met.

## Results

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

## What this model must not be used for

- Any real decision about a real person -- lending, hiring, housing, insurance or eligibility of any kind.
- Present-day income estimation. The data is a 1994 US census extract; the labour market, the $50k threshold and the demographics have all moved.
- Transfer to any population outside the 1994 US Current Population Survey.

## Known limitations

- 1994 data with a fixed nominal $50k threshold; not inflation-adjusted and not comparable to any present-day income band.
- Self-reported survey responses with known non-response bias.
- The ensemble's gain over a single model is within noise on this dataset.
- Distillation fidelity is measured against the ensemble's decisions, not against ground truth.

Full write-up: [`paper.md`](./paper.md).
