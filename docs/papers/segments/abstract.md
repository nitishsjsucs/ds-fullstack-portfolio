# RFM Customer Segmentation — Abstract

**Task** Unsupervised clustering  ·  **Data** UCI Online Retail (customer-level RFM aggregation)  ·  **Metric** Silhouette coefficient

Unsupervised segmentation of 4,339 real UK e-commerce customers. The system follows the full CRISP-DM cycle on 4,339 rows of UCI Online Retail (customer-level RFM aggregation), selecting K-Means by comparative evaluation on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: k 4, silhouette 0.2389, davies bouldin 1.2923, stability ari 0.9273. 4 segments, silhouette 0.239, bootstrap ARI 0.9273. The top two segments hold 86% of revenue. Every segment carries a distinct action. A static leakage audit of the training path passes 9 of 9 checks, and 5 of 6 CRISP-DM phase gates are met.

## Results

| Measure | Value |
|---|---|
| K | 4 |
| Silhouette | 0.2389 |
| Davies bouldin | 1.2923 |
| Stability ari | 0.9273 |
| Segments | 4 |
| Gap supports clustering | yes |

## What this model must not be used for

- Individual credit, pricing or eligibility decisions.
- Predicting whether a specific customer will buy -- this is a description of past behaviour, not a forecast.
- Markets outside the UK-dominated book this was fitted on.

## Known limitations

- Only customers with an ID are represented; roughly a quarter of the source line items are anonymous guest checkouts and are absent entirely.
- Recency is measured against a fixed as-of date, so the model must be refitted rather than merely re-scored as time passes.
- Monetary is gross of returns.
- Customers near a segment boundary have low silhouette and should receive generic rather than segment-specific treatment.

Full write-up: [`paper.md`](./paper.md).
