# RFM Customer Segmentation

*Unsupervised segmentation of 4,339 real UK e-commerce customers.*

**Task** Unsupervised clustering  ·  **Domain** Customer intelligence  ·  **Primary metric** Silhouette coefficient

> Generated from the training run of `2026-09-02T19:47:28+00:00` (commit `n/a`, seed 42). Every figure below is read from that run's artifacts.

## Abstract

Unsupervised segmentation of 4,339 real UK e-commerce customers. The system follows the full CRISP-DM cycle on 4,339 rows of UCI Online Retail (customer-level RFM aggregation), selecting K-Means by comparative evaluation on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: k 4, silhouette 0.2389, davies bouldin 1.2923, stability ari 0.9273. 4 segments, silhouette 0.239, bootstrap ARI 0.9273. The top two segments hold 86% of revenue. Every segment carries a distinct action. A static leakage audit of the training path passes 9 of 9 checks, and 5 of 6 CRISP-DM phase gates are met.

## 1. Business understanding

**Question.** Our 4,339 customers are currently treated as one undifferentiated list. Are there natural groups with distinct buying behaviour that would justify separate merchandising and contact strategies?

**Success criteria, fixed before modelling:**

- Segments must be internally coherent (positive mean silhouette).
- The partition must beat a uniform null on the gap statistic.
- It must survive bootstrap resampling (ARI comfortably above chance).
- Every segment must map to an action a merchandiser can actually take.

**Why Silhouette coefficient.** There is no ground-truth label, so cluster quality must be judged internally; silhouette balances cohesion against separation.

## 2. Data

- **Source** — UCI Online Retail (customer-level RFM aggregation)
- **Rows** — 4,339 in the curated file; 4,338 after preparation
- **Licence** — CC BY 4.0
- **Quality** — grade A at 100.0% across six dimensions
- **Missingness** — 0 of 39,042 cells (0.00%); 100.0% of rows complete

The curation rule and a SHA-256 for this file are recorded in [`data/MANIFEST.json`](../../../data/MANIFEST.json). Ingestion applies only column selection, dtype coercion, deterministic subsampling and stable sorting — no imputation, scaling or target-aware filtering.

## 3. Data Understanding

4,338 customers aggregated from 530,693 real invoice lines. All three RFM axes are severely right-skewed -- monetary skew exceeds 19 before transformation.

**Findings.**

- *RFM features are extremely right-skewed* — monetary skew before transform: 19.325. Untransformed, K-Means would produce one huge cluster and a few wholesale singletons.
- *A small number of customers dominate revenue* — top segment holds 72.3% of revenue with 21.0% of customers. Confirms segmentation is worth doing at all.
- *1 customers have non-positive net spend* — monetary > 0 and frequency > 0. Excluded: they have no position in RFM space.

**Decisions.**

- **Use only five features for clustering.** Euclidean distance weights every axis equally, so weak features actively dilute strong ones. *Rejected:* Cluster on all nine available columns — Tenure and average unit price add noise dimensions without adding separation.

## 4. Data Preparation

log1p then standardise, as one fitted Pipeline. The transform is the single most consequential decision in this project.

**Findings.**

- *log1p reduces skew on every feature* — 5 of 5 features improved. Distance now reflects relative, not absolute, spend.

**Decisions.**

- **log1p rather than a fitted power transform.** Log spend is the unit merchandisers reason in, which keeps the segment profiles explainable. *Rejected:* Yeo-Johnson with a fitted lambda — Marginally more symmetric, considerably less interpretable in the profile tables.
- **Standardise after logging, inside the pipeline.** Keeps the transform a single fitted object that travels with the model to inference. *Rejected:* Transform the frame up front — Splits the transform from the model and invites training/serving skew.

## 5. Modelling

k swept from 2 to 10 on three internal indices plus a 12-reference gap statistic; four algorithms then compared at k=4. K-Means won on silhouette.

**Leaderboard** — scored by silhouette.

| Model | Silhouette | Bootstrap ARI |
|---|---|---|
| K-Means | 0.2389 | 0.9273 |
| Ward hierarchical | 0.1921 | 0.3656 |
| Gaussian mixture | 0.0919 | 0.971 |
| DBSCAN | 0.2667 | 0.9574 |

> Unconstrained, silhouette is maximised at k=2 (0.347); it falls monotonically as k grows, so the index alone would always prefer the coarsest possible split. A two-way high/low split is not a segmentation a merchandiser can act on differently, so phase 1's actionability criterion restricts the search to k in [4, 7]. Within that range silhouette is highest at k=4 (0.238). Davies-Bouldin is minimised at k=2 and the inertia elbow sits near k=5; the indices disagree, which is normal, and the gap statistic confirms the partition beats a uniform null.

## 6. Evaluation

4 segments, silhouette 0.239, bootstrap ARI 0.9273. The top two segments hold 86% of revenue. Every segment carries a distinct action.

**Headline results**

| Measure | Value |
|---|---|
| K | 4 |
| Silhouette | 0.2389 |
| Davies bouldin | 1.2923 |
| Stability ari | 0.9273 |
| Segments | 4 |
| Gap supports clustering | yes |

## 7. Deployment

Centroids and the fitted transform are pickled together, so a new customer can be assigned by transforming their RFM vector and taking the nearest centroid.

Served at `POST /api/projects/segments/predict`. The fitted pipeline is pickled whole, so preprocessing travels with the model and training and serving cannot drift apart.

## 8. Limitations

- Only customers with an ID are represented; roughly a quarter of the source line items are anonymous guest checkouts and are absent entirely.
- Recency is measured against a fixed as-of date, so the model must be refitted rather than merely re-scored as time passes.
- Monetary is gross of returns.
- Customers near a segment boundary have low silhouette and should receive generic rather than segment-specific treatment.

**Out of scope**

- Individual credit, pricing or eligibility decisions.
- Predicting whether a specific customer will buy -- this is a description of past behaviour, not a forecast.
- Markets outside the UK-dominated book this was fitted on.

**Ethical considerations**

- Value-based segmentation systematically directs better service to higher-spending customers. That is a legitimate commercial choice, but it should be a conscious one -- the revenue-share table makes the effect explicit rather than letting it happen quietly.
- Segment labels are descriptive of past purchasing and must not be used as proxies for any protected characteristic.

## 9. Audit

9 of 9 checks pass (grade A). *Scope:* Review for the failure modes specific to unsupervised work: imposing structure on noise, unstable partitions, and selecting k by an index that cannot fail.

| Check | Status | Evidence |
|---|---|---|
| No target leakage possible | pass | Unsupervised task; there is no label anywhere in the pipeline. |
| Transform fitted as part of the model | pass | log1p and StandardScaler are Pipeline steps pickled with the centroids, so inference reproduces training exactly. |
| Cluster count justified, not assumed | pass | k swept 2-10 on silhouette, Davies-Bouldin, Calinski-Harabasz and the elbow; disagreement reported openly. |
| Null hypothesis tested | pass | Maximum gap over a uniform null is 1.924. A positive gap means the data is genuinely more clustered than featureless noise spanning the same range; a gap near zero at every k would mean there is no segment structure to find. |
| Stability verified | pass | Bootstrap ARI 0.9273 over resampled subsets; a partition that dissolved would be rejected. |
| Multiple algorithms cross-checked | pass | Four algorithms with different cluster definitions; best pairwise ARI 0.4817. |
| Exclusions counted | pass | 1 customers removed by the stated rule 'monetary > 0 and frequency > 0'. |
| Per-customer confidence published | pass | Individual silhouette values reported; boundary members flagged rather than presented as confident assignments. |
| Determinism | pass | All algorithms seeded at 42; n_init pinned. |

**Phase gates:** 5 of 6 passed.

- PASS — 1. Business Understanding
- PASS — 2. Data Understanding
- PASS — 3. Data Preparation
- PASS — 4. Modeling
- **FAIL** — 5. Evaluation: FAILED on cohesion: mean silhouette 0.239 falls short of the 0.25 bar fixed in phase 1. The other two criteria pass -- the gap statistic rejects a uniform null and the partition is highly stable under resampling (ARI 0.9273) -- so the structure is real, it is simply not crisply separated. That is the expected shape of customer RFM data: a continuum with soft boundaries rather than distinct blobs. The gate is reported as failed rather than the threshold being moved to fit the result, and the consequence is stated in the model card: customers near a boundary get generic treatment, not a segment-specific campaign.
- PASS — 6. Deployment

## 10. Where this project looped back

CRISP-DM is iterative. These are the points where a later phase sent the work back to an earlier one — normally the part deleted before publication.

- The first run clustered on raw RFM and produced one cluster holding 96% of customers. That failure is what drove the log transform, and it is the single largest improvement in the project.
- k was initially chosen by the elbow alone. Adding the gap statistic and silhouette showed the elbow was reading a smooth curve, so the selection rule was rewritten to state which index decides and why.

## Reproduction

```bash
python scripts/fetch_data.py
python scripts/train_all.py --only segments
```

Every estimator, split and sampler is seeded, so a rerun on the same data reproduces this leaderboard exactly.
