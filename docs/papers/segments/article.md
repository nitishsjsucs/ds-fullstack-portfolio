# RFM Customer Segmentation

*Unsupervised segmentation of 4,339 real UK e-commerce customers.*

## The question

Our 4,339 customers are currently treated as one undifferentiated list. Are there natural groups with distinct buying behaviour that would justify separate merchandising and contact strategies?

## Why this metric and not accuracy

There is no ground-truth label, so cluster quality must be judged internally; silhouette balances cohesion against separation.

## What the data actually looked like

4,338 customers aggregated from 530,693 real invoice lines. All three RFM axes are severely right-skewed -- monetary skew exceeds 19 before transformation.

**RFM features are extremely right-skewed** — monetary skew before transform: 19.325. Untransformed, K-Means would produce one huge cluster and a few wholesale singletons.

**A small number of customers dominate revenue** — top segment holds 72.3% of revenue with 21.0% of customers. Confirms segmentation is worth doing at all.

**1 customers have non-positive net spend** — monetary > 0 and frequency > 0. Excluded: they have no position in RFM space.

## The modelling

k swept from 2 to 10 on three internal indices plus a 12-reference gap statistic; four algorithms then compared at k=4. K-Means won on silhouette.

> **Deploy K-Means at k=4.** Highest silhouette with strong bootstrap stability.
>
> *We rejected DBSCAN:* Labels a large share of customers as noise, which leaves them unassignable to any campaign.

> **Report the gap statistic rather than only the elbow.** The elbow always exists, even in pure noise; the gap statistic is the only test that can say 'no clusters'.
>
> *We rejected Elbow plot alone:* Unfalsifiable -- it cannot fail.

## Results

| Measure | Value |
|---|---|
| K | 4 |
| Silhouette | 0.2389 |
| Davies bouldin | 1.2923 |
| Stability ari | 0.9273 |
| Segments | 4 |
| Gap supports clustering | yes |

## The finding worth reading twice

**RFM features are extremely right-skewed**

monetary skew before transform: 19.325. Untransformed, K-Means would produce one huge cluster and a few wholesale singletons.

## What went wrong first

Every project here records where a later phase sent the work back to an earlier one. That is normally the part deleted before publication, and it is usually the most useful part.

- The first run clustered on raw RFM and produced one cluster holding 96% of customers. That failure is what drove the log transform, and it is the single largest improvement in the project.
- k was initially chosen by the elbow alone. Adding the gap statistic and silhouette showed the elbow was reading a smooth curve, so the selection rule was rewritten to state which index decides and why.

## What it cannot do

- Individual credit, pricing or eligibility decisions.
- Predicting whether a specific customer will buy -- this is a description of past behaviour, not a forecast.
- Markets outside the UK-dominated book this was fitted on.

## Try it

The live inference playground for this project is on the **Live inference** tab of the console, or call it directly:

```bash
curl -X POST localhost:8000/api/projects/segments/predict \
  -H 'Content-Type: application/json' -d '{}'
```
