# NYC Taxi Fare & Duration

*Spatio-temporal gradient boosting on 150,000 real January-2024 yellow-cab trips.*

## The question

Can we tell a rider, at the moment they request a yellow cab, how long the trip will take and what it will cost -- accurately enough that they trust the number and plan around it?

## Why this metric and not accuracy

Riders experience absolute lateness, not squared error, and MAE is robust to the genuine 4-hour outlier trips the TLC publishes.

## What the data actually looked like

150,000 trips sampled from the 2.96M the TLC published for January 2024, joined to the official 265-zone dictionary. The data is real, and so are its defects: the quality scorecard grades it A at 100.0%.

**A visible minority of trips are physically impossible** — 6,312 rows (4.21%) violate at least one plausibility rule. Removed by stated rule, each exclusion counted and published.

**Average speed collapses during the weekday peak** — peak-hour mean duration cell: Wed 4:00. Motivated explicit rush-hour and cyclical-time features.

**Trip distance is strongly but non-linearly related to duration** — Spearman exceeds Pearson across the correlation matrix. Signals that tree ensembles should beat linear models.

## The modelling

79 hill-climbing trials across 6 model families, scored by expanding-window cross-validated MAE. 'XGBoost' won and was deployed.

> **Hill climbing rather than random or grid search.** Produces a readable improvement trajectory and reaches a good configuration in ~14 fits per family.
>
> *We rejected Exhaustive grid search:* Hundreds of fits for a fraction of a minute of MAE.

> **Tune on cross-validated MAE, never on the hold-out.** The hold-out has to stay unseen to mean anything.
>
> *We rejected Early stopping against the test set:* Standard practice in tutorials and a direct leak.

## Results

| Measure | Value |
|---|---|
| Mae minutes | 2.9284 |
| Rmse minutes | 4.6072 |
| R2 | 0.8356 |
| Fare mae usd | 1.737 |
| Interval coverage | 0.7727 |

## The finding worth reading twice

**Tip, tolls and total fare are recorded post-trip**

4 of 14 source columns. Excluded from the feature set; using them would leak the outcome into the prediction.

## What went wrong first

Every project here records where a later phase sent the work back to an earlier one. That is normally the part deleted before publication, and it is usually the most useful part.

- Phase 5 sent us back to phase 3: the first evaluation showed airport trips with double the error of city trips, which is why an explicit trip_type feature exists at all.
- The interval was added after phase 1's review concluded that a bare point estimate would overstate what the data supports.

## What it cannot do

- Green cabs, for-hire vehicles and app-dispatch services, which have different pricing rules and are not in the training data.
- Any month other than January: the model has never seen summer traffic, school holidays, or a different congestion-pricing regime.
- Individual driver or passenger assessment of any kind.

## Try it

The live inference playground for this project is on the **Live inference** tab of the console, or call it directly:

```bash
curl -X POST localhost:8000/api/projects/taxi/predict \
  -H 'Content-Type: application/json' -d '{}'
```
