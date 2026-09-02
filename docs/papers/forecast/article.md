# Bike-Share Demand Forecasting

*Multi-horizon hourly forecasting over 17,379 real Capital Bikeshare observations.*

## The question

How many bikes will be hired in the next hour, and how confident can an operations team be in that number when deciding where to rebalance stock?

## Why this metric and not accuracy

MASE is scale-free and divides by the seasonal-naive error, so a value below 1 proves the model beats 'same hour last week' -- the only baseline that matters.

## What the data actually looked like

17,379 observed hours over two years with 165 gaps. STL attributes 70% of variance to the daily cycle and 28% to residual noise.

**165 hours are missing from the archive** — 0.94% of the expected grid. Reindexed as explicit NaNs so lag distances stay true; never imputed.

**ACF shows dominant peaks at lags 24 and 168** — both far outside the significance band. Directly determined the lag feature set.

**Demand is bimodal on working days** — morning and evening commuting peaks; a single midday hump at weekends. Motivated the hour x working-day interaction that tree models capture and the linear model cannot.

## The modelling

Three models on 5 purged rolling-origin folds. Gradient boosting on lags wins with CV MASE 0.4443.

> **Purge by the maximum look-back and embargo a further day.** Feature windows must not reach across the fold boundary.
>
> *We rejected Plain TimeSeriesSplit:* Leaks whenever any feature uses a trailing window, which every useful one here does.

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

## The finding worth reading twice

**casual and registered sum exactly to the target**

casual + registered == cnt for every row. Both dropped: either alone gives a perfect score and neither is known before the hour occurs.

## What went wrong first

Every project here records where a later phase sent the work back to an earlier one. That is normally the part deleted before publication, and it is usually the most useful part.

- The first version computed rolling means without shifting and reported a MASE around 0.3. That number was impossible, which is what prompted the audit that found the leak. The shift is now the first thing the preparation module does, and the leakage test suite asserts it.
- Plain TimeSeriesSplit was replaced by the purged variant after noticing that CV scores were consistently better than hold-out scores -- the signature of trailing windows crossing the fold boundary.

## What it cannot do

- Long-horizon planning beyond about a week, where the model has only calendar information and degrades to climatology.
- Other cities: demand rhythm is a function of local geography, transit and climate that this model has no access to.
- Individual station-level forecasting -- this is a system-wide total.

## Try it

The live inference playground for this project is on the **Live inference** tab of the console, or call it directly:

```bash
curl -X POST localhost:8000/api/projects/forecast/predict \
  -H 'Content-Type: application/json' -d '{}'
```
