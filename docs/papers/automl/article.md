# AutoML Stacking Tournament

*Hill-climbing search and Caruana ensembling over 48,842 census records.*

## The question

Given a fixed tabular dataset, how much does automated model search actually buy over a sensible default -- and how much of the apparent gain is the search fitting its own validation folds?

## Why this metric and not accuracy

Model selection needs a ranking metric stable across folds; the CV-to-hold-out gap is reported alongside to expose search overfitting.

## What the data actually looked like

48,842 census records, 23.9% above the $50k threshold, 6,465 missing cells across three columns.

**fnlwgt is a survey design artefact** — Census sampling weight -- a survey-design artefact describing how many people the row represents, not a proper. Dropped: it describes the sampling frame, not the person, and trees find spurious structure in it.

**education and education_num are the same variable** — one-to-one text/ordinal correspondence. Kept the ordinal; keeping both would split one signal across two columns.

**Base rates differ sharply across demographic groups** — see the rate-by-group table. Guarantees that demographic parity and calibration cannot both hold -- the trade-off is structural, not a modelling failure.

## The modelling

101 hill-climbing trials over 6 families, then out-of-fold stacking and Caruana greedy selection. Caruana greedy ensemble was deployed.

> **Train the meta-learner strictly on out-of-fold predictions.** The meta-learner is trained exclusively on out-of-fold predictions: every base-model probability it sees was produced by a model that did not train on
>
> *We rejected Fit base models on all training data, then predict that same data for the meta-learner:* The meta-learner then sees the labels through base-model memorisation; the gain is fictional.

> **Report greedy selection and logistic stacking together.** Greedy cannot use negative weights; logistic can, but can overfit the OOF matrix. Neither dominates.
>
> *We rejected Report only the better one:* Selecting the winner post hoc on the hold-out is itself a form of hold-out contamination.

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

## The finding worth reading twice

**A leaderboard without a hold-out gap is not evidence**

CV scores rise with every trial by construction. Both numbers are reported for every family.

## What went wrong first

Every project here records where a later phase sent the work back to an earlier one. That is normally the part deleted before publication, and it is usually the most useful part.

- The first run kept fnlwgt and scored noticeably higher. The audit flagged it as a sampling artefact, and removing it cost real AUC -- which is the point: the lost performance was never available in deployment.
- Stacking initially showed a large gain because the meta-learner was trained on in-fold base predictions. Rebuilding it on out-of-fold predictions shrank the gain to near zero, which is the honest number.

## What it cannot do

- Any real decision about a real person -- lending, hiring, housing, insurance or eligibility of any kind.
- Present-day income estimation. The data is a 1994 US census extract; the labour market, the $50k threshold and the demographics have all moved.
- Transfer to any population outside the 1994 US Current Population Survey.

## Try it

The live inference playground for this project is on the **Live inference** tab of the console, or call it directly:

```bash
curl -X POST localhost:8000/api/projects/automl/predict \
  -H 'Content-Type: application/json' -d '{}'
```
