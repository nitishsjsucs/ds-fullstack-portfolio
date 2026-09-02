# Telco Churn & Retention Economics

*Cost-sensitive classification with calibration and a fairness audit on 7,043 subscribers.*

## The question

Which subscribers are about to leave, and for which of them is a retention offer actually worth making?

## Why this metric and not accuracy

Churn prevalence is 26.5%. Accuracy rewards predicting 'stays' for everyone (73.5%); PR-AUC only rewards ranking real churners highly.

## What the data actually looked like

7,043 subscribers, 21 fields, 11 missing cells. Quality grade A. Churn is heavily concentrated in month-to-month contracts and short tenure.

**All blank TotalCharges belong to tenure-0 customers** — 11 of 11 blanks. Structurally missing, not randomly missing; left as NaN for pipeline imputation.

**Contract type is the single strongest segment signal** — month-to-month churn far exceeds two-year churn. Confirms the analyst heuristic is a fair baseline.

**Churn risk decays sharply with tenure** — see the retention curve. Motivated tenure bucketing and the charges-per-tenure ratio feature.

## The modelling

95 hill-climbing trials over 6 families on cross-validated average precision, then Caruana greedy ensembling over out-of-fold predictions. 'XGBoost' was deployed.

> **Deploy isotonic calibration.** Chosen on ECE because the probability is multiplied by a dollar amount downstream.
>
> *We rejected Ship the raw model score:* Ranks well but is not a probability, so the expected-value calculation would be wrong.

> **Deploy at threshold 0.27, not 0.50.** Minimises expected cost under the stated economics.
>
> *We rejected Default 0.50:* Implicitly assumes symmetric costs and would leave most recoverable churners uncontacted.

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

## The finding worth reading twice

**customerID is unique for every row**

7,043 distinct values in 7,043 rows. Excluded: a tree given it memorises the training set and generalises to nothing.

## What went wrong first

Every project here records where a later phase sent the work back to an earlier one. That is normally the part deleted before publication, and it is usually the most useful part.

- The first pass used a 0.5 threshold and reported 80% accuracy. The evaluation review rejected it: at that cut the model contacted too few churners to fund the campaign. Introducing the cost matrix sent us back to phase 1 to write the economics down properly.
- Calibration was added after noticing that ranking was strong while predicted probabilities systematically understated observed churn.

## What it cannot do

- Pricing decisions for individual customers.
- Any use where the prediction affects service eligibility.
- Markets other than the one this snapshot describes -- contract mix and payment norms differ enough to invalidate the learned relationships.

## Try it

The live inference playground for this project is on the **Live inference** tab of the console, or call it directly:

```bash
curl -X POST localhost:8000/api/projects/churn/predict \
  -H 'Content-Type: application/json' -d '{}'
```
