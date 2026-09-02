# Network Intrusion Anomaly Detection

*Unsupervised outlier detection on 80,000 KDD-99 connections with a 3.4% attack base rate.*

## The question

Without any labelled attack data, can we rank network connections so that the intrusions rise to the top of an analyst's alert queue?

## Why this metric and not accuracy

At 3.4% prevalence a detector that flags nothing is 96.6% accurate. Analysts triage a fixed queue, so precision@k is the operational metric.

## What the data actually looked like

80,000 connections, 39 features, 3.37% attacks spanning 10 named families.

**Byte counts span nine orders of magnitude** — src_bytes ranges from 0 to hundreds of megabytes. Log transforms and a robust scaler are required before any distance-based method.

**Attack families are extremely unbalanced among themselves** — 10 families, most with a handful of instances. Aggregate recall would hide entire rare classes; per-family recall is reported instead.

## The modelling

Five detectors with genuinely different definitions of 'anomalous', all fitted on identical features. Best single: Isolation Forest at PR-AUC 0.5056. Rank fusion did not beat it.

> **Include Elliptic Envelope despite expecting it to lose.** A deliberately mis-specified Gaussian reference shows how much the distributional assumption costs.
>
> *We rejected Only run methods likely to win:* Removes the comparison that makes the others interpretable.

> **Fuse by rank, not by raw score.** The five score scales are incomparable.
>
> *We rejected Average raw scores:* Whichever detector has the largest numeric range would silently dominate.

## Results

| Measure | Value |
|---|---|
| Pr auc | 0.4739 |
| Roc auc | 0.9659 |
| Attack rate | 0.0337 |
| Lift over random | 14.1 |
| Precision at 100 | 0.56 |
| Detectors compared | 4 |

## The finding worth reading twice

**Accuracy is actively misleading here**

flagging nothing scores 96.6%. PR-AUC and precision@k are the reported metrics.

## What went wrong first

Every project here records where a later phase sent the work back to an earlier one. That is normally the part deleted before publication, and it is usually the most useful part.

- The first run passed the observed attack rate as contamination and produced flattering numbers. The audit caught it as leakage through a hyperparameter, which is why contamination is now a stated 2% assumption.
- Per-family recall was added after aggregate PR-AUC looked strong; it revealed that whole attack classes sit inside the normal density and cannot be found by any outlier method.

## What it cannot do

- Automated blocking. These are unsupervised anomaly scores with a substantial false-positive rate; they prioritise human review, they do not authorise action.
- Modern network traffic. KDD-99 is a 1999 simulated environment with protocols and attack patterns that no longer reflect reality.
- Any claim that a flagged connection is an attack -- it is merely unusual.

## Try it

The live inference playground for this project is on the **Live inference** tab of the console, or call it directly:

```bash
curl -X POST localhost:8000/api/projects/anomaly/predict \
  -H 'Content-Type: application/json' -d '{}'
```
