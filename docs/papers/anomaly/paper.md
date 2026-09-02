# Network Intrusion Anomaly Detection

*Unsupervised outlier detection on 80,000 KDD-99 connections with a 3.4% attack base rate.*

**Task** Anomaly / outlier detection  ·  **Domain** Security telemetry  ·  **Primary metric** PR-AUC + precision@k

> Generated from the training run of `2026-09-02T21:29:36+00:00` (commit `n/a`, seed 42). Every figure below is read from that run's artifacts.

## Abstract

Unsupervised outlier detection on 80,000 KDD-99 connections with a 3.4% attack base rate. The system follows the full CRISP-DM cycle on 80,000 rows of KDD Cup 1999 network intrusion (SA subset), selecting Isolation Forest by comparative evaluation on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: pr auc 0.4739, roc auc 0.9659, attack rate 0.0337, lift over random 14.1. Isolation Forest achieves PR-AUC 0.474 against a 0.034 base rate -- 14x random. In a 100-alert queue, 56% of alerts are real intrusions. A static leakage audit of the training path passes 10 of 10 checks, and 6 of 6 CRISP-DM phase gates are met.

## 1. Business understanding

**Question.** Without any labelled attack data, can we rank network connections so that the intrusions rise to the top of an analyst's alert queue?

**Success criteria, fixed before modelling:**

- PR-AUC many times the 3.4% attack base rate.
- Useful precision within a realistic alert budget, not just in aggregate.
- No detector may see a label during fitting.
- Report which attack families remain invisible rather than only the average.

**Why PR-AUC + precision@k.** At 3.4% prevalence a detector that flags nothing is 96.6% accurate. Analysts triage a fixed queue, so precision@k is the operational metric.

## 2. Data

- **Source** — KDD Cup 1999 network intrusion (SA subset)
- **Rows** — 80,000 in the curated file; 80,000 after preparation
- **Licence** — Public domain (DARPA/MIT Lincoln Labs)
- **Quality** — grade A at 96.6% across six dimensions
- **Missingness** — 0 of 1,040,000 cells (0.00%); 100.0% of rows complete

The curation rule and a SHA-256 for this file are recorded in [`data/MANIFEST.json`](../../../data/MANIFEST.json). Ingestion applies only column selection, dtype coercion, deterministic subsampling and stable sorting — no imputation, scaling or target-aware filtering.

## 3. Data Understanding

80,000 connections, 39 features, 3.37% attacks spanning 10 named families.

**Findings.**

- *Byte counts span nine orders of magnitude* — src_bytes ranges from 0 to hundreds of megabytes. Log transforms and a robust scaler are required before any distance-based method.
- *Attack families are extremely unbalanced among themselves* — 10 families, most with a handful of instances. Aggregate recall would hide entire rare classes; per-family recall is reported instead.

**Decisions.**

- **Keep the label strictly out of the feature frame.** Removes any code path where a detector could see it. *Rejected:* Drop the label at fit time — One forgotten column selection and the whole project silently becomes supervised.

## 4. Data Preparation

RobustScaler on the numerics, one-hot on protocol/service/flag, log transforms on byte counts. Fitted once, on unlabelled data.

**Findings.**

- *StandardScaler would suppress the signal* — mean and variance of src_bytes are set by the outliers themselves. RobustScaler used: median and IQR are unmoved by the very connections we are hunting.

**Decisions.**

- **RobustScaler with a 5-95 percentile range.** Anomalies must stay far from the centre after scaling. *Rejected:* StandardScaler — Normalising by outlier-inflated variance shrinks exactly the deviations being detected.

## 5. Modelling

Five detectors with genuinely different definitions of 'anomalous', all fitted on identical features. Best single: Isolation Forest at PR-AUC 0.5056. Rank fusion did not beat it.

**Leaderboard** — scored by average_precision.

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| — | 0.5056 | 0.9659 |
| — | 0.0403 | 0.2388 |
| — | 0.0335 | 0.3372 |
| — | 0.0275 | 0.2779 |
| — | — | — |

> All five detectors are unsupervised and fitted on identical transformed features. Labels are used only to score the resulting rankings.

## 6. Evaluation

Isolation Forest achieves PR-AUC 0.474 against a 0.034 base rate -- 14x random. In a 100-alert queue, 56% of alerts are real intrusions.

**Headline results**

| Measure | Value |
|---|---|
| Pr auc | 0.4739 |
| Roc auc | 0.9659 |
| Attack rate | 0.0337 |
| Lift over random | 14.1 |
| Precision at 100 | 0.56 |
| Detectors compared | 4 |

**Baselines the model had to beat**

| Baseline | Score | What it does |
|---|---|---|
| Random ranking | 0.0337 | The floor: equals the attack rate by construction. |
| Rule: high SYN error rate | 0.2538 | Rank by connection error rate alone -- the classic hand-written scan signature. |
| Rule: unusual payload size | 0.0681 | Rank by smallest source payload. |

## 7. Deployment

Scored connections are served at /api/projects/anomaly/predict, returning a percentile-normalised anomaly score and the features that deviate most from normal.

Served at `POST /api/projects/anomaly/predict`. The fitted pipeline is pickled whole, so preprocessing travels with the model and training and serving cannot drift apart.

## 8. Limitations

- KDD-99 is a simulated 1999 dataset with well-documented redundancy; absolute numbers here should not be read as contemporary performance.
- Attack families smurf, back, ipsweep are not surfaced at realistic alert budgets -- they need signature rules.
- Contamination is a stated assumption, not a measurement.
- No temporal structure is modelled: each connection is scored independently, so slow multi-stage intrusions are invisible by construction.

**Out of scope**

- Automated blocking. These are unsupervised anomaly scores with a substantial false-positive rate; they prioritise human review, they do not authorise action.
- Modern network traffic. KDD-99 is a 1999 simulated environment with protocols and attack patterns that no longer reflect reality.
- Any claim that a flagged connection is an attack -- it is merely unusual.

**Ethical considerations**

- Anomaly scores flag deviation from the majority, which in a human-facing context means flagging the unusual rather than the harmful. Applied to people rather than packets, that distinction is the whole ethical problem.
- A high false-positive rate is acceptable for a triage queue and unacceptable for an automated block; the deployment boundary matters more than the metric.

## 9. Audit

10 of 10 checks pass (grade A). *Scope:* Review for the failure modes specific to unsupervised detection: label leakage through hyperparameters, accuracy theatre under extreme imbalance, and aggregate metrics concealing blind classes.

| Check | Status | Evidence |
|---|---|---|
| Labels never used for fitting | pass | prepare() asserts the label columns are absent from X; all five detectors receive the transformed feature matrix only. |
| Contamination not set from observed rate | pass | Set to the assumed 2%, not the true 3.37% -- passing the truth would be leakage through a hyperparameter. |
| Metric appropriate to extreme imbalance | pass | PR-AUC and precision@k are the headline; accuracy is flagged as misleading at a 3.4% base rate. |
| Scaler robust to the signal | pass | RobustScaler(5-95) used so outlier-inflated variance does not shrink the deviations being detected. |
| Multiple detector assumptions compared | pass | 4 detectors with distinct definitions of anomaly, each with its blind spot documented. |
| Fusion evaluated, not assumed | pass | Rank fusion did not improve on the best single detector and was deployed accordingly. |
| Per-class performance reported | pass | Per-family recall published, including families with zero recall. |
| Baselines published | pass | Random ranking and hand-written signature rules both reported. |
| Dataset limitations disclosed | pass | Model card states prominently that KDD-99 is a 1999 simulation and that results are a methodology demonstration. |
| Determinism | pass | All detectors and samplers seeded at 42. |

**Phase gates:** 6 of 6 passed.

- PASS — 1. Business Understanding
- PASS — 2. Data Understanding
- PASS — 3. Data Preparation
- PASS — 4. Modeling
- PASS — 5. Evaluation: PR-AUC 0.474 against a 0.034 base rate.
- PASS — 6. Deployment

## 10. Where this project looped back

CRISP-DM is iterative. These are the points where a later phase sent the work back to an earlier one — normally the part deleted before publication.

- The first run passed the observed attack rate as contamination and produced flattering numbers. The audit caught it as leakage through a hyperparameter, which is why contamination is now a stated 2% assumption.
- Per-family recall was added after aggregate PR-AUC looked strong; it revealed that whole attack classes sit inside the normal density and cannot be found by any outlier method.

## Reproduction

```bash
python scripts/fetch_data.py
python scripts/train_all.py --only anomaly
```

Every estimator, split and sampler is seeded, so a rerun on the same data reproduces this leaderboard exactly.
