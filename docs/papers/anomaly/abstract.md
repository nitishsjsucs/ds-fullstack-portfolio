# Network Intrusion Anomaly Detection — Abstract

**Task** Anomaly / outlier detection  ·  **Data** KDD Cup 1999 network intrusion (SA subset)  ·  **Metric** PR-AUC + precision@k

Unsupervised outlier detection on 80,000 KDD-99 connections with a 3.4% attack base rate. The system follows the full CRISP-DM cycle on 80,000 rows of KDD Cup 1999 network intrusion (SA subset), selecting Isolation Forest by comparative evaluation on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: pr auc 0.4739, roc auc 0.9659, attack rate 0.0337, lift over random 14.1. Isolation Forest achieves PR-AUC 0.474 against a 0.034 base rate -- 14x random. In a 100-alert queue, 56% of alerts are real intrusions. A static leakage audit of the training path passes 10 of 10 checks, and 6 of 6 CRISP-DM phase gates are met.

## Results

| Measure | Value |
|---|---|
| Pr auc | 0.4739 |
| Roc auc | 0.9659 |
| Attack rate | 0.0337 |
| Lift over random | 14.1 |
| Precision at 100 | 0.56 |
| Detectors compared | 4 |

## What this model must not be used for

- Automated blocking. These are unsupervised anomaly scores with a substantial false-positive rate; they prioritise human review, they do not authorise action.
- Modern network traffic. KDD-99 is a 1999 simulated environment with protocols and attack patterns that no longer reflect reality.
- Any claim that a flagged connection is an attack -- it is merely unusual.

## Known limitations

- KDD-99 is a simulated 1999 dataset with well-documented redundancy; absolute numbers here should not be read as contemporary performance.
- Attack families smurf, back, ipsweep are not surfaced at realistic alert budgets -- they need signature rules.
- Contamination is a stated assumption, not a measurement.
- No temporal structure is modelled: each connection is scored independently, so slow multi-stage intrusions are invisible by construction.

Full write-up: [`paper.md`](./paper.md).
