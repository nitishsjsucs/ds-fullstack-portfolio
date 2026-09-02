# Market Basket Association Mining

*Apriori and FP-Growth over 20,136 real invoices and 3,925 SKUs.*

**Task** Association rule mining  ·  **Domain** Retail merchandising  ·  **Primary metric** Lift (with conviction & leverage)

> Generated from the training run of `2026-09-02T20:39:49+00:00` (commit `n/a`, seed 42). Every figure below is read from that run's artifacts.

## Abstract

Apriori and FP-Growth over 20,136 real invoices and 3,925 SKUs. The system follows the full CRISP-DM cycle on 530,693 rows of UCI Online Retail (invoice line items), selecting Apriori by comparative evaluation on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: baskets 16,785, items 480, frequent itemsets 1,497, rules 1,861. 1,861 rules pass the thresholds. 1672 of 1672 measurable rules still show lift above 1.2 on invoices the miner never saw (189 more could not be scored because an item vanished from the hold-out vocabulary). Note that this archive ends in December, so the chronological hold-out is the Christmas peak: the test rules out patterns that faded across the year, but it cannot refute ones specific to that season. A static leakage audit of the training path passes 9 of 9 checks, and 6 of 6 CRISP-DM phase gates are met.

## 1. Business understanding

**Question.** Which products are genuinely bought together, strongly enough to justify changing shelf adjacency, bundle pricing or on-site recommendations?

**Success criteria, fixed before modelling:**

- Rules must show dependence (lift > 1.2), not just co-popularity.
- Rules must survive on invoices the miner never saw.
- Support must rest on enough baskets to be a stable estimate.
- The output must be short enough for a merchandiser to read.

**Why Lift (with conviction & leverage).** Confidence alone promotes rules that merely restate a popular item's base rate; lift and conviction measure genuine dependence.

## 2. Data

- **Source** — UCI Online Retail (invoice line items)
- **Rows** — 530,693 in the curated file; 16,785 after preparation
- **Licence** — CC BY 4.0

The curation rule and a SHA-256 for this file are recorded in [`data/MANIFEST.json`](../../../data/MANIFEST.json). Ingestion applies only column selection, dtype coercion, deterministic subsampling and stable sorting — no imputation, scaling or target-aware filtering.

## 3. Data Understanding

530,693 line items across 20,136 invoices and 4,060 distinct products. The item distribution is a severe long tail.

**Findings.**

- *Product frequency follows a severe long tail* — 480 of 4,060 items cover 49.6% of all line items. Justifies a capped vocabulary without losing meaningful co-occurrence.
- *Volume is strongly seasonal* — monthly invoice counts peak sharply before Christmas. Directly motivates the temporal hold-out: rules mined across the peak may not hold after it.

**Decisions.**

- **Cap the vocabulary at the most frequent items.** Below roughly 1% basket share, support estimates rest on too few baskets to be stable. *Rejected:* Mine all 3,925 SKUs — Explodes the lattice for rules that are statistically noise.

## 4. Data Preparation

A boolean 16,785 x 480 matrix at 3.181% density. Singleton baskets dropped; quantity discarded.

**Findings.**

- *Quantity is irrelevant to co-occurrence* — matrix is boolean by construction. Buying six of an item is still one co-occurrence.
- *3,351 singleton baskets removed* — median basket size 10. They carry no pair information but deflate every support figure.

**Decisions.**

- **Split the hold-out by invoice date, not at random.** Seasonal assortments make a random split leak the period into both halves. *Rejected:* Random 80/20 invoice split — Would confirm seasonal rules as durable.

## 5. Modelling

Apriori and FP-Growth both run at min_support=0.012. Both algorithms return identical itemsets -- they must, because both are exact; only the search strategy differs. On this configuration (480 items at min_support=0.012, 3.2% density) Apriori is 3.2x faster. That is the opposite of the usual textbook ordering when Apriori is Apriori: FP-Growth's advantage comes from avoiding candidate generation, which only pays off once the support floor is low enough for candidates to explode. Here the floor is high, the lattice is small, and building the tree costs more than it saves.

**Leaderboard** — scored by seconds to mine all frequent itemsets (lower is better).

| Model | Seconds |
|---|---|
| Apriori | 7.209 |
| FP-Growth | 23.218 |

> Both algorithms return identical itemsets -- they must, because both are exact; only the search strategy differs. On this configuration (480 items at min_support=0.012, 3.2% density) Apriori is 3.2x faster. That is the opposite of the usual textbook ordering when Apriori is Apriori: FP-Growth's advantage comes from avoiding candidate generation, which only pays off once the support floor is low enough for candidates to explode. Here the floor is high, the lattice is small, and building the tree costs more than it saves.

## 6. Evaluation

1,861 rules pass the thresholds. 1672 of 1672 measurable rules still show lift above 1.2 on invoices the miner never saw (189 more could not be scored because an item vanished from the hold-out vocabulary). Note that this archive ends in December, so the chronological hold-out is the Christmas peak: the test rules out patterns that faded across the year, but it cannot refute ones specific to that season.

**Headline results**

| Measure | Value |
|---|---|
| Baskets | 16,785 |
| Items | 480 |
| Frequent itemsets | 1,497 |
| Rules | 1,861 |
| Max lift | 43.99 |
| Faster algorithm | Apriori |
| Speedup | 3.22 |
| Rules holding out of sample | 1,672 |
| Rule survival rate | 1 |

## 7. Deployment

Rules are served as a lookup: given a basket, return the highest-lift confirmed consequents. No model object is needed at inference -- the rule table is the model.

Served at `POST /api/projects/basket/predict`. The fitted pipeline is pickled whole, so preprocessing travels with the model and training and serving cannot drift apart.

## 8. Limitations

- Vocabulary capped at 480 items; the long tail is unrepresented and cannot generate rules at all.
- Purely correlational -- no experiment supports a causal reading.
- Strongly seasonal source period; rules should be re-mined at least quarterly and after any assortment change.
- Anonymous guest checkouts are included as baskets but cannot be linked to a customer, so no repeat-purchase structure is modelled.

**Out of scope**

- Personalised recommendation -- these rules are population-level and have no notion of an individual's taste.
- Causal claims: co-occurrence is not evidence that promoting A will sell B.
- Assortments materially different from this UK gift retailer's.

**Ethical considerations**

- Cross-sell prompts derived from these rules nudge purchasing. That is ordinary merchandising, but the rules should not be used to construct artificial scarcity or dark-pattern bundling.
- Population-level rules can encode assumptions that do not fit individual shoppers; they belong in ambient placement, not in aggressive targeting.

## 9. Audit

9 of 9 checks pass (grade A). *Scope:* Review for the failure modes specific to pattern mining: rules that restate base rates, seasonal artefacts presented as durable, and undisclosed thresholds that manufacture the result.

| Check | Status | Evidence |
|---|---|---|
| Exact algorithms cross-validated | pass | Apriori and FP-Growth itemsets identical: True. |
| Rules validated out of sample | pass | Rules mined on the first 80% of invoices by date; lift recomputed on the final 20%. A rule is 'confirmed' if its out-of-sample lift still exceeds 1.2. |
| Hold-out split is temporal, not random | pass | Invoices ordered by date; the final 20% form the hold-out, so seasonal patterns cannot appear on both sides. |
| Ranking metric corrects for base rates | pass | Rules ranked by lift; confidence retained only as a filter. Leverage, conviction and Zhang's metric reported alongside. |
| Support floor and vocabulary cut disclosed | pass | The vocabulary is capped at 480 items covering 49.6% of all line items. The excluded tail appears in too few baskets for a support estimate to be stable, and including it would inf |
| No silent truncation | pass | 3,351 singleton baskets and the item tail are both counted and reported in the EDA payload. |
| Failed rules reported | pass | 0 degraded and 189 unmeasurable rules retained in the payload with their status, not dropped. |
| Causal language avoided | pass | Model card states explicitly that co-occurrence is not causation. |
| Determinism | pass | Both miners are deterministic; the vocabulary cut is by rank with a stable sort. |

**Phase gates:** 6 of 6 passed.

- PASS — 1. Business Understanding
- PASS — 2. Data Understanding
- PASS — 3. Data Preparation
- PASS — 4. Modeling
- PASS — 5. Evaluation: 1672 of 1672 measurable rules still show lift above 1.2 on invoices the miner never saw (189 more could not be scored because an item vanished from th
- PASS — 6. Deployment

## 10. Where this project looped back

CRISP-DM is iterative. These are the points where a later phase sent the work back to an earlier one — normally the part deleted before publication.

- The first run mined all 3,925 SKUs at 0.5% support and produced tens of thousands of rules, most resting on a handful of baskets. That is what drove the vocabulary cap and the higher support floor.
- The temporal hold-out was added after noticing that several of the highest-lift rules paired Christmas-specific items -- real patterns, but not ones that justify a permanent shelf change.

## Reproduction

```bash
python scripts/fetch_data.py
python scripts/train_all.py --only basket
```

Every estimator, split and sampler is seeded, so a rerun on the same data reproduces this leaderboard exactly.
