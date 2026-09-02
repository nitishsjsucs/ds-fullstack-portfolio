# Market Basket Association Mining

*Apriori and FP-Growth over 20,136 real invoices and 3,925 SKUs.*

## The question

Which products are genuinely bought together, strongly enough to justify changing shelf adjacency, bundle pricing or on-site recommendations?

## Why this metric and not accuracy

Confidence alone promotes rules that merely restate a popular item's base rate; lift and conviction measure genuine dependence.

## What the data actually looked like

530,693 line items across 20,136 invoices and 4,060 distinct products. The item distribution is a severe long tail.

**Product frequency follows a severe long tail** — 480 of 4,060 items cover 49.6% of all line items. Justifies a capped vocabulary without losing meaningful co-occurrence.

**Volume is strongly seasonal** — monthly invoice counts peak sharply before Christmas. Directly motivates the temporal hold-out: rules mined across the peak may not hold after it.

## The modelling

Apriori and FP-Growth both run at min_support=0.012. Both algorithms return identical itemsets -- they must, because both are exact; only the search strategy differs. On this configuration (480 items at min_support=0.012, 3.2% density) Apriori is 3.2x faster. That is the opposite of the usual textbook ordering when Apriori is Apriori: FP-Growth's advantage comes from avoiding candidate generation, which only pays off once the support floor is low enough for candidates to explode. Here the floor is high, the lattice is small, and building the tree costs more than it saves.

> **Cap itemsets at length 3.** Longer rules are rarer, less stable and harder to act on; a merchandiser cannot merchandise a 5-way bundle.
>
> *We rejected Unbounded itemset length:* Combinatorial cost for rules nobody deploys.

## Results

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

## The finding worth reading twice

**Confidence alone would rank popular items highest**

any rule ending in a top-10 item inherits its base rate. Ranking switched to lift; confidence retained only as a secondary filter.

## What went wrong first

Every project here records where a later phase sent the work back to an earlier one. That is normally the part deleted before publication, and it is usually the most useful part.

- The first run mined all 3,925 SKUs at 0.5% support and produced tens of thousands of rules, most resting on a handful of baskets. That is what drove the vocabulary cap and the higher support floor.
- The temporal hold-out was added after noticing that several of the highest-lift rules paired Christmas-specific items -- real patterns, but not ones that justify a permanent shelf change.

## What it cannot do

- Personalised recommendation -- these rules are population-level and have no notion of an individual's taste.
- Causal claims: co-occurrence is not evidence that promoting A will sell B.
- Assortments materially different from this UK gift retailer's.

## Try it

The live inference playground for this project is on the **Live inference** tab of the console, or call it directly:

```bash
curl -X POST localhost:8000/api/projects/basket/predict \
  -H 'Content-Type: application/json' -d '{}'
```
