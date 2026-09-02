# Market Basket Association Mining — Abstract

**Task** Association rule mining  ·  **Data** UCI Online Retail (invoice line items)  ·  **Metric** Lift (with conviction & leverage)

Apriori and FP-Growth over 20,136 real invoices and 3,925 SKUs. The system follows the full CRISP-DM cycle on 530,693 rows of UCI Online Retail (invoice line items), selecting Apriori by comparative evaluation on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: baskets 16,785, items 480, frequent itemsets 1,497, rules 1,861. 1,861 rules pass the thresholds. 1672 of 1672 measurable rules still show lift above 1.2 on invoices the miner never saw (189 more could not be scored because an item vanished from the hold-out vocabulary). Note that this archive ends in December, so the chronological hold-out is the Christmas peak: the test rules out patterns that faded across the year, but it cannot refute ones specific to that season. A static leakage audit of the training path passes 9 of 9 checks, and 6 of 6 CRISP-DM phase gates are met.

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

## What this model must not be used for

- Personalised recommendation -- these rules are population-level and have no notion of an individual's taste.
- Causal claims: co-occurrence is not evidence that promoting A will sell B.
- Assortments materially different from this UK gift retailer's.

## Known limitations

- Vocabulary capped at 480 items; the long tail is unrepresented and cannot generate rules at all.
- Purely correlational -- no experiment supports a causal reading.
- Strongly seasonal source period; rules should be re-mined at least quarterly and after any assortment change.
- Anonymous guest checkouts are included as baskets but cannot be linked to a customer, so no repeat-purchase structure is modelled.

Full write-up: [`paper.md`](./paper.md).
