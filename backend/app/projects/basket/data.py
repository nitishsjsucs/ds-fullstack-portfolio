"""Basket construction for association rule mining.

Association mining has an unusual failure mode: it is trivially easy to produce
thousands of "rules" that are all true and all useless. Three preparation
decisions do most of the work in avoiding that.

**The unit of analysis is the invoice, not the line.** A basket is the set of
distinct products bought together; quantity is irrelevant to co-occurrence.

**The item vocabulary is capped.** With 3,925 SKUs the one-hot matrix is 79M
cells and the vast majority of columns appear in under 0.1% of baskets, where no
support threshold can produce a stable estimate. Restricting to the most frequent
items is not a shortcut -- it is the only way the support statistics mean
anything. The cut is stated and its coverage reported.

**Single-item baskets are dropped.** A basket of one contributes nothing to
co-occurrence but inflates the denominator of every support calculation, which
deflates every rule's support toward zero and makes thresholds hard to reason
about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core import paths

# Items must clear this share of baskets to enter the vocabulary. Below roughly
# this level, support estimates rest on too few baskets to be stable.
TOP_N_ITEMS = 480
MIN_BASKET_SIZE = 2


def load_raw() -> pd.DataFrame:
    return pd.read_parquet(paths.curated("online_retail_transactions"))


def build_baskets(df: pd.DataFrame | None = None,
                  *, top_n: int = TOP_N_ITEMS,
                  min_basket_size: int = MIN_BASKET_SIZE) -> tuple[pd.DataFrame, dict]:
    """Return a boolean invoice x item matrix plus the preparation audit trail."""
    df = load_raw() if df is None else df
    n_lines = len(df)
    n_invoices_raw = df["invoice_no"].nunique()
    n_items_raw = df["description"].nunique()

    freq = df.groupby("description")["invoice_no"].nunique().sort_values(ascending=False)
    vocab = list(freq.head(top_n).index)
    coverage_lines = float(df["description"].isin(vocab).mean())

    sub = df[df["description"].isin(vocab)]
    pairs = sub[["invoice_no", "description"]].drop_duplicates()

    sizes = pairs.groupby("invoice_no").size()
    keep = sizes[sizes >= min_basket_size].index
    pairs = pairs[pairs["invoice_no"].isin(keep)]

    matrix = pd.crosstab(pairs["invoice_no"], pairs["description"]).astype(bool)

    audit = {
        "source_lines": int(n_lines),
        "source_invoices": int(n_invoices_raw),
        "source_items": int(n_items_raw),
        "vocabulary_size": len(vocab),
        "vocabulary_rule": f"top {top_n} items by number of distinct invoices",
        "line_coverage": round(coverage_lines, 4),
        "baskets_modelled": int(matrix.shape[0]),
        "baskets_dropped_singleton": int(n_invoices_raw - matrix.shape[0]),
        "min_basket_size": min_basket_size,
        "mean_basket_size": round(float(matrix.sum(axis=1).mean()), 2),
        "median_basket_size": int(matrix.sum(axis=1).median()),
        "matrix_density": round(float(matrix.to_numpy().mean()), 5),
        "rationale": (
            f"The vocabulary is capped at {top_n} items covering {coverage_lines:.1%} of "
            f"all line items. The excluded tail appears in too few baskets for a support "
            f"estimate to be stable, and including it would inflate the itemset lattice "
            f"by orders of magnitude for rules no merchandiser could act on. Baskets of "
            f"fewer than {min_basket_size} distinct items are dropped because they carry "
            f"no co-occurrence information while deflating every support figure."
        ),
    }
    return matrix, audit


def basket_size_distribution(matrix: pd.DataFrame) -> list[dict]:
    sizes = matrix.sum(axis=1)
    counts = sizes.value_counts().sort_index()
    return [{"basket_size": int(k), "n_baskets": int(v)}
            for k, v in counts.items() if k <= 40]


def item_frequency(matrix: pd.DataFrame, top: int = 25) -> list[dict]:
    freq = matrix.sum(axis=0).sort_values(ascending=False)
    n = len(matrix)
    return [{"item": str(i), "baskets": int(v), "support": round(float(v / n), 5)}
            for i, v in freq.head(top).items()]


def rule_quality(support_ab: float, support_a: float, support_b: float) -> dict:
    """The full metric family for one rule, with the interpretation attached.

    Confidence alone is the classic trap: a rule "X -> POPULAR_ITEM" can have 80%
    confidence purely because POPULAR_ITEM is in 80% of baskets, telling you
    nothing. Lift, leverage, conviction and Zhang's metric all correct for the
    consequent's base rate in different ways.
    """
    conf = support_ab / support_a if support_a > 0 else 0.0
    lift = conf / support_b if support_b > 0 else 0.0
    leverage = support_ab - support_a * support_b
    conviction = ((1 - support_b) / (1 - conf)) if conf < 1 else float("inf")
    denom = max(conf * (1 - support_b), support_b * (1 - conf))
    zhang = (conf - support_b) / denom if denom > 0 else 0.0
    return {
        "confidence": round(conf, 5),
        "lift": round(lift, 5),
        "leverage": round(leverage, 6),
        "conviction": (None if not np.isfinite(conviction) else round(conviction, 4)),
        "zhangs_metric": round(zhang, 4),
    }


def interpret_lift(lift: float) -> str:
    if lift >= 3:
        return "strong positive dependence -- worth a physical or on-site adjacency"
    if lift >= 1.5:
        return "moderate positive dependence -- viable cross-sell prompt"
    if lift > 1.1:
        return "weak positive dependence -- likely not worth acting on alone"
    if lift > 0.9:
        return "effectively independent -- the rule restates base rates"
    return "negative dependence -- these items substitute rather than complement"
