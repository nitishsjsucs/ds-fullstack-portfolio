"""A reusable exploratory-data-analysis profiler.

One function, :func:`profile`, turns any DataFrame into a JSON payload rich enough
to drive a full EDA dashboard: distributions with outlier fences, correlation
matrices, categorical breakdowns, missingness, target relationships and a
six-dimension data-quality scorecard.

Centralising this is the main structural payoff of the monorepo. In the reference
portfolio each project reimplemented its own histogram endpoint; here eight
projects share one profiler, so a fix to the IQR fence logic lands everywhere at
once and the dashboards stay visually consistent.

The profiler is **descriptive only**. It never imputes, clips or drops. If 2,979
taxi trips report zero distance, the scorecard says so and the histogram shows the
spike -- hiding it would defeat the purpose.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_BINS = 28
TOP_K_CATEGORIES = 12
MAX_CORRELATION_COLS = 16


# --------------------------------------------------------------------------- #
# column-level profiles
# --------------------------------------------------------------------------- #
def numeric_profile(s: pd.Series, *, bins: int = DEFAULT_BINS) -> dict:
    """Distribution, moments and Tukey outlier fences for one numeric column."""
    clean = s.dropna().astype(float)
    if clean.empty:
        return {"column": s.name, "kind": "numeric", "count": 0, "empty": True}

    q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = clean[(clean < lo_fence) | (clean > hi_fence)]

    # Histogram over the 0.5th-99.5th percentile range: a single $200k fare would
    # otherwise compress every real bar into the first bucket. We report the
    # clipped tail counts separately so nothing is silently hidden.
    p_lo, p_hi = float(clean.quantile(0.005)), float(clean.quantile(0.995))
    if p_hi <= p_lo:
        p_lo, p_hi = float(clean.min()), float(clean.max()) or 1.0
    body = clean[(clean >= p_lo) & (clean <= p_hi)]
    counts, edges = np.histogram(body, bins=bins, range=(p_lo, p_hi))

    return {
        "column": str(s.name),
        "kind": "numeric",
        "count": int(clean.size),
        "missing": int(s.isna().sum()),
        "missing_pct": float(s.isna().mean()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "skew": float(clean.skew()),
        "kurtosis": float(clean.kurtosis()),
        "zeros": int((clean == 0).sum()),
        "negatives": int((clean < 0).sum()),
        "n_unique": int(clean.nunique()),
        "outlier_fences": {"lower": float(lo_fence), "upper": float(hi_fence)},
        "outlier_count": int(outliers.size),
        "outlier_pct": float(outliers.size / clean.size),
        "histogram": [
            {"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(counts[i])}
            for i in range(len(counts))
        ],
        "histogram_range": {"lower": p_lo, "upper": p_hi,
                            "clipped_below": int((clean < p_lo).sum()),
                            "clipped_above": int((clean > p_hi).sum())},
        "percentiles": {f"p{int(p * 100)}": float(clean.quantile(p))
                        for p in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
    }


def categorical_profile(s: pd.Series, *, top_k: int = TOP_K_CATEGORIES) -> dict:
    counts = s.astype("string").value_counts(dropna=True)
    total = int(counts.sum())
    head = counts.head(top_k)
    return {
        "column": str(s.name),
        "kind": "categorical",
        "count": total,
        "missing": int(s.isna().sum()),
        "missing_pct": float(s.isna().mean()),
        "n_unique": int(counts.size),
        "cardinality_ratio": float(counts.size / max(1, total)),
        "mode": str(counts.index[0]) if counts.size else None,
        "top_values": [
            {"value": str(v), "count": int(c), "share": float(c / total)}
            for v, c in head.items()
        ],
        "tail_share": float(counts.iloc[top_k:].sum() / total) if counts.size > top_k else 0.0,
        # High-cardinality strings are a leakage smell: an ID used as a feature
        # lets a tree memorise rows instead of learning structure.
        "is_high_cardinality": bool(counts.size > 50 and counts.size / max(1, total) > 0.5),
    }


# --------------------------------------------------------------------------- #
# dataset-level structure
# --------------------------------------------------------------------------- #
def correlation_payload(df: pd.DataFrame, cols: Sequence[str]) -> dict:
    """Pearson (linear) and Spearman (monotone) matrices side by side.

    Showing both is the point: a pair that is strongly Spearman-correlated but
    weakly Pearson-correlated is a monotone non-linear relationship, which is a
    direct hint that a tree model will beat a linear one.
    """
    cols = [c for c in cols if c in df.columns][:MAX_CORRELATION_COLS]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.loc[:, sub.notna().any() & (sub.nunique() > 1)]
    if sub.shape[1] < 2:
        return {"columns": [], "pearson": [], "spearman": [], "strong_pairs": []}

    pear = sub.corr(method="pearson").fillna(0.0)
    spear = sub.corr(method="spearman").fillna(0.0)
    names = list(pear.columns)

    strong = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p, s = float(pear.loc[a, b]), float(spear.loc[a, b])
            if abs(p) >= 0.5 or abs(s) >= 0.5:
                strong.append({
                    "a": a, "b": b, "pearson": round(p, 4), "spearman": round(s, 4),
                    "nonlinear_gap": round(abs(s) - abs(p), 4),
                })
    strong.sort(key=lambda r: -max(abs(r["pearson"]), abs(r["spearman"])))

    return {
        "columns": names,
        "pearson": [[round(float(pear.iloc[i, j]), 4) for j in range(len(names))]
                    for i in range(len(names))],
        "spearman": [[round(float(spear.iloc[i, j]), 4) for j in range(len(names))]
                     for i in range(len(names))],
        "strong_pairs": strong[:12],
    }


def missingness_payload(df: pd.DataFrame) -> dict:
    miss = df.isna()
    per_col = miss.mean().sort_values(ascending=False)
    per_col = per_col[per_col > 0]
    # Co-missingness: columns that go blank together usually share one upstream
    # cause, which changes the imputation strategy from per-column to structural.
    pairs = []
    cols = list(per_col.index)[:8]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            both = float((miss[a] & miss[b]).mean())
            if both > 0:
                pairs.append({"a": a, "b": b, "jointly_missing_pct": round(both, 5)})
    return {
        "total_cells": int(df.size),
        "missing_cells": int(miss.sum().sum()),
        "missing_pct": float(miss.sum().sum() / max(1, df.size)),
        "complete_rows_pct": float((~miss.any(axis=1)).mean()),
        "columns": [{"column": c, "missing_pct": float(v), "missing": int(miss[c].sum())}
                    for c, v in per_col.items()],
        "co_missing_pairs": sorted(pairs, key=lambda r: -r["jointly_missing_pct"])[:8],
    }


def target_relationship(df: pd.DataFrame, target: str, features: Sequence[str],
                        *, max_features: int = 10) -> list[dict]:
    """Univariate association between each feature and the target.

    This is exploratory only -- it is computed on the *training* rows the caller
    passes in, and no feature is ever selected on the basis of these numbers,
    because doing that on the full dataset is textbook selection leakage.
    """
    if target not in df.columns:
        return []
    y = df[target]
    y_is_numeric = pd.api.types.is_numeric_dtype(y) and y.nunique() > 10
    out = []
    for col in features[:max_features]:
        if col == target or col not in df.columns:
            continue
        s = df[col]
        try:
            if pd.api.types.is_numeric_dtype(s):
                if y_is_numeric:
                    out.append({"feature": col, "kind": "numeric-vs-numeric",
                                "pearson": round(float(s.corr(y)), 4),
                                "spearman": round(float(s.corr(y, method="spearman")), 4)})
                else:
                    grp = s.groupby(y).mean()
                    out.append({"feature": col, "kind": "numeric-by-class",
                                "means": {str(k): round(float(v), 4) for k, v in grp.items()},
                                "separation": round(float(grp.max() - grp.min()), 4)})
            else:
                if not y_is_numeric:
                    ct = pd.crosstab(s, y, normalize="index")
                    if ct.shape[1] >= 2:
                        pos = ct.columns[-1]
                        rates = ct[pos].sort_values(ascending=False).head(6)
                        out.append({"feature": col, "kind": "rate-by-category",
                                    "positive_class": str(pos),
                                    "rates": {str(k): round(float(v), 4) for k, v in rates.items()},
                                    "spread": round(float(ct[pos].max() - ct[pos].min()), 4)})
                else:
                    grp = y.groupby(s).mean().sort_values(ascending=False).head(6)
                    out.append({"feature": col, "kind": "target-mean-by-category",
                                "means": {str(k): round(float(v), 4) for k, v in grp.items()}})
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# six-dimension quality scorecard
# --------------------------------------------------------------------------- #
def quality_scorecard(df: pd.DataFrame, *, key: str | None = None,
                      time_col: str | None = None,
                      validity_rules: dict[str, str] | None = None) -> dict:
    """Score the frame on completeness, uniqueness, validity, consistency,
    timeliness and cardinality-health, each in [0, 1] with the evidence attached.

    The rules are supplied by the calling project because "valid" is
    domain-specific: a negative fare is invalid for taxis, a negative return is
    perfectly normal for equities.
    """
    dims: list[dict] = []

    completeness = float(1 - df.isna().sum().sum() / max(1, df.size))
    dims.append({
        "dimension": "Completeness",
        "score": completeness,
        "detail": f"{int(df.isna().sum().sum()):,} of {df.size:,} cells are null",
        "worst_column": (df.isna().mean().idxmax() if df.isna().any().any() else None),
    })

    if key and key in df.columns:
        dup = int(df[key].duplicated().sum())
        uniqueness = float(1 - dup / max(1, len(df)))
        detail = f"{dup:,} duplicate values in key column '{key}'"
    else:
        dup = int(df.duplicated().sum())
        uniqueness = float(1 - dup / max(1, len(df)))
        detail = f"{dup:,} fully duplicated rows"
    dims.append({"dimension": "Uniqueness", "score": uniqueness, "detail": detail})

    violations, checked = 0, 0
    rule_results = []
    for col, expr in (validity_rules or {}).items():
        if col not in df.columns:
            continue
        try:
            ok = df.eval(expr)
            bad = int((~ok).sum())
            violations += bad
            checked += len(df)
            rule_results.append({"column": col, "rule": expr, "violations": bad,
                                 "violation_pct": round(bad / max(1, len(df)), 5)})
        except Exception:
            continue
    validity = float(1 - violations / checked) if checked else 1.0
    dims.append({
        "dimension": "Validity",
        "score": validity,
        "detail": (f"{violations:,} domain-rule violations across {len(rule_results)} rules"
                   if rule_results else "no domain rules declared"),
        "rules": rule_results,
    })

    # Consistency: columns that are constant carry no information and usually mean
    # an upstream join went wrong.
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    consistency = float(1 - len(constant) / max(1, df.shape[1]))
    dims.append({"dimension": "Consistency", "score": consistency,
                 "detail": (f"{len(constant)} constant column(s): {constant[:5]}"
                            if constant else "no constant columns")})

    if time_col and time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
        span = t.max() - t.min()
        monotone = bool(t.is_monotonic_increasing)
        gaps = int(t.isna().sum())
        timeliness = float(1 - gaps / max(1, len(df)))
        detail = (f"spans {t.min().date()} to {t.max().date()} ({span.days} days); "
                  f"{'sorted' if monotone else 'UNSORTED'}; {gaps} unparseable timestamps")
    else:
        timeliness, detail = 1.0, "no temporal column declared"
    dims.append({"dimension": "Timeliness", "score": timeliness, "detail": detail})

    obj_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    risky = [c for c in obj_cols if df[c].nunique() > 50 and df[c].nunique() / max(1, len(df)) > 0.5]
    card = float(1 - len(risky) / max(1, len(obj_cols))) if len(obj_cols) else 1.0
    dims.append({"dimension": "Cardinality health", "score": card,
                 "detail": (f"{len(risky)} near-unique text column(s) -- identifier-like, "
                            f"excluded from features: {risky[:4]}"
                            if risky else "no identifier-like text columns among features")})

    overall = float(np.mean([d["score"] for d in dims]))
    return {
        "dimensions": dims,
        "overall_score": overall,
        "grade": ("A" if overall >= 0.95 else "B" if overall >= 0.85
                  else "C" if overall >= 0.7 else "D"),
        "note": ("Scores describe the data as published. Nothing here is repaired at "
                 "the ingestion layer; remediation happens inside the modelling "
                 "pipeline where it can be fitted per-fold."),
    }


# --------------------------------------------------------------------------- #
# top-level entry point
# --------------------------------------------------------------------------- #
def _column(df: pd.DataFrame, name: str) -> pd.Series:
    """Fetch one column even if the caller handed us duplicated column labels.

    ``df[name]`` returns a DataFrame rather than a Series when ``name`` appears
    twice, which then fails deep inside a quantile call with an unhelpful message.
    Collapsing to the first occurrence here keeps the profiler total.
    """
    col = df[name]
    return col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col


def profile(df: pd.DataFrame, *,
            target: str | None = None,
            numeric: Sequence[str] | None = None,
            categorical: Sequence[str] | None = None,
            key: str | None = None,
            time_col: str | None = None,
            validity_rules: dict[str, str] | None = None,
            bins: int = DEFAULT_BINS,
            sample_for_correlation: int = 50_000,
            seed: int = 42) -> dict:
    """Build the complete EDA payload for one dataset."""
    df = df.loc[:, ~df.columns.duplicated()]
    numeric = list(dict.fromkeys(numeric)) if numeric is not None else [
        c for c in df.select_dtypes(include=[np.number]).columns if c != key
    ]
    categorical = list(dict.fromkeys(categorical)) if categorical is not None else [
        c for c in df.select_dtypes(include=["object", "string", "category", "bool"]).columns
        if c != key
    ]

    corr_df = df if len(df) <= sample_for_correlation else df.sample(
        sample_for_correlation, random_state=seed
    )

    return {
        "shape": {"rows": int(len(df)), "columns": int(df.shape[1])},
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 2),
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "numeric": [numeric_profile(df[c], bins=bins) for c in numeric if c in df.columns],
        "categorical": [categorical_profile(df[c]) for c in categorical if c in df.columns],
        "correlation": correlation_payload(corr_df, numeric),
        "missingness": missingness_payload(df),
        "quality": quality_scorecard(df, key=key, time_col=time_col,
                                     validity_rules=validity_rules),
        "target_relationships": target_relationship(
            df, target, [*numeric, *categorical]) if target else [],
        "target": target,
    }


def temporal_heatmap(times: pd.Series, values: pd.Series | None = None,
                     *, agg: str = "count") -> dict:
    """A 7x24 day-of-week x hour matrix -- the standard demand-rhythm view."""
    t = pd.to_datetime(times, errors="coerce")
    frame = pd.DataFrame({"dow": t.dt.dayofweek, "hour": t.dt.hour})
    if values is not None:
        frame["value"] = pd.to_numeric(values, errors="coerce").to_numpy()
        pivot = frame.pivot_table(index="dow", columns="hour", values="value", aggfunc=agg)
    else:
        pivot = frame.pivot_table(index="dow", columns="hour", aggfunc="size")
    pivot = pivot.reindex(index=range(7), columns=range(24)).fillna(0.0)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    matrix = [[float(pivot.iloc[d, h]) for h in range(24)] for d in range(7)]
    flat = [v for row in matrix for v in row]
    peak = max(range(len(flat)), key=lambda i: flat[i]) if flat else 0
    return {
        "days": days,
        "hours": list(range(24)),
        "matrix": matrix,
        "aggregation": agg if values is not None else "count",
        "max": float(max(flat)) if flat else 0.0,
        "peak_cell": {"day": days[peak // 24], "hour": peak % 24,
                      "value": float(flat[peak]) if flat else 0.0},
    }
