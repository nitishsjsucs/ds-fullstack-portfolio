"""Group fairness diagnostics for binary classifiers.

Four criteria, reported together because they are mathematically incompatible and
a project that quotes only the flattering one is not being honest:

* **Demographic parity** -- equal positive-prediction rates across groups.
  Operationalised as the disparate-impact ratio and judged against the US EEOC's
  four-fifths rule.
* **Equal opportunity** -- equal true-positive rates. The right criterion when a
  positive prediction unlocks a benefit and a miss is the harm.
* **Equalised odds** -- equal TPR *and* FPR.
* **Calibration within groups** -- a predicted 0.7 means 70% for every group.

Kleinberg et al. (2016) and Chleeborough/Chouldechova (2017) proved you cannot
satisfy calibration and equalised odds simultaneously unless base rates are equal
or the classifier is perfect. So this module does not emit a single "fair/unfair"
verdict. It reports all four and names which one the project chose to prioritise,
leaving the trade-off visible.

Sensitive attributes are used **only here**, for measurement. They are excluded
from the model's feature set -- though note that removing them does not remove the
bias, because correlated proxies remain, which is exactly why measurement is
necessary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FOUR_FIFTHS = 0.8


def _rates(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict:
    n = len(y_true)
    if n == 0:
        return {}
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    pos, neg = tp + fn, tn + fp
    out = {
        "n": n,
        "base_rate": float(np.mean(y_true)),
        "selection_rate": float(np.mean(y_pred)),
        "tpr": float(tp / pos) if pos else None,
        "fpr": float(fp / neg) if neg else None,
        "tnr": float(tn / neg) if neg else None,
        "fnr": float(fn / pos) if pos else None,
        "precision": float(tp / (tp + fp)) if (tp + fp) else None,
        "accuracy": float((tp + tn) / n),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    if y_prob is not None and n > 20:
        # Calibration within group: mean predicted vs observed.
        out["mean_predicted"] = float(np.mean(y_prob))
        out["observed_rate"] = float(np.mean(y_true))
        out["calibration_error"] = float(abs(np.mean(y_prob) - np.mean(y_true)))
    return out


def group_report(y_true, y_prob, sensitive: pd.Series, *, threshold: float = 0.5,
                 attribute: str = "group", min_group: int = 30) -> dict:
    """Fairness metrics for one sensitive attribute."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    sensitive = pd.Series(sensitive).astype(str).reset_index(drop=True)

    groups = {}
    for value, idx in sensitive.groupby(sensitive).groups.items():
        pos = np.asarray(idx)
        if len(pos) < min_group:
            continue
        groups[str(value)] = _rates(y_true[pos], y_pred[pos], y_prob[pos])

    if len(groups) < 2:
        return {"attribute": attribute, "groups": groups,
                "note": f"Fewer than two groups of at least {min_group} rows; "
                        "fairness metrics are not meaningful at this sample size."}

    sel = {g: v["selection_rate"] for g, v in groups.items()}
    tpr = {g: v["tpr"] for g, v in groups.items() if v["tpr"] is not None}
    fpr = {g: v["fpr"] for g, v in groups.items() if v["fpr"] is not None}

    ref = max(sel, key=lambda g: sel[g])           # most-selected group is reference
    di = {g: (sel[g] / sel[ref] if sel[ref] > 0 else None) for g in sel}
    worst_di = min((v for v in di.values() if v is not None), default=None)

    return {
        "attribute": attribute,
        "threshold": float(threshold),
        "groups": groups,
        "reference_group": ref,
        "disparate_impact": {g: (None if v is None else round(v, 4)) for g, v in di.items()},
        "min_disparate_impact": None if worst_di is None else round(worst_di, 4),
        "passes_four_fifths": bool(worst_di is not None and worst_di >= FOUR_FIFTHS),
        "demographic_parity_difference": round(max(sel.values()) - min(sel.values()), 4),
        "equal_opportunity_difference": (
            round(max(tpr.values()) - min(tpr.values()), 4) if len(tpr) >= 2 else None
        ),
        "equalised_odds_difference": (
            round(max(max(tpr.values()) - min(tpr.values()),
                      max(fpr.values()) - min(fpr.values())), 4)
            if len(tpr) >= 2 and len(fpr) >= 2 else None
        ),
        "base_rate_difference": round(
            max(v["base_rate"] for v in groups.values())
            - min(v["base_rate"] for v in groups.values()), 4
        ),
        "calibration_spread": round(
            max(v.get("calibration_error", 0) for v in groups.values())
            - min(v.get("calibration_error", 0) for v in groups.values()), 4
        ),
    }


def audit(y_true, y_prob, sensitive_frame: pd.DataFrame, *, threshold: float = 0.5,
          priority: str = "equal opportunity") -> dict:
    """Run the report across every declared sensitive attribute."""
    reports = [
        group_report(y_true, y_prob, sensitive_frame[col], threshold=threshold,
                     attribute=col)
        for col in sensitive_frame.columns
    ]
    failures = [r for r in reports if r.get("passes_four_fifths") is False]
    real = [r for r in reports if "min_disparate_impact" in r]
    return {
        "attributes": reports,
        "n_attributes": len(reports),
        "four_fifths_failures": [r["attribute"] for r in failures],
        "worst_disparate_impact": (
            min((r["min_disparate_impact"] for r in real
                 if r["min_disparate_impact"] is not None), default=None)
        ),
        "prioritised_criterion": priority,
        "impossibility_note": (
            "Demographic parity, equalised odds and calibration cannot all hold at "
            "once when base rates differ across groups (Kleinberg et al. 2016). All "
            f"four are reported; this project prioritises {priority}, and the "
            "trade-off is a stated choice rather than an oversight."
        ),
        "measurement_only_note": (
            "Sensitive attributes are excluded from the model's feature set and used "
            "solely for this audit. Exclusion does not eliminate bias -- correlated "
            "proxies survive -- which is precisely why it has to be measured."
        ),
    }


def threshold_equalisation(y_true, y_prob, sensitive: pd.Series,
                           *, target: str = "tpr", grid: int = 81) -> dict:
    """Find per-group thresholds that equalise a chosen rate.

    Presented as a diagnostic, not a recommendation. Group-specific thresholds
    equalise the metric but mean two people with identical predicted risk get
    different decisions, which is itself contestable and in several jurisdictions
    unlawful. The point is to show what parity would cost.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    sensitive = pd.Series(sensitive).astype(str).reset_index(drop=True)
    thresholds = np.linspace(0.05, 0.95, grid)

    # Global reference rate at the default 0.5 cut.
    base_pred = (y_prob >= 0.5).astype(int)
    if target == "tpr":
        pos = y_true == 1
        reference = float(base_pred[pos].mean()) if pos.any() else 0.5
    else:
        reference = float(base_pred.mean())

    per_group = {}
    for value, idx in sensitive.groupby(sensitive).groups.items():
        pos = np.asarray(idx)
        if len(pos) < 30:
            continue
        yt, yp = y_true[pos], y_prob[pos]
        best_t, best_gap = 0.5, float("inf")
        for t in thresholds:
            pred = (yp >= t).astype(int)
            if target == "tpr":
                mask = yt == 1
                rate = float(pred[mask].mean()) if mask.any() else 0.0
            else:
                rate = float(pred.mean())
            gap = abs(rate - reference)
            if gap < best_gap:
                best_gap, best_t = gap, float(t)
        per_group[str(value)] = {"threshold": round(best_t, 3),
                                 "residual_gap": round(best_gap, 4)}
    return {
        "target_metric": target,
        "reference_rate": round(reference, 4),
        "per_group_threshold": per_group,
        "caveat": (
            "Group-specific thresholds equalise the chosen rate but give different "
            "decisions to individuals with identical predicted risk. Shown to "
            "quantify the trade-off, not as a deployment recommendation."
        ),
    }
