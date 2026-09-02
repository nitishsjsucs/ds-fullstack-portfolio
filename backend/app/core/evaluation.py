"""Metric computation and chart-ready evaluation payloads.

Each ``*_report`` function returns a plain dict that the API serves verbatim and
the React charts consume directly. Curves are decimated to a fixed number of
points so a 150k-row test set does not ship 150k SVG vertices to the browser.

A deliberate choice runs through this module: **for imbalanced targets the
headline metric is average precision (PR-AUC), never accuracy.** A 3.4% attack
rate makes "96.6% accurate" achievable by predicting nothing, and reporting it
would be exactly the kind of metric theatre the audit is meant to catch. Accuracy
is still computed, but it is labelled as misleading wherever prevalence is skewed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

MAX_CURVE_POINTS = 220


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _decimate(*arrays: np.ndarray, n: int = MAX_CURVE_POINTS) -> list[np.ndarray]:
    """Evenly subsample parallel arrays, always keeping the endpoints."""
    length = len(arrays[0])
    if length <= n:
        return [np.asarray(a) for a in arrays]
    idx = np.unique(np.linspace(0, length - 1, n).astype(int))
    return [np.asarray(a)[idx] for a in arrays]


def _safe(fn, *args, default=None, **kwargs):
    """Metrics that are undefined on degenerate folds should not crash training."""
    try:
        value = fn(*args, **kwargs)
        return None if (isinstance(value, float) and not np.isfinite(value)) else value
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float = 0.5) -> dict:
    """Point metrics at one operating threshold, plus threshold-free rankings."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    prevalence = float(y_true.mean())
    return {
        "threshold": float(threshold),
        "prevalence": prevalence,
        # Threshold-free -- these are the ones we optimise and report.
        "roc_auc": _safe(roc_auc_score, y_true, y_prob),
        "pr_auc": _safe(average_precision_score, y_true, y_prob),
        "brier": _safe(brier_score_loss, y_true, y_prob),
        "log_loss": _safe(log_loss, y_true, np.clip(y_prob, 1e-15, 1 - 1e-15)),
        # Threshold-dependent.
        "precision": _safe(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe(recall_score, y_true, y_pred, zero_division=0),
        "f1": _safe(f1_score, y_true, y_pred, zero_division=0),
        "accuracy": _safe(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe(balanced_accuracy_score, y_true, y_pred),
        "mcc": _safe(matthews_corrcoef, y_true, y_pred),
        # Context for the reader: how good is "always predict majority"?
        "baseline_accuracy": max(prevalence, 1 - prevalence),
        "baseline_pr_auc": prevalence,
        "accuracy_is_misleading": bool(min(prevalence, 1 - prevalence) < 0.20),
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, *, n: int = 101,
                    cost_fn: float = 1.0, cost_fp: float = 1.0) -> dict:
    """Sweep the decision threshold and expose the cost-optimal operating point.

    Choosing a threshold is a business decision, not a modelling one: a missed
    fraud and a false alarm rarely cost the same. We surface the full trade-off
    curve and mark the minimum-cost point under the supplied asymmetry so the UI
    can let a user drag the ratio and watch the recommendation move.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    grid = np.linspace(0.01, 0.99, n)
    rows = []
    for t in grid:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": float(t),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
            "expected_cost": float(cost_fn * fn + cost_fp * fp),
        })
    best_f1 = max(rows, key=lambda r: r["f1"])
    best_cost = min(rows, key=lambda r: r["expected_cost"])
    return {
        "grid": rows,
        "cost_ratio": {"false_negative": cost_fn, "false_positive": cost_fp},
        "best_f1_threshold": best_f1["threshold"],
        "best_cost_threshold": best_cost["threshold"],
        "note": (
            "F1 assumes a false positive and a false negative are equally bad. The "
            "cost-optimal threshold uses the supplied asymmetry and is the one an "
            "operator should actually deploy."
        ),
    }


def classification_report_payload(y_true, y_prob, *, threshold: float = 0.5,
                                  cost_fn: float = 1.0, cost_fp: float = 1.0,
                                  n_calibration_bins: int = 10) -> dict:
    """Everything the classification dashboards need, in one payload."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    fpr_d, tpr_d = _decimate(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    prec_d, rec_d = _decimate(prec, rec)

    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_calibration_bins, strategy="quantile"
        )
        calibration = [{"predicted": float(p), "observed": float(o)}
                       for p, o in zip(mean_pred, frac_pos)]
        ece = float(np.mean(np.abs(np.asarray(frac_pos) - np.asarray(mean_pred))))
    except Exception:
        calibration, ece = [], None

    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    # Lift/gain: how much of the positive mass sits in the top decile of scores?
    order = np.argsort(-y_prob)
    sorted_y = y_true[order]
    cum = np.cumsum(sorted_y)
    total_pos = max(1, int(sorted_y.sum()))
    deciles = []
    for d in range(1, 11):
        k = max(1, int(len(sorted_y) * d / 10))
        captured = float(cum[k - 1] / total_pos)
        deciles.append({
            "decile": d,
            "population_fraction": round(d / 10, 2),
            "captured_positives": captured,
            "lift": round(captured / (d / 10), 3),
        })

    return {
        "metrics": binary_metrics(y_true, y_prob, threshold=threshold),
        "roc_curve": [{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr_d, tpr_d)],
        "pr_curve": [{"recall": float(a), "precision": float(b)} for a, b in zip(rec_d, prec_d)],
        "calibration": calibration,
        "expected_calibration_error": ece,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold_sweep": threshold_sweep(y_true, y_prob, cost_fn=cost_fn, cost_fp=cost_fp),
        "gain_chart": deciles,
        "score_distribution": _score_histogram(y_true, y_prob),
        "n_test": int(len(y_true)),
    }


def _score_histogram(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 24) -> list[dict]:
    """Predicted-probability histogram split by true class -- the separation plot."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (y_prob >= lo) & (y_prob < hi if i < bins - 1 else y_prob <= hi)
        out.append({
            "bin_start": float(lo),
            "bin_end": float(hi),
            "negatives": int(np.sum(in_bin & (y_true == 0))),
            "positives": int(np.sum(in_bin & (y_true == 1))),
        })
    return out


# --------------------------------------------------------------------------- #
# regression
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    # MAPE explodes near zero, so we report it only over rows bounded away from 0
    # and say how many rows that covers rather than quietly dropping them.
    nz = np.abs(y_true) > 1e-6
    mape = float(np.mean(np.abs(resid[nz] / y_true[nz])) * 100) if nz.any() else None
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "medae": float(np.median(np.abs(resid))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": mape,
        "mape_coverage": float(nz.mean()),
        "bias": float(np.mean(resid)),
        "residual_std": float(np.std(resid)),
        "p90_abs_error": float(np.percentile(np.abs(resid), 90)),
        # A model must beat predicting the training mean; R^2 says so implicitly,
        # this says so in the target's own units.
        "baseline_mae_mean": float(mean_absolute_error(y_true, np.full_like(y_true, y_true.mean()))),
    }


def regression_report_payload(y_true, y_pred, *, sample: int = 1200, seed: int = 42) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=min(sample, len(y_true)), replace=False)
    idx.sort()

    counts, edges = np.histogram(resid, bins=40)
    # Calibration for a point forecast: are the quantiles of prediction honest?
    qs = np.linspace(0.05, 0.95, 19)
    quantile_cal = [
        {"quantile": float(q),
         "predicted": float(np.quantile(y_pred, q)),
         "observed": float(np.quantile(y_true, q))}
        for q in qs
    ]
    return {
        "metrics": regression_metrics(y_true, y_pred),
        "scatter": [{"actual": float(y_true[i]), "predicted": float(y_pred[i])} for i in idx],
        "residual_histogram": [
            {"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(counts[i])}
            for i in range(len(counts))
        ],
        "residuals_vs_fitted": [
            {"fitted": float(y_pred[i]), "residual": float(resid[i])} for i in idx
        ],
        "quantile_calibration": quantile_cal,
        "n_test": int(len(y_true)),
    }


# --------------------------------------------------------------------------- #
# forecasting
# --------------------------------------------------------------------------- #
def forecast_metrics(y_true, y_pred, *, y_train=None, seasonality: int = 1) -> dict:
    """Point-forecast accuracy, including the scale-free MASE.

    MASE divides by the in-sample error of a seasonal naive forecast, so a value
    below 1 means the model genuinely beats "repeat last season" -- the only
    baseline that matters in forecasting and the one most portfolio projects skip.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    base = regression_metrics(y_true, y_pred)
    mase = None
    if y_train is not None and len(np.asarray(y_train)) > seasonality:
        yt = np.asarray(y_train, dtype=float)
        scale = np.mean(np.abs(yt[seasonality:] - yt[:-seasonality]))
        if scale > 0:
            mase = float(np.mean(np.abs(y_true - y_pred)) / scale)
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.where(denom > 0, 2 * np.abs(y_true - y_pred) / denom, 0.0)) * 100)
    return {**base, "mase": mase, "smape_pct": smape, "seasonality": seasonality}


def interval_metrics(y_true, lower, upper, *, nominal: float = 0.8) -> dict:
    """Do the prediction intervals actually contain what they claim to?

    An 80% band that covers 55% of outcomes is worse than no band at all, because
    it invites a false sense of precision. Pinball loss scores the quantiles
    themselves; coverage checks the promise.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (y_true >= lower) & (y_true <= upper)
    alpha = (1 - nominal) / 2
    pin_lo = np.mean(np.maximum(alpha * (y_true - lower), (alpha - 1) * (y_true - lower)))
    pin_hi = np.mean(np.maximum((1 - alpha) * (y_true - upper), -alpha * (y_true - upper)))
    return {
        "nominal_coverage": float(nominal),
        "empirical_coverage": float(covered.mean()),
        "coverage_gap": float(covered.mean() - nominal),
        "mean_interval_width": float(np.mean(upper - lower)),
        "pinball_loss": float((pin_lo + pin_hi) / 2),
        "verdict": (
            "well calibrated" if abs(covered.mean() - nominal) <= 0.05
            else ("over-confident (bands too narrow)" if covered.mean() < nominal
                  else "conservative (bands too wide)")
        ),
    }


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def clustering_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    labels = np.asarray(labels)
    mask = labels >= 0  # DBSCAN marks noise as -1; it must not count as a cluster
    n_clusters = int(len(set(labels[mask])))
    if n_clusters < 2 or mask.sum() < 3:
        return {"n_clusters": n_clusters, "silhouette": None, "davies_bouldin": None,
                "calinski_harabasz": None, "noise_fraction": float((~mask).mean())}
    return {
        "n_clusters": n_clusters,
        "silhouette": _safe(silhouette_score, X[mask], labels[mask]),
        "davies_bouldin": _safe(davies_bouldin_score, X[mask], labels[mask]),
        "calinski_harabasz": _safe(calinski_harabasz_score, X[mask], labels[mask]),
        "noise_fraction": float((~mask).mean()),
        "cluster_sizes": pd.Series(labels[mask]).value_counts().sort_index().to_dict(),
    }
