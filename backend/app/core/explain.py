"""Model explainability: global importance, SHAP attributions and partial dependence.

Three complementary views, because each answers a different question and each can
mislead alone:

* **Permutation importance** -- "how much does test performance degrade if this
  feature is shuffled?" Model-agnostic and measured on held-out data, so unlike
  a tree's built-in ``feature_importances_`` it cannot reward a feature the model
  merely *split on often* while gaining nothing.
* **SHAP** -- "for this one prediction, how did each feature push the output away
  from the base rate?" Additive and locally exact for trees.
* **Partial dependence** -- "what shape is the learned relationship?" This is
  where you discover the model learned a step function where the physics is
  smooth.

Permutation importance is deliberately the *headline* number in the dashboards.
Impurity importance is computed too, but shown next to it precisely so the
divergence is visible: a feature that ranks high on impurity and near-zero on
permutation is a memorisation smell worth auditing.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance


def permutation_payload(model, X: pd.DataFrame, y, *, scoring: str,
                        n_repeats: int = 8, seed: int = 42,
                        max_samples: int = 6000, top_k: int = 20) -> dict:
    """Permutation importance on held-out rows, with dispersion across repeats."""
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=seed)
        y = np.asarray(y)[X.index.to_numpy()] if isinstance(y, np.ndarray) else y.loc[X.index]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(
            model, X, y, scoring=scoring, n_repeats=n_repeats,
            random_state=seed, n_jobs=1,
        )
    rows = [
        {
            "feature": str(col),
            "importance": float(r.importances_mean[i]),
            "std": float(r.importances_std[i]),
            # A feature whose mean drop is smaller than its own noise band is not
            # distinguishable from an irrelevant one.
            "significant": bool(r.importances_mean[i] > 2 * r.importances_std[i]),
        }
        for i, col in enumerate(X.columns)
    ]
    rows.sort(key=lambda d: -d["importance"])
    n_sig = sum(1 for r_ in rows if r_["significant"])
    return {
        "method": "permutation importance (held-out)",
        "scoring": scoring,
        "n_repeats": n_repeats,
        "n_samples": int(len(X)),
        "features": rows[:top_k],
        "n_significant": n_sig,
        "note": (
            f"{n_sig} of {len(rows)} features shift {scoring} by more than twice their "
            "own permutation noise. Measured on held-out rows, so it reflects "
            "generalisation rather than how often a split was taken."
        ),
    }


def impurity_payload(model, feature_names: Sequence[str], *, top_k: int = 20) -> dict | None:
    """Tree impurity importance -- shown only for contrast with permutation."""
    est = getattr(model, "named_steps", {}).get("model", model) if hasattr(model, "named_steps") else model
    imp = getattr(est, "feature_importances_", None)
    if imp is None or len(imp) != len(feature_names):
        return None
    rows = [{"feature": str(f), "importance": float(v)} for f, v in zip(feature_names, imp)]
    rows.sort(key=lambda d: -d["importance"])
    return {
        "method": "mean impurity decrease (in-sample)",
        "features": rows[:top_k],
        "caveat": (
            "Impurity importance is biased toward high-cardinality and continuous "
            "features and is computed on training data. Compare against permutation "
            "importance; large disagreement indicates memorisation."
        ),
    }


def _tree_shap(est, X: pd.DataFrame) -> tuple[np.ndarray, float] | None:
    """Exact TreeSHAP values plus the base value, or ``None`` if unsupported.

    Two paths, in order of preference:

    1. ``shap.TreeExplainer`` -- works for scikit-learn ensembles and LightGBM.
    2. XGBoost's own ``pred_contribs=True`` -- exact TreeSHAP implemented inside
       the booster, and the *only* reliable path across XGBoost major versions.
       The shap package parses XGBoost's serialised model directly, so a change
       to that format (as happened between XGBoost 2 and 3, where ``base_score``
       began serialising as ``'[1.5029671E1]'``) breaks TreeExplainer while the
       booster's own implementation keeps working.

    Going through the booster is not a workaround for a bug we could not fix; it
    removes a version coupling between two independently-released packages from a
    code path the dashboards depend on.
    """
    # XGBoost first: its native path has no cross-package version coupling.
    try:
        import xgboost as xgb

        if isinstance(est, (xgb.XGBRegressor, xgb.XGBClassifier)):
            booster = est.get_booster()
            dm = xgb.DMatrix(X, feature_names=list(X.columns))
            contribs = booster.predict(dm, pred_contribs=True)
            contribs = np.asarray(contribs)
            if contribs.ndim == 3:          # multi-class: take the positive class
                contribs = contribs[:, -1, :]
            # The final column is the bias term, identical for every row.
            return contribs[:, :-1], float(contribs[0, -1])
    except Exception:
        pass

    try:
        import shap
    except ImportError:  # pragma: no cover
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(est)
            values = np.asarray(explainer.shap_values(X))
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base).ravel()[-1])
        if values.ndim == 3:
            values = values[:, :, -1]
        return values, float(base)
    except Exception:
        return None


def shap_payload(model, X_background: pd.DataFrame, X_explain: pd.DataFrame,
                 *, max_background: int = 200, top_k: int = 14,
                 seed: int = 42) -> dict | None:
    """Global mean-|SHAP| plus per-instance attributions for a few examples.

    Returns ``None`` only when the estimator genuinely has no tree structure to
    explain. A *failure* to compute SHAP on a tree model returns a payload with
    an ``error`` key instead, so the dashboard says "SHAP unavailable because X"
    rather than silently dropping the panel -- a missing chart that nobody
    notices is worse than a visible error.
    """
    est = model
    if hasattr(model, "named_steps"):
        est = model.named_steps.get("model", model)

    if not hasattr(est, "get_booster") and not hasattr(est, "estimators_") \
            and not hasattr(est, "tree_") and not hasattr(est, "_predictors"):
        return None  # not a tree ensemble; nothing to explain with TreeSHAP

    result = _tree_shap(est, X_explain)
    if result is None:
        return {
            "method": "TreeSHAP",
            "error": (
                f"TreeSHAP could not be computed for {type(est).__name__}. This is "
                f"usually a version mismatch between the shap package and the booster "
                f"library. Permutation importance on the Explain tab is unaffected."
            ),
        }
    values, base = result

    if values.shape[1] != X_explain.shape[1]:
        return {
            "method": "TreeSHAP",
            "error": (
                f"TreeSHAP returned {values.shape[1]} attributions for "
                f"{X_explain.shape[1]} features; refusing to display a misaligned "
                f"explanation."
            ),
        }

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_k]
    cols = list(X_explain.columns)

    # Beeswarm sample: SHAP value vs the feature's own (rank-normalised) value,
    # which is what makes direction readable -- "high income pushes score up".
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(X_explain), size=min(180, len(X_explain)), replace=False)
    beeswarm = []
    for j in order[:8]:
        col = cols[j]
        raw = pd.to_numeric(X_explain.iloc[pick, j], errors="coerce")
        rank = raw.rank(pct=True).fillna(0.5).to_numpy()
        beeswarm.append({
            "feature": str(col),
            "points": [{"shap": float(values[pick[i], j]), "value_rank": float(rank[i])}
                       for i in range(len(pick))],
        })

    # Waterfall for three instances spanning the prediction range.
    totals = values.sum(axis=1)
    picks = {"lowest": int(np.argmin(totals)),
             "median": int(np.argsort(totals)[len(totals) // 2]),
             "highest": int(np.argmax(totals))}
    waterfalls = {}
    for label, i in picks.items():
        contribs = [
            {"feature": str(cols[j]),
             "value": _fmt(X_explain.iloc[i, j]),
             "shap": float(values[i, j])}
            for j in np.argsort(-np.abs(values[i]))[:10]
        ]
        waterfalls[label] = {
            "base_value": base,
            "prediction": float(base + totals[i]),
            "contributions": contribs,
            "residual_other_features": float(
                totals[i] - sum(c["shap"] for c in contribs)
            ),
        }

    return {
        "method": "TreeSHAP (exact Shapley values for tree ensembles)",
        "base_value": base,
        "global_importance": [
            {"feature": str(cols[j]), "mean_abs_shap": float(mean_abs[j])} for j in order
        ],
        "beeswarm": beeswarm,
        "waterfalls": waterfalls,
        "n_explained": int(len(X_explain)),
        "identity": (
            "f(x) = E[f(X)] + sum_i phi_i(x): the base value plus every feature's "
            "contribution reconstructs the prediction exactly."
        ),
    }


def local_explanation(model, X_background: pd.DataFrame, row: pd.DataFrame,
                      *, top_k: int = 10) -> dict | None:
    """Explain a single live prediction for the inference UI."""
    est = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    result = _tree_shap(est, row)
    if result is None:
        return None
    values, base = result
    vals = values[0]
    cols = list(row.columns)
    order = np.argsort(-np.abs(vals))[:top_k]
    shown = [
        {"feature": str(cols[j]), "value": _fmt(row.iloc[0, j]), "shap": float(vals[j])}
        for j in order
    ]
    return {
        "base_value": float(base),
        "prediction": float(base + vals.sum()),
        "contributions": shown,
        # The waterfall shows the top-k features; the rest still contribute, and
        # saying so by how much is what keeps the additivity claim honest.
        "residual_other_features": float(vals.sum() - sum(c["shap"] for c in shown)),
        "n_features_total": int(len(cols)),
    }


def pdp_payload(model, X: pd.DataFrame, features: Sequence[str],
                *, grid_resolution: int = 24, max_samples: int = 2500,
                seed: int = 42) -> list[dict]:
    """Partial dependence curves, with a decile rug so sparse regions are visible."""
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=seed)
    out = []
    for feat in features:
        if feat not in X.columns:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd_result = partial_dependence(
                    model, X, [feat], grid_resolution=grid_resolution, kind="average"
                )
            grid = np.asarray(pd_result["grid_values"][0], dtype=float)
            avg = np.asarray(pd_result["average"][0], dtype=float)
        except Exception:
            continue
        raw = pd.to_numeric(X[feat], errors="coerce").dropna()
        out.append({
            "feature": str(feat),
            "curve": [{"x": float(g), "y": float(v)} for g, v in zip(grid, avg)],
            "deciles": [float(raw.quantile(q)) for q in np.linspace(0.1, 0.9, 9)] if len(raw) else [],
            "effect_range": float(avg.max() - avg.min()),
            "monotone": bool(np.all(np.diff(avg) >= -1e-9) or np.all(np.diff(avg) <= 1e-9)),
        })
    out.sort(key=lambda d: -d["effect_range"])
    return out


def _fmt(v):
    """Render a feature value for display without leaking numpy repr noise."""
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return round(f, 4) if np.isfinite(f) else None
    return str(v)
