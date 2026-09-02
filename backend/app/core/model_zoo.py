"""Model families and their hill-climbing search grids.

Each factory returns a full ``Pipeline`` (preprocessor + estimator), which is what
makes the leakage guarantee structural rather than aspirational: the preprocessor
is re-fitted inside every cross-validation fold automatically, because it is part
of the estimator scikit-learn is cloning.

The grids are deliberately small and hand-chosen. Hill climbing walks a discrete
neighbourhood, so a grid of three sensible values per hyperparameter explores more
usefully than a hundred random draws -- and it produces a readable trajectory
("max_depth 6 -> 8 bought +0.004") which is the pedagogical point.

Every estimator is seeded. Two runs of ``train_all.py`` on the same data produce
the same leaderboard.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .autoresearch import ModelSpec

SEED = 42


def _pipe(preprocessor, model) -> Pipeline:
    return Pipeline([("prep", preprocessor), ("model", model)])


def _pipe_scaled(preprocessor, model) -> Pipeline:
    """Pipeline with standardisation before a linear model.

    Trees are invariant to monotone rescaling, so they skip this. Linear models
    are not: mixing a 0-60 mile distance with sin/cos terms in [-1, 1] and
    frequency encodings in [0, 1] leaves the normal equations badly conditioned,
    which shows up as overflow warnings and a meaningless coefficient vector.
    Centring and scaling inside the pipeline keeps it fold-safe.
    """
    return Pipeline([
        ("prep", preprocessor),
        ("scale", StandardScaler(with_mean=True)),
        ("model", model),
    ])


# --------------------------------------------------------------------------- #
# optional third-party boosters
# --------------------------------------------------------------------------- #
def has_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def has_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# regression
# --------------------------------------------------------------------------- #
def regression_specs(preprocessor_factory: Callable[[], object],
                     *, include_slow: bool = True) -> list[ModelSpec]:
    """The standard regression tournament: linear baseline -> bagging -> boosting."""
    P = preprocessor_factory
    specs: list[ModelSpec] = [
        ModelSpec(
            name="Ridge regression", family="linear",
            factory=lambda p: _pipe_scaled(P(), Ridge(random_state=SEED, **p)),
            seed_params={"alpha": 1.0},
            grid={"alpha": [0.1, 1.0, 10.0, 100.0]},
            hypothesis=("A regularised linear model is the honest floor. If boosting "
                        "cannot beat it materially, the relationship is linear and the "
                        "extra complexity is not worth deploying."),
        ),
        ModelSpec(
            name="Random forest", family="bagging",
            factory=lambda p: _pipe(P(), RandomForestRegressor(
                random_state=SEED, n_jobs=-1, **p)),
            seed_params={"n_estimators": 160, "max_depth": 16, "min_samples_leaf": 4,
                         "max_features": 0.5},
            grid={"max_depth": [12, 16, 22], "min_samples_leaf": [2, 4, 10],
                  "max_features": [0.4, 0.5, 0.8]},
            hypothesis=("Bagging reduces variance without sequential bias, so it shows "
                        "how much of the signal is available from unordered splits alone."),
        ),
        ModelSpec(
            name="Hist gradient boosting", family="boosting",
            factory=lambda p: _pipe(P(), HistGradientBoostingRegressor(
                random_state=SEED, early_stopping=False, **p)),
            seed_params={"max_iter": 260, "learning_rate": 0.08, "max_depth": None,
                         "min_samples_leaf": 24, "l2_regularization": 0.5},
            grid={"learning_rate": [0.05, 0.08, 0.14], "max_depth": [None, 8, 12],
                  "min_samples_leaf": [12, 24, 48], "l2_regularization": [0.0, 0.5, 2.0]},
            hypothesis=("Histogram boosting is the strongest general-purpose tabular "
                        "learner in scikit-learn and the reference to beat."),
        ),
    ]

    if include_slow:
        specs.insert(2, ModelSpec(
            name="Extra trees", family="bagging",
            factory=lambda p: _pipe(P(), ExtraTreesRegressor(
                random_state=SEED, n_jobs=-1, **p)),
            seed_params={"n_estimators": 180, "max_depth": 20, "min_samples_leaf": 4},
            grid={"max_depth": [14, 20, 28], "min_samples_leaf": [1, 4, 8]},
            hypothesis=("Randomised split thresholds trade a little bias for lower "
                        "variance; it wins when features are noisy."),
        ))

    if has_lightgbm():
        import lightgbm as lgb
        specs.append(ModelSpec(
            name="LightGBM", family="boosting",
            factory=lambda p: _pipe(P(), lgb.LGBMRegressor(
                random_state=SEED, n_jobs=-1, verbose=-1, **p)),
            seed_params={"n_estimators": 350, "learning_rate": 0.06, "num_leaves": 48,
                         "min_child_samples": 25, "subsample": 0.9,
                         "colsample_bytree": 0.9, "reg_lambda": 1.0},
            grid={"learning_rate": [0.04, 0.06, 0.11], "num_leaves": [31, 48, 96],
                  "min_child_samples": [10, 25, 60], "reg_lambda": [0.0, 1.0, 5.0],
                  "colsample_bytree": [0.7, 0.9, 1.0]},
            hypothesis=("Leaf-wise growth fits deep interactions with fewer trees; "
                        "expected to lead on structured tabular data."),
        ))

    if has_xgboost():
        import xgboost as xgb
        specs.append(ModelSpec(
            name="XGBoost", family="boosting",
            factory=lambda p: _pipe(P(), xgb.XGBRegressor(
                random_state=SEED, n_jobs=-1, tree_method="hist",
                verbosity=0, **p)),
            seed_params={"n_estimators": 350, "learning_rate": 0.06, "max_depth": 7,
                         "subsample": 0.9, "colsample_bytree": 0.9,
                         "min_child_weight": 4, "reg_lambda": 1.0},
            grid={"learning_rate": [0.04, 0.06, 0.11], "max_depth": [5, 7, 10],
                  "min_child_weight": [1, 4, 10], "reg_lambda": [0.5, 1.0, 5.0]},
            hypothesis=("Level-wise growth with strong regularisation; the usual "
                        "close rival to LightGBM and a check that the lead is real."),
        ))
    return specs


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def classification_specs(preprocessor_factory: Callable[[], object],
                         *, include_slow: bool = True,
                         class_weight: str | None = None) -> list[ModelSpec]:
    """The standard binary-classification tournament."""
    P = preprocessor_factory
    specs: list[ModelSpec] = [
        ModelSpec(
            name="Logistic regression", family="linear",
            factory=lambda p: _pipe_scaled(P(), LogisticRegression(
                random_state=SEED, max_iter=2000, class_weight=class_weight, **p)),
            seed_params={"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
            grid={"C": [0.05, 0.3, 1.0, 4.0]},
            hypothesis=("A calibrated linear model is often within a point of boosting "
                        "on wide one-hot tabular data, and it is far easier to defend "
                        "to a regulator."),
        ),
        ModelSpec(
            name="Random forest", family="bagging",
            factory=lambda p: _pipe(P(), RandomForestClassifier(
                random_state=SEED, n_jobs=-1, class_weight=class_weight, **p)),
            seed_params={"n_estimators": 220, "max_depth": 14, "min_samples_leaf": 4,
                         "max_features": "sqrt"},
            grid={"max_depth": [8, 14, 22], "min_samples_leaf": [2, 4, 12],
                  "max_features": ["sqrt", 0.4, 0.7]},
            hypothesis="Variance reduction without sequential bias; a robust mid-point.",
        ),
        ModelSpec(
            name="Hist gradient boosting", family="boosting",
            factory=lambda p: _pipe(P(), HistGradientBoostingClassifier(
                random_state=SEED, early_stopping=False, **p)),
            seed_params={"max_iter": 260, "learning_rate": 0.08, "max_depth": None,
                         "min_samples_leaf": 24, "l2_regularization": 0.5},
            grid={"learning_rate": [0.04, 0.08, 0.15], "max_depth": [None, 6, 10],
                  "min_samples_leaf": [12, 24, 60], "l2_regularization": [0.0, 0.5, 2.0]},
            hypothesis="The strongest scikit-learn tabular classifier; the target to beat.",
        ),
    ]
    if include_slow:
        specs.insert(2, ModelSpec(
            name="Extra trees", family="bagging",
            factory=lambda p: _pipe(P(), ExtraTreesClassifier(
                random_state=SEED, n_jobs=-1, class_weight=class_weight, **p)),
            seed_params={"n_estimators": 240, "max_depth": 18, "min_samples_leaf": 3},
            grid={"max_depth": [12, 18, 26], "min_samples_leaf": [1, 3, 8]},
            hypothesis="Extra randomisation as a variance control on noisy categoricals.",
        ))
    if has_lightgbm():
        import lightgbm as lgb
        specs.append(ModelSpec(
            name="LightGBM", family="boosting",
            factory=lambda p: _pipe(P(), lgb.LGBMClassifier(
                random_state=SEED, n_jobs=-1, verbose=-1,
                class_weight=class_weight, **p)),
            seed_params={"n_estimators": 340, "learning_rate": 0.06, "num_leaves": 40,
                         "min_child_samples": 25, "colsample_bytree": 0.9,
                         "reg_lambda": 1.0},
            grid={"learning_rate": [0.03, 0.06, 0.12], "num_leaves": [20, 40, 80],
                  "min_child_samples": [10, 25, 60], "reg_lambda": [0.0, 1.0, 5.0]},
            hypothesis="Leaf-wise boosting; usually the leaderboard leader.",
        ))
    if has_xgboost():
        import xgboost as xgb
        specs.append(ModelSpec(
            name="XGBoost", family="boosting",
            factory=lambda p: _pipe(P(), xgb.XGBClassifier(
                random_state=SEED, n_jobs=-1, tree_method="hist",
                eval_metric="logloss", verbosity=0, **p)),
            seed_params={"n_estimators": 340, "learning_rate": 0.06, "max_depth": 6,
                         "subsample": 0.9, "colsample_bytree": 0.9,
                         "min_child_weight": 3, "reg_lambda": 1.0},
            grid={"learning_rate": [0.03, 0.06, 0.12], "max_depth": [4, 6, 9],
                  "min_child_weight": [1, 3, 8], "reg_lambda": [0.5, 1.0, 4.0]},
            hypothesis="Level-wise boosting; the independent check on LightGBM's lead.",
        ))
    return specs


def quantile_regressors(preprocessor_factory: Callable[[], object],
                        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
                        **params) -> dict[float, Pipeline]:
    """Paired quantile models that turn a point forecast into an honest interval.

    Each model minimises the pinball loss for one quantile, so the 10th and 90th
    together give an 80% band whose empirical coverage we then verify -- a band
    that is not coverage-tested is decoration.
    """
    defaults = {"max_iter": 240, "learning_rate": 0.08, "min_samples_leaf": 24}
    defaults.update(params)
    return {
        q: _pipe(preprocessor_factory(), HistGradientBoostingRegressor(
            loss="quantile", quantile=q, random_state=SEED,
            early_stopping=False, **defaults))
        for q in quantiles
    }
