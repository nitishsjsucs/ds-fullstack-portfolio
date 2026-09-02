"""Reusable feature transformers.

The rule this module exists to enforce: **any transformation that learns
something from the data must be a fitted estimator inside a Pipeline**, never a
pandas expression applied to the full frame before splitting.

Frequency encoding is the classic trap. Writing
``df["zone_freq"] = df.zone.map(df.zone.value_counts())`` before the split lets
every training row see how often its zone appears in the *test* set. It is a small
leak, it never shows up as an error, and it inflates the score. :class:`FrequencyEncoder`
does the same job correctly by fitting on the training fold only.

Row-wise transformations that learn nothing (cyclical time encoding, haversine
distance) are safe to apply during data preparation and are provided here as plain
functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

EARTH_RADIUS_KM = 6371.0088


# --------------------------------------------------------------------------- #
# stateless row-wise helpers (safe before the split)
# --------------------------------------------------------------------------- #
def add_cyclical(df: pd.DataFrame, col: str, period: int, *, prefix: str | None = None) -> pd.DataFrame:
    """Encode a periodic integer as a point on a circle.

    Hour-of-day as a raw integer tells a model that 23:00 and 00:00 are 23 units
    apart. Projecting onto (sin, cos) makes them adjacent, which is what the
    physical process actually looks like.
    """
    name = prefix or col
    theta = 2.0 * np.pi * df[col].astype(float) / period
    df[f"{name}_sin"] = np.sin(theta)
    df[f"{name}_cos"] = np.cos(theta)
    return df


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in kilometres between coordinate arrays."""
    lat1, lon1, lat2, lon2 = map(lambda a: np.radians(np.asarray(a, dtype=float)),
                                 (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial compass bearing -- lets a model learn direction-specific traffic."""
    lat1, lon1, lat2, lon2 = map(lambda a: np.radians(np.asarray(a, dtype=float)),
                                 (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


# --------------------------------------------------------------------------- #
# fitted transformers (must live inside a Pipeline)
# --------------------------------------------------------------------------- #
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its relative frequency in the training fold.

    A leakage-safe alternative to one-hot encoding for high-cardinality columns
    such as the 265 TLC zone IDs. Categories unseen during fit map to
    ``unseen_value`` (0 by default), which is the honest encoding of "this is
    rarer than anything I was trained on".
    """

    def __init__(self, unseen_value: float = 0.0, normalise: bool = True) -> None:
        self.unseen_value = unseen_value
        self.normalise = normalise

    def fit(self, X, y=None):
        X = self._frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.mappings_: dict[str, dict] = {}
        for col in X.columns:
            counts = X[col].value_counts(dropna=True)
            if self.normalise:
                counts = counts / max(1, len(X))
            self.mappings_[col] = counts.to_dict()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = self._frame(X)
        out = pd.DataFrame(index=X.index)
        for col in self.feature_names_in_:
            out[f"{col}_freq"] = X[col].map(self.mappings_.get(col, {})).fillna(self.unseen_value)
        return out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_freq" for c in self.feature_names_in_], dtype=object)

    @staticmethod
    def _frame(X) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Fold categories below a training-set frequency threshold into ``__rare__``.

    Keeps one-hot dimensionality bounded and prevents a level seen three times in
    training from becoming a memorisation hook.
    """

    def __init__(self, min_frequency: float = 0.01, other_label: str = "__rare__") -> None:
        self.min_frequency = min_frequency
        self.other_label = other_label

    def fit(self, X, y=None):
        X = FrequencyEncoder._frame(X)
        # Names are coerced to str and kept positionally. When this transformer
        # sits after a SimpleImputer that emits a bare numpy array, the incoming
        # "column names" are integers; passing those on unchanged makes the next
        # OneHotEncoder build a name as `0 + "_" + "Private"` and raise. Storing
        # strings and re-labelling on the way out keeps the chain total whatever
        # the upstream step decides to emit.
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.keep_: dict[str, set] = {}
        for col in X.columns:
            freq = X[col].value_counts(normalize=True, dropna=True)
            self.keep_[str(col)] = set(freq[freq >= self.min_frequency].index)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = FrequencyEncoder._frame(X).copy()
        X.columns = list(self.feature_names_in_[:X.shape[1]])
        for col in X.columns:
            keep = self.keep_.get(col, set())
            X[col] = X[col].where(X[col].isin(keep), self.other_label)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray([str(f) for f in input_features], dtype=object)
        return np.asarray(self.feature_names_in_, dtype=object)


class ClippedWinsoriser(BaseEstimator, TransformerMixin):
    """Clip numeric columns to training-fold percentiles.

    Fitting the bounds on the training fold is the whole point: computing them on
    the full dataset would let extreme test values inform the training transform.
    """

    def __init__(self, lower_pct: float = 0.001, upper_pct: float = 0.999) -> None:
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        arr = np.asarray(FrequencyEncoder._frame(X).to_numpy(), dtype=float)
        self.lower_ = np.nanquantile(arr, self.lower_pct, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper_pct, axis=0)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        arr = np.asarray(FrequencyEncoder._frame(X).to_numpy(), dtype=float)
        return np.clip(arr, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None
                          else [f"x{i}" for i in range(self.n_features_in_)], dtype=object)


def expanded_feature_names(preprocessor) -> list[str]:
    """Best-effort readable names after a ColumnTransformer expands one-hots."""
    try:
        return [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:
        n = getattr(preprocessor, "n_features_out_", None)
        return [f"f{i}" for i in range(n or 0)]
