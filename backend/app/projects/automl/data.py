"""Data preparation for the Census Income AutoML tournament.

Two columns need explaining because they are where this dataset traps people.

**``fnlwgt`` is dropped.** It is the census sampling weight -- how many people in
the population this row represents. It is an artefact of survey design, carries no
information about the individual, and a boosted tree will happily find spurious
structure in it because it correlates with the geographic strata the survey used.
Including it is a subtle form of leakage from the sampling frame.

**``education`` and ``education_num`` are the same variable.** One is the text
label, the other its ordinal code, in perfect one-to-one correspondence. Keeping
both gives the model two copies of one signal, which distorts every importance
measure without adding information. We keep the ordinal.

``sex`` and ``race`` are excluded from the feature matrix and reserved for the
fairness audit. As always, exclusion does not remove the bias -- occupation,
relationship and hours-per-week are all correlated proxies -- which is exactly why
the disparity has to be measured rather than assumed away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ...core import paths
from ...core.features import RareCategoryGrouper

TARGET = "high_income"
SENSITIVE = ["sex", "race", "age_band"]

DROPPED = {
    "fnlwgt": ("Census sampling weight -- a survey-design artefact describing how many "
               "people the row represents, not a property of the person. Correlates "
               "with geographic strata, so a tree finds spurious structure in it."),
    "education": ("Perfectly redundant with education_num, which is the same variable "
                  "as an ordinal code. Keeping both splits one signal across two "
                  "columns and corrupts every importance measure."),
    "split_origin": "Bookkeeping column recording which archive file the row came from.",
    "sex": "Reserved for the fairness audit; excluded from features.",
    "race": "Reserved for the fairness audit; excluded from features.",
}

NUMERIC_FEATURES = [
    "age", "education_num", "capital_gain", "capital_loss", "hours_per_week",
    "capital_net", "has_capital_gain", "hours_band",
]
CATEGORICAL_FEATURES = [
    "workclass", "marital_status", "occupation", "relationship", "native_country",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw() -> pd.DataFrame:
    return pd.read_csv(paths.curated("adult_census_income"))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TARGET] = (out["income"].astype(str).str.strip() == ">50K").astype(int)

    # Capital gain and loss are both overwhelmingly zero with a long tail; the net
    # and a simple indicator are far more usable than the raw amounts alone.
    out["capital_net"] = out["capital_gain"].fillna(0) - out["capital_loss"].fillna(0)
    out["has_capital_gain"] = (out["capital_gain"].fillna(0) > 0).astype(int)
    # Hours banded the way labour statistics conventionally report them.
    out["hours_band"] = pd.cut(
        out["hours_per_week"], bins=[-1, 20, 35, 45, 60, 200], labels=[0, 1, 2, 3, 4]
    ).astype(float)

    out["age_band"] = pd.cut(
        out["age"], bins=[0, 25, 35, 45, 55, 65, 200],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
    ).astype(str)

    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            # np.nan, not pd.NA: scikit-learn's imputer tests missingness with
            # `X != X` on object arrays, and pd.NA raises rather than returning
            # False there. The two are interchangeable for our purposes; only one
            # survives the round trip into sklearn.
            out[col] = out[col].astype(str).replace({"nan": np.nan, "<NA>": np.nan,
                                                     "None": np.nan})
    return out


def prepare() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    raw = load_raw()
    eng = engineer(raw)
    X = eng[ALL_FEATURES].copy()
    y = eng[TARGET]
    sensitive = eng[SENSITIVE].astype(str)

    for col in DROPPED:
        assert col not in X.columns, f"{col} must not be a feature"

    meta = {
        "rows": int(len(X)),
        "features": len(ALL_FEATURES),
        "positive_rate": float(y.mean()),
        "dropped_columns": DROPPED,
        "missing_cells": int(X.isna().sum().sum()),
        "missing_by_column": {c: int(v) for c, v in X.isna().sum().items() if v > 0},
        "sensitive_attributes": SENSITIVE,
        "sensitive_policy": (
            "sex, race and age band are excluded from the feature matrix and used only "
            "to measure disparity. Exclusion does not make the model fair -- "
            "occupation, relationship and hours are all correlated proxies -- which is "
            "precisely why the audit exists."
        ),
    }
    return X, y, sensitive, meta


def build_preprocessor() -> ColumnTransformer:
    """Median/mode imputation, rare-level grouping, one-hot -- all fitted per fold."""
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            # set_output("pandas") makes the imputer emit a DataFrame with the real
            # column names, so the rare-level grouper and the encoder downstream see
            # 'occupation' rather than the integer 2. Without it the one-hot feature
            # names come out as '2_Sales', which makes every importance table and the
            # distilled tree's printed rules unreadable.
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                # native_country has 42 levels, most with a handful of rows. Grouping
                # them prevents a level seen 3 times becoming a memorisation hook.
                ("rare", RareCategoryGrouper(min_frequency=0.01)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]).set_output(transform="pandas"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_inference_row(payload: dict) -> tuple[pd.DataFrame, dict]:
    defaults = {
        "age": 39, "education_num": 10, "capital_gain": 0, "capital_loss": 0,
        "hours_per_week": 40, "workclass": "Private", "marital_status": "Never-married",
        "occupation": "Prof-specialty", "relationship": "Not-in-family",
        "native_country": "United-States", "income": "<=50K", "sex": "Male",
        "race": "White", "fnlwgt": 0, "education": "Bachelors",
    }
    raw = {**defaults, **{k: v for k, v in (payload or {}).items() if v is not None}}
    frame = engineer(pd.DataFrame([raw]))
    context = {
        "age": int(raw["age"]),
        "education_num": int(raw["education_num"]),
        "hours_per_week": int(raw["hours_per_week"]),
        "occupation": str(raw["occupation"]),
        "marital_status": str(raw["marital_status"]),
        "capital_net": float(frame["capital_net"].iat[0]),
    }
    return frame[ALL_FEATURES], context


def profiles() -> list[dict]:
    return [
        {"label": "Senior manager, married, 55h/week",
         "note": "The profile the model is most confident about at the high end.",
         "payload": {"age": 47, "education_num": 14, "hours_per_week": 55,
                     "workclass": "Private", "marital_status": "Married-civ-spouse",
                     "occupation": "Exec-managerial", "relationship": "Husband",
                     "capital_gain": 0}},
        {"label": "Early-career service worker",
         "note": "Young, part-time, no capital income -- confidently below the threshold.",
         "payload": {"age": 22, "education_num": 9, "hours_per_week": 25,
                     "workclass": "Private", "marital_status": "Never-married",
                     "occupation": "Other-service", "relationship": "Own-child"}},
        {"label": "Mid-career professional, single",
         "note": "The genuinely ambiguous middle, where the threshold decides the answer.",
         "payload": {"age": 38, "education_num": 13, "hours_per_week": 45,
                     "workclass": "Private", "marital_status": "Never-married",
                     "occupation": "Prof-specialty", "relationship": "Not-in-family"}},
        {"label": "Self-employed with capital gains",
         "note": "Capital gains are the single most decisive feature in the model.",
         "payload": {"age": 52, "education_num": 12, "hours_per_week": 50,
                     "workclass": "Self-emp-not-inc", "marital_status": "Married-civ-spouse",
                     "occupation": "Sales", "relationship": "Husband",
                     "capital_gain": 15024}},
    ]
