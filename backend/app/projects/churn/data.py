"""Data preparation for Telco customer churn.

The dataset is small (7,043 rows) and clean, which makes it a good place to be
strict about the things that are easy to get wrong on small data:

* ``customerID`` is dropped. It is unique per row, so any model given it can
  memorise the training set perfectly and learn nothing.
* ``TotalCharges`` has 11 blanks, all for tenure-0 customers. They stay NaN
  through to the pipeline's imputer rather than being filled here, because a
  median computed over the whole file is a median that saw the test rows.
* ``gender``, ``SeniorCitizen`` and ``Partner`` are held out of the feature
  matrix and passed to the fairness audit instead.

The engineered features are all ratios and interactions a retention analyst would
recognise -- ``charges_per_tenure``, ``services_subscribed`` -- constructed
row-wise so they carry no cross-row information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ...core import paths

TARGET = "churned"
ID_COLUMN = "customerID"

# Excluded from features, retained for the fairness audit.
SENSITIVE = ["gender", "SeniorCitizen", "Partner"]

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "charges_per_tenure", "services_subscribed", "tenure_bucket_idx",
    "monthly_vs_avg_ratio", "has_dependents_and_partner",
]
CATEGORICAL_FEATURES = [
    "Contract", "PaymentMethod", "PaperlessBilling", "InternetService",
    "OnlineSecurity", "TechSupport", "StreamingTV", "StreamingMovies",
    "DeviceProtection", "OnlineBackup", "MultipleLines", "PhoneService", "Dependents",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Retention economics. These are the numbers that turn a probability into a
# decision, and they are assumptions -- stated here so they can be argued with.
ECONOMICS = {
    "monthly_margin_pct": 0.30,
    "retention_offer_cost": 45.0,
    "offer_acceptance_rate": 0.35,
    "expected_remaining_months": 18,
    "source": ("Illustrative but conventional telco assumptions: 30% contribution "
               "margin, a $45 retention incentive accepted by roughly a third of "
               "those offered, and an 18-month remaining-life horizon."),
}


def load_raw() -> pd.DataFrame:
    return pd.read_csv(paths.curated("telco_churn"))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature construction; nothing fitted, so split-safe."""
    out = df.copy()
    out[TARGET] = (out["Churn"].astype(str).str.strip() == "Yes").astype(int)

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")
    out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")

    # Average realised monthly spend. Diverges from MonthlyCharges when a customer
    # has changed plan, which is itself a churn signal.
    out["charges_per_tenure"] = np.where(
        out["tenure"] > 0, out["TotalCharges"] / out["tenure"], out["MonthlyCharges"]
    )
    out["monthly_vs_avg_ratio"] = np.where(
        out["charges_per_tenure"] > 0,
        out["MonthlyCharges"] / out["charges_per_tenure"], 1.0,
    )

    subscribed = pd.Series(0, index=out.index)
    for col in SERVICE_COLUMNS:
        if col in out.columns:
            vals = out[col].astype(str).str.strip()
            subscribed += (~vals.isin(["No", "No phone service",
                                       "No internet service", "nan"])).astype(int)
    out["services_subscribed"] = subscribed

    # Tenure bands mirror how retention teams actually segment a book.
    bins = [-0.1, 6, 12, 24, 48, 1e9]
    out["tenure_bucket"] = pd.cut(
        out["tenure"], bins=bins,
        labels=["0-6m", "6-12m", "12-24m", "24-48m", "48m+"],
    )
    out["tenure_bucket_idx"] = out["tenure_bucket"].cat.codes.astype(float)

    out["has_dependents_and_partner"] = (
        (out["Dependents"].astype(str).str.strip() == "Yes")
        & (out["Partner"].astype(str).str.strip() == "Yes")
    ).astype(int)

    out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna("No")
    return out


def prepare() -> pd.DataFrame:
    return engineer(load_raw())


def build_preprocessor() -> ColumnTransformer:
    """Median imputation and one-hot encoding, both fitted per fold."""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
            ]), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                         sparse_output=False)),
            ]), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def expected_value(prob: float, monthly_charges: float,
                   econ: dict | None = None) -> dict:
    """Translate a churn probability into the decision an operator actually makes.

    The comparison is between doing nothing (and losing the margin with
    probability p) and making an offer (paying its cost now, saving the margin
    with probability p x acceptance). Below the break-even probability the offer
    destroys value -- which is why a fixed 0.5 threshold is almost always wrong.
    """
    e = {**ECONOMICS, **(econ or {})}
    clv = monthly_charges * e["monthly_margin_pct"] * e["expected_remaining_months"]
    loss_if_nothing = prob * clv
    saved = prob * e["offer_acceptance_rate"] * clv
    value_of_offer = saved - e["retention_offer_cost"]
    denom = clv * e["offer_acceptance_rate"]
    breakeven = e["retention_offer_cost"] / denom if denom > 0 else 1.0
    return {
        "churn_probability": round(float(prob), 4),
        "customer_lifetime_margin": round(float(clv), 2),
        "expected_loss_if_no_action": round(float(loss_if_nothing), 2),
        "expected_value_of_offer": round(float(value_of_offer), 2),
        "breakeven_probability": round(float(min(1.0, breakeven)), 4),
        "recommended_action": ("send retention offer" if value_of_offer > 0
                               else "no action -- offer costs more than it saves"),
        "assumptions": e,
    }


def build_inference_row(payload: dict) -> tuple[pd.DataFrame, dict]:
    """Build one model-ready row from an API request, filling sane defaults."""
    defaults = {
        "tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": None,
        "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "DeviceProtection": "No", "OnlineBackup": "No",
        "MultipleLines": "No", "PhoneService": "Yes", "Dependents": "No",
        "Partner": "No", "gender": "Female", "SeniorCitizen": 0,
    }
    raw = {**defaults, **{k: v for k, v in (payload or {}).items() if v is not None}}
    raw["Churn"] = "No"  # placeholder; engineer() needs the column, it is not a feature
    if raw.get("TotalCharges") is None:
        raw["TotalCharges"] = float(raw["MonthlyCharges"]) * max(1, float(raw["tenure"]))

    frame = engineer(pd.DataFrame([raw]))
    context = {
        "tenure": int(raw["tenure"]),
        "monthly_charges": float(raw["MonthlyCharges"]),
        "contract": str(raw["Contract"]),
        "payment_method": str(raw["PaymentMethod"]),
        "internet": str(raw["InternetService"]),
        "services_subscribed": int(frame["services_subscribed"].iat[0]),
        "tenure_bucket": str(frame["tenure_bucket"].iat[0]),
    }
    return frame[ALL_FEATURES], context


def personas() -> list[dict]:
    """Preset customers spanning the risk range, for the demo UI."""
    return [
        {"label": "New fibre customer, month-to-month",
         "note": "The archetypal high-risk profile: short tenure, no contract, "
                 "premium price, electronic-check billing.",
         "payload": {"tenure": 2, "MonthlyCharges": 89.9, "Contract": "Month-to-month",
                     "PaymentMethod": "Electronic check", "InternetService": "Fiber optic",
                     "OnlineSecurity": "No", "TechSupport": "No"}},
        {"label": "Long-tenure two-year contract",
         "note": "Locked in and well-served; the model should be confident this "
                 "customer stays.",
         "payload": {"tenure": 62, "MonthlyCharges": 105.5, "Contract": "Two year",
                     "PaymentMethod": "Bank transfer (automatic)",
                     "InternetService": "Fiber optic", "OnlineSecurity": "Yes",
                     "TechSupport": "Yes", "StreamingTV": "Yes", "StreamingMovies": "Yes"}},
        {"label": "Mid-tenure DSL, one-year contract",
         "note": "The ambiguous middle, where the threshold choice actually decides "
                 "the outcome.",
         "payload": {"tenure": 24, "MonthlyCharges": 56.0, "Contract": "One year",
                     "PaymentMethod": "Credit card (automatic)", "InternetService": "DSL",
                     "OnlineSecurity": "Yes", "TechSupport": "No"}},
        {"label": "Phone-only, minimal spend",
         "note": "Low monthly margin: even a modest retention offer may not pay for "
                 "itself here.",
         "payload": {"tenure": 8, "MonthlyCharges": 20.05, "Contract": "Month-to-month",
                     "PaymentMethod": "Mailed check", "InternetService": "No",
                     "PhoneService": "Yes"}},
    ]
