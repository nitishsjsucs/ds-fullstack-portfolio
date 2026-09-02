"""Feature preparation for network-intrusion anomaly detection.

The defining constraint of this project: **the label is never used to fit
anything.** Every detector here is unsupervised. `is_attack` exists only so we can
measure, after the fact, whether the scores an unsupervised method produced happen
to rank intrusions highly.

That constraint is what makes the project honest, and it is easy to violate by
accident. Two guards:

* The label column is dropped from the feature frame at construction, not at fit
  time, so there is no code path where a detector can see it.
* The contamination parameter -- which several scikit-learn detectors accept, and
  which is really "what fraction of the data is anomalous?" -- is set from a
  *stated operational assumption*, never from the observed attack rate. Passing
  the true rate would be feeding the answer to the model through the back door.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from ...core import paths

LABEL = "is_attack"
LABEL_TEXT = "labels"

# What an analyst team would assume before seeing any labels: roughly 2% of
# connections merit review. The true rate in this sample is 3.37%; we deliberately
# do NOT use that number, because in deployment nobody knows it.
ASSUMED_CONTAMINATION = 0.02

CATEGORICAL = ["protocol_type", "service", "flag"]

NUMERIC = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "num_root", "num_file_creations", "num_access_files", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

ENGINEERED = ["log_src_bytes", "log_dst_bytes", "bytes_ratio", "is_zero_payload"]


def load_raw() -> pd.DataFrame:
    return pd.read_parquet(paths.curated("kdd99_intrusion"))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features only. Byte counts span nine orders of magnitude, so the
    logs are what any distance- or density-based detector actually needs."""
    out = df.copy()
    for col in NUMERIC:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in CATEGORICAL:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    out["log_src_bytes"] = np.log1p(out["src_bytes"].clip(lower=0))
    out["log_dst_bytes"] = np.log1p(out["dst_bytes"].clip(lower=0))
    out["bytes_ratio"] = out["src_bytes"] / (out["dst_bytes"] + 1.0)
    out["is_zero_payload"] = ((out["src_bytes"] == 0) & (out["dst_bytes"] == 0)).astype(int)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in NUMERIC + ENGINEERED if c in df.columns]
    cols += [c for c in CATEGORICAL if c in df.columns]
    return cols


def prepare() -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Return (features, binary label, attack-family label, metadata).

    The label is split out here and never rejoined. Callers get X without it.
    """
    raw = load_raw()
    eng = engineer(raw)
    y = eng[LABEL].astype(int)
    family = eng[LABEL_TEXT].astype(str)
    cols = feature_columns(eng)
    X = eng[cols].copy()

    assert LABEL not in X.columns and LABEL_TEXT not in X.columns, (
        "Label leaked into the feature matrix"
    )

    meta = {
        "rows": int(len(X)),
        "features": len(cols),
        "numeric": len([c for c in cols if c not in CATEGORICAL]),
        "categorical": len([c for c in cols if c in CATEGORICAL]),
        "true_attack_rate": float(y.mean()),
        "assumed_contamination": ASSUMED_CONTAMINATION,
        "label_policy": (
            "Detectors are fitted on X alone. The label is used only to score the "
            "resulting rankings. Contamination is set from a stated operational "
            f"assumption ({ASSUMED_CONTAMINATION:.0%}), not from the observed rate "
            f"({y.mean():.2%}) -- in deployment that rate is unknown, and supplying it "
            "would be leaking the answer through a hyperparameter."
        ),
        "attack_families": family[y == 1].value_counts().to_dict(),
    }
    return X, y, family, meta


def build_preprocessor() -> ColumnTransformer:
    """RobustScaler for the numerics; one-hot for the three categoricals.

    RobustScaler rather than StandardScaler because the mean and standard
    deviation of ``src_bytes`` are themselves dominated by the outliers we are
    trying to find -- standardising by them would shrink exactly the signal that
    matters. Median and IQR are unmoved by a handful of extreme connections.
    """
    return ColumnTransformer(
        transformers=[
            ("num", RobustScaler(quantile_range=(5, 95)),
             [c for c in NUMERIC + ENGINEERED]),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005,
                                  sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def alert_budget_simulation(scores: np.ndarray, y_true: np.ndarray,
                            budgets: tuple[int, ...] = (25, 50, 100, 200, 400, 800, 1600)) -> list[dict]:
    """What an analyst team actually experiences: a fixed-size queue.

    Nobody reviews "everything above 0.5". They review the top N alerts a shift can
    handle. Precision@k and recall@k are therefore the operational metrics, and
    they behave very differently from PR-AUC when the queue is small.
    """
    order = np.argsort(-np.asarray(scores))
    y_true = np.asarray(y_true)
    total_attacks = max(1, int(y_true.sum()))
    out = []
    for k in budgets:
        k = min(k, len(order))
        top = order[:k]
        caught = int(y_true[top].sum())
        out.append({
            "alert_budget": int(k),
            "true_positives": caught,
            "false_positives": int(k - caught),
            "precision_at_k": round(caught / k, 4),
            "recall_at_k": round(caught / total_attacks, 4),
            "missed_attacks": int(total_attacks - caught),
            "analyst_hours": round(k * 6 / 60, 1),  # ~6 minutes to triage one alert
        })
    return out
