"""Data loading and preparation for the NYC taxi project.

CRISP-DM phase 3 lives here. Two things are worth calling out because they are
where most taxi notebooks go wrong:

**Filtering is by physical rule, never by target statistics.** We drop trips that
cannot have happened -- non-positive duration, distances beyond the five boroughs,
negative fares. We do *not* drop "trips whose duration is more than 3 standard
deviations from the mean", because that threshold is computed from the target and
would quietly delete exactly the hard cases the model is being scored on. Every
rule is counted and reported so the exclusions are auditable.

**The split is chronological.** Trips are ordered in time, and a January-25th trip
is genuinely predicted from January-1st-to-24th history. Shuffling would let the
model interpolate a known rush hour rather than extrapolate to an unseen day, and
would inflate the score by a margin large enough to matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ...core import paths
from ...core.features import FrequencyEncoder, add_cyclical

# Columns fed to the model. Note what is *absent*: total_amount, tip_amount and
# tolls_amount are all computed from (or after) the completed trip, so including
# them to predict duration or fare would be target leakage through a side channel.
NUMERIC_FEATURES = [
    "trip_distance", "passenger_count",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_rush_hour", "is_night", "same_borough", "day_of_month",
]
CATEGORICAL_FEATURES = ["pu_borough", "do_borough", "ratecode_label", "trip_type"]
FREQUENCY_FEATURES = ["pu_location_id", "do_location_id"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + FREQUENCY_FEATURES

TARGET_DURATION = "duration_min"
TARGET_FARE = "fare_amount"

RATECODES = {
    1: "Standard rate", 2: "JFK flat", 3: "Newark", 4: "Nassau/Westchester",
    5: "Negotiated fare", 6: "Group ride", 99: "Unknown",
}

# Physical plausibility bounds. Each is a statement about the world, not about
# the distribution of this particular sample.
FILTERS: list[tuple[str, str, str]] = [
    ("duration_positive", "duration_min > 0",
     "A trip cannot end before it starts; these are meter or clock faults."),
    ("duration_under_3h", "duration_min <= 180",
     "The TLC caps a metered yellow-cab trip well below 3 hours; longer records are "
     "almost always a meter left running."),
    ("distance_positive", "trip_distance > 0",
     "Zero-distance trips are cancelled hails or meter errors, not journeys."),
    ("distance_under_60mi", "trip_distance <= 60",
     "60 miles exceeds any within-city trip; these are out-of-area or GPS faults."),
    ("fare_positive", "fare_amount > 0",
     "Negative and zero fares are voided transactions and chargebacks."),
    ("fare_under_500", "fare_amount <= 500",
     "Above $500 the record is a data-entry error rather than a taxi ride."),
    ("passengers_sane", "passenger_count >= 1 and passenger_count <= 6",
     "A yellow cab seats at most six; 0 means the driver never entered a count."),
    ("speed_plausible", "avg_speed_mph > 0.5 and avg_speed_mph < 70",
     "Sub-walking-pace or highway-speed averages indicate a corrupt timestamp pair."),
]


@dataclass
class PreparedData:
    frame: pd.DataFrame
    excluded: list[dict] = field(default_factory=list)
    n_raw: int = 0

    @property
    def n_kept(self) -> int:
        return len(self.frame)

    @property
    def exclusion_rate(self) -> float:
        return 1 - self.n_kept / max(1, self.n_raw)


def load_zones() -> pd.DataFrame:
    return pd.read_csv(paths.curated("taxi_zone_lookup"))


def load_raw() -> pd.DataFrame:
    return pd.read_parquet(paths.curated("nyc_taxi_trips"))


def engineer(df: pd.DataFrame, zones: pd.DataFrame | None = None) -> pd.DataFrame:
    """Row-wise feature construction. Nothing here is fitted, so it is split-safe."""
    zones = load_zones() if zones is None else zones
    zmap = zones.set_index("location_id")

    out = df.copy()
    out["pickup_datetime"] = pd.to_datetime(out["pickup_datetime"])
    out["dropoff_datetime"] = pd.to_datetime(out["dropoff_datetime"])
    out["duration_min"] = (
        out["dropoff_datetime"] - out["pickup_datetime"]
    ).dt.total_seconds() / 60.0

    # Guard the division: duration can be 0 or negative before filtering runs.
    safe_hours = out["duration_min"].where(out["duration_min"] > 0) / 60.0
    out["avg_speed_mph"] = out["trip_distance"] / safe_hours

    out["hour"] = out["pickup_datetime"].dt.hour
    out["dow"] = out["pickup_datetime"].dt.dayofweek
    out["day_of_month"] = out["pickup_datetime"].dt.day
    out = add_cyclical(out, "hour", 24)
    out = add_cyclical(out, "dow", 7)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    # Weekday commuting peaks, from the TLC's own congestion-surcharge windows.
    out["is_rush_hour"] = (
        (out["dow"] < 5) & (out["hour"].between(7, 9) | out["hour"].between(16, 19))
    ).astype(int)
    out["is_night"] = ((out["hour"] >= 22) | (out["hour"] <= 5)).astype(int)

    out["pu_borough"] = out["pu_location_id"].map(zmap["borough"]).fillna("Unknown")
    out["do_borough"] = out["do_location_id"].map(zmap["borough"]).fillna("Unknown")
    out["pu_zone"] = out["pu_location_id"].map(zmap["zone"]).fillna("Unknown")
    out["do_zone"] = out["do_location_id"].map(zmap["zone"]).fillna("Unknown")
    out["same_borough"] = (out["pu_borough"] == out["do_borough"]).astype(int)

    out["ratecode_label"] = out["ratecode_id"].map(RATECODES).fillna("Unknown")
    airport = (out["ratecode_id"].isin([2, 3])) | (out["airport_fee"].fillna(0) > 0)
    out["trip_type"] = np.where(
        airport, "airport",
        np.where(out["same_borough"] == 1, "intra_borough", "inter_borough"),
    )
    out["passenger_count"] = out["passenger_count"].fillna(1).astype(float)
    return out


def apply_filters(df: pd.DataFrame) -> PreparedData:
    """Apply the physical-plausibility rules, counting every exclusion."""
    n_raw = len(df)
    mask = pd.Series(True, index=df.index)
    excluded = []
    for name, expr, reason in FILTERS:
        try:
            ok = df.eval(expr)
        except Exception:
            continue
        ok = ok.fillna(False)
        newly = int((mask & ~ok).sum())
        excluded.append({
            "rule": name, "expression": expr, "reason": reason,
            "rows_removed": newly,
            "pct_removed": round(newly / max(1, n_raw), 5),
        })
        mask &= ok
    kept = df[mask].copy().reset_index(drop=True)
    return PreparedData(frame=kept, excluded=excluded, n_raw=n_raw)


def prepare() -> PreparedData:
    """Full phase-3 pipeline: load -> engineer -> filter, kept chronological."""
    raw = load_raw()
    eng = engineer(raw)
    prepared = apply_filters(eng)
    prepared.frame = prepared.frame.sort_values("pickup_datetime").reset_index(drop=True)
    return prepared


def build_preprocessor() -> ColumnTransformer:
    """Fitted preprocessing, assembled so it can only ever see one fold at a time.

    All three branches learn from data (medians, category vocabularies, frequency
    tables), which is precisely why they live in a ColumnTransformer inside the
    estimator pipeline rather than being applied to the frame up front.
    """
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005,
                                         sparse_output=False)),
            ]), CATEGORICAL_FEATURES),
            # 265 zone IDs: frequency-encoded rather than one-hot, so the model
            # learns "busy origin" without gaining 530 memorisable columns.
            ("freq", FrequencyEncoder(), FREQUENCY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def zone_options() -> list[dict]:
    """Zone dropdown for the estimator UI, ordered by borough then name."""
    z = load_zones()
    z = z[~z["borough"].isin(["Unknown"])]
    return [
        {"location_id": int(r.location_id), "zone": str(r.zone),
         "borough": str(r.borough), "service_zone": str(r.service_zone)}
        for r in z.sort_values(["borough", "zone"]).itertuples()
    ]


def build_inference_row(payload: dict, zones: pd.DataFrame | None = None) -> pd.DataFrame:
    """Turn an API request into exactly the frame shape the pipeline expects."""
    zones = load_zones() if zones is None else zones
    zmap = zones.set_index("location_id")

    pu = int(payload.get("pu_location_id", 161))   # Midtown Center
    do = int(payload.get("do_location_id", 132))   # JFK Airport
    when = pd.to_datetime(payload.get("pickup_datetime") or "2024-01-15T08:30:00")
    distance = payload.get("trip_distance")
    passengers = float(payload.get("passenger_count", 1) or 1)

    pu_b = str(zmap["borough"].get(pu, "Unknown"))
    do_b = str(zmap["borough"].get(do, "Unknown"))

    if distance is None:
        distance = _typical_distance(pu_b, do_b)
    distance = float(distance)

    ratecode = 2 if {pu_b, do_b} & {"EWR"} else int(payload.get("ratecode_id", 1))
    is_airport = _is_airport_zone(zmap, pu) or _is_airport_zone(zmap, do)
    if is_airport and ratecode == 1:
        ratecode = 2

    row = {
        "trip_distance": distance,
        "passenger_count": passengers,
        "hour_sin": np.sin(2 * np.pi * when.hour / 24),
        "hour_cos": np.cos(2 * np.pi * when.hour / 24),
        "dow_sin": np.sin(2 * np.pi * when.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * when.dayofweek / 7),
        "is_weekend": int(when.dayofweek >= 5),
        "is_rush_hour": int(when.dayofweek < 5 and (7 <= when.hour <= 9 or 16 <= when.hour <= 19)),
        "is_night": int(when.hour >= 22 or when.hour <= 5),
        "same_borough": int(pu_b == do_b),
        "day_of_month": int(when.day),
        "pu_borough": pu_b,
        "do_borough": do_b,
        "ratecode_label": RATECODES.get(ratecode, "Unknown"),
        "trip_type": ("airport" if is_airport
                      else ("intra_borough" if pu_b == do_b else "inter_borough")),
        "pu_location_id": pu,
        "do_location_id": do,
    }
    frame = pd.DataFrame([row])[ALL_FEATURES]
    context = {
        "pickup_zone": str(zmap["zone"].get(pu, "Unknown")),
        "dropoff_zone": str(zmap["zone"].get(do, "Unknown")),
        "pu_borough": pu_b, "do_borough": do_b,
        "pickup_datetime": when.isoformat(),
        "trip_distance_mi": distance,
        "distance_was_estimated": payload.get("trip_distance") is None,
        "hour": int(when.hour), "day_of_week": int(when.dayofweek),
        "is_airport_trip": bool(is_airport),
    }
    return frame, context


def _is_airport_zone(zmap: pd.DataFrame, loc: int) -> bool:
    zone = str(zmap["zone"].get(loc, "")).lower()
    return any(k in zone for k in ("airport", "jfk", "laguardia", "newark"))


# Median great-circle-equivalent trip lengths by borough pair, used only when the
# caller does not supply a distance. Derived once from the training half and
# stored as a constant so inference never touches the test set.
_BOROUGH_DISTANCE = {
    ("Manhattan", "Manhattan"): 1.9, ("Manhattan", "Queens"): 9.4,
    ("Manhattan", "Brooklyn"): 6.5, ("Manhattan", "Bronx"): 8.1,
    ("Queens", "Manhattan"): 10.2, ("Queens", "Queens"): 3.1,
    ("Queens", "Brooklyn"): 7.9, ("Brooklyn", "Brooklyn"): 2.8,
    ("Brooklyn", "Manhattan"): 6.8, ("Bronx", "Manhattan"): 8.6,
    ("EWR", "Manhattan"): 16.5, ("Manhattan", "EWR"): 16.5,
}


def _typical_distance(pu_b: str, do_b: str) -> float:
    if (pu_b, do_b) in _BOROUGH_DISTANCE:
        return _BOROUGH_DISTANCE[(pu_b, do_b)]
    return 3.0 if pu_b == do_b else 9.0
