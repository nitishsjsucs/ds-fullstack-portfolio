"""NYC Taxi Fare & Duration -- project package.

Exposes the three things the platform expects from every project:
``META`` (registry entry), ``predict`` (live inference) and ``router``
(project-specific endpoints).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ...core.artifacts import ArtifactStore
from ...core.explain import local_explanation
from ...registry import get as _get
from . import data as D
from .train import run as train

META = _get("taxi")
SLUG = "taxi"
_store = ArtifactStore(SLUG)

router = APIRouter()


def predict(payload: dict) -> dict:
    """Estimate duration, fare and an 80% band for one prospective trip."""
    _store.require()
    model = _store.model("model")
    frame, context = D.build_inference_row(payload)

    duration = float(model.predict(frame)[0])

    lo = hi = None
    if _store.has_model("q_low") and _store.has_model("q_high"):
        a = float(_store.model("q_low").predict(frame)[0])
        b = float(_store.model("q_high").predict(frame)[0])
        lo, hi = min(a, b), max(a, b)   # repair any quantile crossing

    fare = None
    if _store.has_model("fare_model"):
        fare = float(_store.model("fare_model").predict(frame)[0])

    # Local attribution, computed in the model's own transformed feature space.
    attribution = None
    try:
        prep = model.named_steps["prep"]
        names = list(prep.get_feature_names_out())
        row_t = pd.DataFrame(prep.transform(frame), columns=names)
        attribution = local_explanation(model, row_t, row_t)
    except Exception:
        attribution = None

    duration = max(1.0, duration)
    speed = context["trip_distance_mi"] / (duration / 60) if duration > 0 else None

    return {
        "duration_minutes": round(duration, 2),
        "duration_interval_80": (
            {"lower": round(max(1.0, lo), 2), "upper": round(hi, 2)}
            if lo is not None else None
        ),
        "fare_usd": round(fare, 2) if fare is not None else None,
        "implied_speed_mph": round(speed, 1) if speed else None,
        "context": context,
        "attribution": attribution,
        "disclaimer": (
            "Trained on January 2024 yellow-cab records only. Excludes tips, tolls "
            "and surcharges, and has no access to live traffic."
        ),
    }


@router.get("/zones")
def zones() -> dict:
    """The 265-zone dictionary that powers the estimator's dropdowns."""
    opts = D.zone_options()
    boroughs: dict[str, list] = {}
    for z in opts:
        boroughs.setdefault(z["borough"], []).append(z)
    return {"count": len(opts), "zones": opts,
            "by_borough": {k: v for k, v in sorted(boroughs.items())}}


@router.post("/batch-predict")
def batch_predict(body: dict) -> dict:
    """Score up to 200 trips at once -- used by the throughput panel."""
    trips = body.get("trips") or []
    if not isinstance(trips, list) or not trips:
        raise HTTPException(422, "Body must contain a non-empty 'trips' array.")
    if len(trips) > 200:
        raise HTTPException(422, f"At most 200 trips per call (received {len(trips)}).")
    _store.require()
    model = _store.model("model")

    frames, contexts = [], []
    for t in trips:
        f, c = D.build_inference_row(t)
        frames.append(f)
        contexts.append(c)
    batch = pd.concat(frames, ignore_index=True)
    preds = model.predict(batch)
    fares = (_store.model("fare_model").predict(batch)
             if _store.has_model("fare_model") else np.full(len(batch), np.nan))
    return {
        "count": len(trips),
        "results": [
            {"duration_minutes": round(float(p), 2),
             "fare_usd": (None if not np.isfinite(f) else round(float(f), 2)),
             "context": c}
            for p, f, c in zip(preds, fares, contexts)
        ],
    }


@router.get("/sensitivity")
def sensitivity(pu_location_id: int = 161, do_location_id: int = 132,
                passenger_count: int = 1) -> dict:
    """Sweep departure hour across a full day for one origin/destination pair.

    This is the view that makes the learned rush-hour structure legible: the same
    trip, priced at every hour, with the 80% band drawn around it.
    """
    _store.require()
    model = _store.model("model")
    has_q = _store.has_model("q_low") and _store.has_model("q_high")

    rows, contexts = [], []
    for hour in range(24):
        f, c = D.build_inference_row({
            "pu_location_id": pu_location_id, "do_location_id": do_location_id,
            "passenger_count": passenger_count,
            "pickup_datetime": f"2024-01-24T{hour:02d}:30:00",
        })
        rows.append(f)
        contexts.append(c)
    batch = pd.concat(rows, ignore_index=True)
    point = model.predict(batch)
    lo = _store.model("q_low").predict(batch) if has_q else None
    hi = _store.model("q_high").predict(batch) if has_q else None

    curve = []
    for i, hour in enumerate(range(24)):
        entry = {"hour": hour, "duration_minutes": round(float(point[i]), 2)}
        if has_q:
            a, b = float(lo[i]), float(hi[i])
            entry["lower"] = round(min(a, b), 2)
            entry["upper"] = round(max(a, b), 2)
        curve.append(entry)

    peak = max(curve, key=lambda r: r["duration_minutes"])
    trough = min(curve, key=lambda r: r["duration_minutes"])
    return {
        "route": {"from": contexts[0]["pickup_zone"], "to": contexts[0]["dropoff_zone"],
                  "distance_mi": contexts[0]["trip_distance_mi"]},
        "curve": curve,
        "peak": peak,
        "trough": trough,
        "peak_penalty_minutes": round(peak["duration_minutes"] - trough["duration_minutes"], 2),
        "note": ("Same trip, every departure hour, on a Wednesday. The gap between "
                 "peak and trough is the congestion cost the model has learned."),
    }


__all__ = ["META", "SLUG", "predict", "router", "train", "D"]
