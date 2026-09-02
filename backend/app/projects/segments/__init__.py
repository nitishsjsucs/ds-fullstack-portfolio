"""RFM Customer Segmentation -- project package."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ...core.artifacts import ArtifactStore
from ...registry import get as _get
from . import data as D
from .train import run as train

META = _get("segments")
SLUG = "segments"
_store = ArtifactStore(SLUG)

router = APIRouter()


def predict(payload: dict) -> dict:
    """Assign one customer to a segment, with an explicit confidence measure."""
    _store.require()
    prep = _store.model("preprocessor")
    model = _store.model("model")
    profiles = (_store.payload("extras") or {}).get("profiles", [])

    row = {
        "recency_days": float(payload.get("recency_days", 30)),
        "frequency": float(payload.get("frequency", 3)),
        "monetary": float(payload.get("monetary", 800)),
        "n_distinct_skus": float(payload.get("n_distinct_skus", 25)),
    }
    row["avg_basket_value"] = float(
        payload.get("avg_basket_value") or row["monetary"] / max(1.0, row["frequency"])
    )
    if row["monetary"] <= 0 or row["frequency"] <= 0:
        raise ValueError("monetary and frequency must both be positive.")

    frame = pd.DataFrame([row])[D.FEATURES]
    X = prep.transform(frame)

    centroids = getattr(model, "cluster_centers_", None)
    if centroids is None:                      # Gaussian mixture
        centroids = getattr(model, "means_", None)
    if centroids is None:
        raise RuntimeError("Fitted model exposes no centroids.")

    distances = np.linalg.norm(centroids - X[0], axis=1)
    order = np.argsort(distances)
    nearest, runner_up = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    d0, d1 = float(distances[nearest]), float(distances[runner_up])

    # Confidence: how much closer is the winner than the runner-up? A ratio near 1
    # means the customer sits on a boundary and should get generic treatment.
    ratio = d0 / d1 if d1 > 0 else 0.0
    confidence = "high" if ratio < 0.6 else "medium" if ratio < 0.85 else "borderline"

    profile = next((p for p in profiles if p["cluster"] == nearest), None)
    return {
        "cluster": nearest,
        "segment": profile["name"] if profile else f"Cluster {nearest}",
        "recommended_action": profile["recommended_action"] if profile else None,
        "confidence": confidence,
        "distance_ratio": round(ratio, 4),
        "distances": [{"cluster": int(c), "distance": round(float(distances[c]), 4),
                       "segment": next((p["name"] for p in profiles
                                        if p["cluster"] == int(c)), f"Cluster {c}")}
                      for c in order],
        "input": row,
        "segment_profile": profile,
        "caveat": (
            "A 'borderline' result means the customer is nearly equidistant from two "
            "centroids. Treat them with generic merchandising rather than a "
            "segment-specific campaign."
            if confidence == "borderline" else
            "Assignment is by nearest centroid in log-scaled RFM space."
        ),
    }


@router.get("/profiles")
def profiles() -> dict:
    _store.require()
    extras = _store.payload("extras") or {}
    ev = _store.payload("evaluation") or {}
    return {
        "k": ev.get("chosen_k"),
        "champion": ev.get("champion"),
        "profiles": extras.get("profiles", []),
        "population": extras.get("population", {}),
        "rationale": ev.get("k_rationale"),
    }


@router.get("/scatter")
def scatter(cluster: int | None = None) -> dict:
    """The PCA projection, optionally filtered to one segment."""
    _store.require()
    ev = _store.payload("evaluation") or {}
    points = ev.get("scatter", [])
    if cluster is not None:
        points = [p for p in points if p["cluster"] == cluster]
    return {"points": points, "projection": ev.get("projection", {}),
            "n": len(points), "filtered_to": cluster}


@router.post("/what-if")
def what_if(body: dict) -> dict:
    """Sweep one RFM axis and report where the segment boundary is crossed.

    The interesting output is not the curve but the crossing points: "this customer
    moves from At-risk to Loyal once they order twice more" is a concrete,
    testable merchandising target.
    """
    _store.require()
    field = body.get("field", "recency_days")
    if field not in D.FEATURES:
        raise HTTPException(422, f"field must be one of {D.FEATURES}")
    base = dict(body.get("customer") or {})

    ranges = {
        "recency_days": np.linspace(1, 365, 30),
        "frequency": np.arange(1, 31),
        "monetary": np.linspace(50, 12000, 30),
        "n_distinct_skus": np.linspace(1, 200, 30),
        "avg_basket_value": np.linspace(20, 2000, 30),
    }
    curve, previous = [], None
    transitions = []
    for v in ranges[field]:
        result = predict({**base, field: float(v)})
        entry = {"value": round(float(v), 2), "cluster": result["cluster"],
                 "segment": result["segment"], "confidence": result["confidence"]}
        curve.append(entry)
        if previous is not None and previous["cluster"] != entry["cluster"]:
            transitions.append({
                "at_value": entry["value"],
                "from_segment": previous["segment"], "to_segment": entry["segment"],
            })
        previous = entry

    return {
        "field": field, "curve": curve, "transitions": transitions,
        "note": ("Segment boundaries in one dimension, holding the others fixed. The "
                 "transition points are the concrete targets a campaign can aim at."),
    }


@router.get("/stability")
def stability() -> dict:
    """Algorithm comparison, pairwise agreement and the gap statistic."""
    _store.require()
    ev = _store.payload("evaluation") or {}
    return {
        "leaderboard": ev.get("leaderboard", []),
        "agreement": ev.get("agreement", []),
        "gap_statistic": ev.get("gap_statistic", {}),
        "k_sweep": ev.get("k_sweep", []),
        "silhouette_distribution": ev.get("silhouette_distribution", []),
        "note": ev.get("validation_note"),
    }


__all__ = ["META", "SLUG", "predict", "router", "train", "D"]
