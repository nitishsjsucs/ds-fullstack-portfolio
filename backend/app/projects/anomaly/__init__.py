"""Network Intrusion Anomaly Detection -- project package."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ...core.artifacts import ArtifactStore
from ...registry import get as _get
from . import data as D
from .train import run as train

META = _get("anomaly")
SLUG = "anomaly"
_store = ArtifactStore(SLUG)

router = APIRouter()


def _reference_percentiles() -> list[float] | None:
    """Score distribution from training, used to turn a raw score into a percentile."""
    ev = _store.payload("evaluation") or {}
    sep = ev.get("score_separation")
    return sep


def predict(payload: dict) -> dict:
    """Score one connection and explain which features look unusual."""
    _store.require()
    if not _store.has_model("model"):
        raise FileNotFoundError(
            "The deployed detector for this project is a rank-fusion ensemble, which "
            "has no single picklable estimator. Use /api/projects/anomaly/score-sample "
            "to score one of the stored example connections instead."
        )
    model = _store.model("model")
    extras = _store.payload("extras") or {}
    ranges = extras.get("feature_ranges", {})

    row = {}
    for col, spec in ranges.items():
        if isinstance(spec, dict):
            row[col] = float(payload.get(col, spec.get("median", 0.0)))
        else:
            row[col] = str(payload.get(col, spec[0] if spec else "other"))
    frame = pd.DataFrame([row])

    est = model.named_steps["model"]
    prep = model.named_steps["prep"]
    Xt = prep.transform(frame)
    raw = float(-est.score_samples(Xt)[0]) if hasattr(est, "score_samples") else 0.0

    deviations = _deviation(row, extras)
    return {
        "anomaly_score": round(raw, 5),
        "verdict": ("unusual -- worth review" if raw > 0 else "consistent with normal traffic"),
        "deviating_features": deviations[:8],
        "input": row,
        "caveat": (
            "An anomaly score measures deviation from the majority, not maliciousness. "
            "A flagged connection is unusual; whether it is an attack is a judgement "
            "for an analyst."
        ),
    }


def _deviation(row: dict, extras: dict) -> list[dict]:
    ranges = extras.get("feature_ranges", {})
    out = []
    for col, value in row.items():
        spec = ranges.get(col)
        if not isinstance(spec, dict) or not isinstance(value, (int, float)):
            continue
        med = spec.get("median", 0.0)
        span = max(1e-9, spec.get("max", 1.0) - spec.get("min", 0.0))
        z = (value - med) / span
        out.append({"feature": col, "value": value, "population_median": med,
                    "normalised_deviation": round(float(z), 4)})
    out.sort(key=lambda r: -abs(r["normalised_deviation"]))
    return out


@router.get("/leaderboard")
def leaderboard() -> dict:
    """Five detectors, their assumptions, and where each one is blind."""
    _store.require()
    lb = _store.payload("leaderboard") or {}
    ex = _store.payload("explain") or {}
    return {**lb, "assumptions": ex.get("detector_assumptions", [])}


@router.get("/alert-budget")
def alert_budget(budget: int | None = None) -> dict:
    """Analyst-queue economics: what a shift of N alerts actually catches."""
    _store.require()
    ev = _store.payload("evaluation") or {}
    rows = ev.get("alert_budget", [])
    if not rows:
        raise HTTPException(503, "Alert-budget simulation is not in the artifacts.")
    selected = None
    if budget is not None:
        selected = min(rows, key=lambda r: abs(r["alert_budget"] - budget))
    return {
        "curve": rows,
        "selected": selected,
        "per_family_recall": ev.get("per_family_recall", []),
        "note": ("Precision falls as the queue grows: the top alerts are the most "
                 "clear-cut. Sizing an analyst team is therefore a direct trade-off "
                 "between recall and hours, and this table prices it."),
    }


@router.get("/samples")
def samples() -> dict:
    """Real scored connections from the dataset, for the live demo."""
    _store.require()
    extras = _store.payload("extras") or {}
    return {
        "samples": extras.get("sample_connections", []),
        "attack_families": extras.get("attack_families", {}),
        "note": ("Half are the highest-scoring connections in the dataset, half are "
                 "median-scoring. True labels are shown so you can see where the "
                 "detector is right and where it is not."),
    }


@router.get("/projection")
def projection() -> dict:
    """PCA scatter coloured by ground truth -- shows where detection is impossible."""
    _store.require()
    ev = _store.payload("evaluation") or {}
    proj = ev.get("projection")
    if not proj:
        raise HTTPException(503, "Projection is not in the artifacts.")
    return proj


__all__ = ["META", "SLUG", "predict", "router", "train", "D"]
