"""Bike-Share Demand Forecasting -- project package."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ...core.artifacts import ArtifactStore
from ...registry import get as _get
from . import data as D
from .train import run as train

META = _get("forecast")
SLUG = "forecast"
_store = ArtifactStore(SLUG)

router = APIRouter()


def _feature_row(payload: dict) -> tuple[pd.DataFrame, dict]:
    """Assemble one feature vector from a caller-supplied context.

    The model needs recent history. Rather than pretend otherwise, the endpoint
    accepts an explicit ``recent_demand`` array and says plainly in the response
    when it has fallen back to defaults, because a forecast built on invented
    history is worse than no forecast.
    """
    extras = _store.payload("extras") or {}
    cols = extras.get("features", [])

    when = pd.to_datetime(payload.get("timestamp") or "2012-10-15T08:00:00")
    recent = payload.get("recent_demand")
    supplied = isinstance(recent, list) and len(recent) >= 1
    if supplied:
        hist = np.asarray([float(v) for v in recent], dtype=float)
    else:
        window = extras.get("recent_window", [])
        hist = np.asarray([float(r["actual"]) for r in window], dtype=float)

    def lag(h: int) -> float:
        return float(hist[-h]) if len(hist) >= h else float(np.median(hist) if len(hist) else 190.0)

    row: dict = {}
    row["hour"] = when.hour
    row["dayofweek"] = when.dayofweek
    row["month"] = when.month
    row["dayofyear"] = when.dayofyear
    row["is_weekend"] = int(when.dayofweek >= 5)
    for col, period in (("hour", 24), ("dayofweek", 7), ("month", 12)):
        theta = 2 * np.pi * row[col] / period
        row[f"{col}_sin"], row[f"{col}_cos"] = np.sin(theta), np.cos(theta)

    for l in D.LAGS:
        row[f"lag_{l}"] = lag(l)
    past = hist[:-1] if len(hist) > 1 else hist
    for w in D.ROLLING_WINDOWS:
        tail = past[-w:] if len(past) else np.array([190.0])
        row[f"roll_mean_{w}"] = float(np.mean(tail))
        row[f"roll_std_{w}"] = float(np.std(tail)) if len(tail) > 1 else 0.0
    row["roll_max_24"] = float(np.max(past[-24:])) if len(past) else 190.0
    row["delta_24"] = row["lag_1"] - row["lag_25"]

    row["season"] = int(payload.get("season", (when.month % 12) // 3 + 1))
    row["holiday"] = int(payload.get("holiday", 0))
    row["weekday"] = when.dayofweek
    row["workingday"] = int(payload.get("workingday", 0 if when.dayofweek >= 5 else 1))
    row["weathersit"] = int(payload.get("weathersit", 1))
    row["temp"] = float(payload.get("temp", 0.5))
    row["atemp"] = float(payload.get("atemp", row["temp"]))
    row["hum"] = float(payload.get("hum", 0.55))
    row["windspeed"] = float(payload.get("windspeed", 0.19))

    frame = pd.DataFrame([{c: row.get(c, 0.0) for c in cols}])
    context = {
        "timestamp": when.isoformat(),
        "hour": int(when.hour),
        "day_of_week": int(when.dayofweek),
        "working_day": bool(row["workingday"]),
        "weather": int(row["weathersit"]),
        "temp_normalised": row["temp"],
        "history_hours_supplied": int(len(hist)) if supplied else 0,
        "history_source": ("caller-supplied" if supplied
                           else "last observed window from the training archive"),
    }
    return frame, context


def predict(payload: dict) -> dict:
    """Forecast demand for one hour, with an 80% band."""
    _store.require()
    model = _store.model("model")
    frame, context = _feature_row(payload)

    point = float(np.clip(model.predict(frame)[0], 0, None))
    extras = _store.payload("extras") or {}
    conformal = extras.get("conformal") or {}
    adjustment = float(conformal.get("adjustment") or 0.0)

    lo = hi = None
    if _store.has_model("q_low") and _store.has_model("q_high"):
        a = float(np.clip(_store.model("q_low").predict(frame)[0], 0, None))
        b = float(np.clip(_store.model("q_high").predict(frame)[0], 0, None))
        # Apply the same conformal widening that was fitted at training time --
        # serving the raw quantile band would give a nominal 80% interval that
        # actually covers about 70%.
        lo, hi = max(0.0, min(a, b) - adjustment), max(a, b) + adjustment

    ev = _store.payload("evaluation") or {}
    coverage = (ev.get("interval") or {}).get("empirical_coverage")

    return {
        "predicted_rides": round(point, 1),
        "interval_80": ({"lower": round(lo, 1), "upper": round(hi, 1)}
                        if lo is not None else None),
        "context": context,
        "interval_note": (
            f"Conformally calibrated: the raw quantile band covered "
            f"{(ev.get('interval') or {}).get('before_calibration', {}).get('empirical_coverage', 0):.1%} "
            f"of held-out hours, and after a +/-{adjustment:.1f} ride adjustment it "
            f"covers {coverage:.1%} against 80% nominal."
            if coverage else None
        ),
        "caveat": (
            "This model needs 168 hours of contiguous recent history. "
            + ("Caller supplied history was used."
               if context["history_hours_supplied"]
               else "No history was supplied, so the last observed window from the "
                    "training archive stands in -- treat this as illustrative rather "
                    "than a live forecast.")
        ),
    }


@router.get("/decomposition")
def decomposition() -> dict:
    """STL trend/seasonal/residual split with variance attribution."""
    _store.require()
    eda = _store.payload("eda") or {}
    dec = eda.get("decomposition")
    if not dec:
        raise HTTPException(503, "Decomposition is not in the artifacts.")
    return {**dec, "stationarity": eda.get("stationarity", {})}


@router.get("/correlogram")
def correlogram() -> dict:
    """ACF and PACF with significance bands -- the basis for the lag set."""
    _store.require()
    eda = _store.payload("eda") or {}
    c = eda.get("correlogram")
    if not c:
        raise HTTPException(503, "Correlogram is not in the artifacts.")
    return c


@router.get("/backtest")
def backtest() -> dict:
    """Per-fold rolling-origin results and the purge/embargo protocol."""
    _store.require()
    lb = _store.payload("leaderboard") or {}
    ev = _store.payload("evaluation") or {}
    return {
        "leaderboard": lb.get("leaderboard", []),
        "folds": lb.get("backtest", {}),
        "protocol": lb.get("protocol", {}),
        "baselines": ev.get("baselines", []),
        "horizons": ev.get("horizons", {}),
    }


@router.get("/series")
def series(limit: int = 800) -> dict:
    """Actual vs predicted with the 80% band, over the hold-out period."""
    _store.require()
    ev = _store.payload("evaluation") or {}
    pts = ev.get("series", [])[:limit]
    inside = sum(1 for p in pts if p.get("inside"))
    return {
        "points": pts,
        "n": len(pts),
        "empirical_coverage_shown": round(inside / len(pts), 4) if pts else None,
        "interval": ev.get("interval", {}),
        "champion": ev.get("champion"),
    }


@router.post("/scenario")
def scenario(body: dict) -> dict:
    """Forecast a full day under a chosen weather scenario.

    The comparison against clear weather is the actionable output: it prices the
    demand impact of rain for a rebalancing team.
    """
    _store.require()
    base_date = body.get("date") or "2012-10-15"
    scenarios = {1: "Clear", 2: "Mist/cloudy", 3: "Light rain/snow"}
    chosen = int(body.get("weathersit", 1))
    if chosen not in scenarios:
        raise HTTPException(422, f"weathersit must be one of {sorted(scenarios)}")

    out = {}
    for w in (1, chosen) if chosen != 1 else (1,):
        curve = []
        for hour in range(24):
            r = predict({**body, "timestamp": f"{base_date}T{hour:02d}:00:00",
                         "weathersit": w})
            curve.append({"hour": hour, "predicted_rides": r["predicted_rides"],
                          "lower": (r["interval_80"] or {}).get("lower"),
                          "upper": (r["interval_80"] or {}).get("upper")})
        out[scenarios[w]] = curve

    baseline_total = sum(p["predicted_rides"] for p in out["Clear"])
    result = {"date": base_date, "scenarios": out,
              "daily_totals": {k: round(sum(p["predicted_rides"] for p in v), 1)
                               for k, v in out.items()}}
    if chosen != 1:
        chosen_total = sum(p["predicted_rides"] for p in out[scenarios[chosen]])
        result["weather_impact"] = {
            "scenario": scenarios[chosen],
            "rides_lost": round(baseline_total - chosen_total, 1),
            "pct_change": round((chosen_total - baseline_total) / max(1, baseline_total), 4),
        }
    return result


__all__ = ["META", "SLUG", "predict", "router", "train", "D"]
