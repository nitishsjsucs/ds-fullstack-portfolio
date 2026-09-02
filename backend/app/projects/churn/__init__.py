"""Telco Churn & Retention Economics -- project package."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ...core.artifacts import ArtifactStore
from ...core.explain import local_explanation
from ...registry import get as _get
from . import data as D
from .train import run as train

META = _get("churn")
SLUG = "churn"
_store = ArtifactStore(SLUG)

router = APIRouter()


def _threshold() -> float:
    extras = _store.payload("extras") or {}
    return float(extras.get("deployed_threshold", 0.5))


def predict(payload: dict) -> dict:
    """Score one subscriber and say whether an offer is worth making."""
    _store.require()
    model = _store.model("model")
    frame, context = D.build_inference_row(payload)

    prob = float(model.predict_proba(frame)[:, 1][0])
    threshold = float(payload.get("threshold") or _threshold())
    econ = {k: v for k, v in (payload.get("economics") or {}).items()
            if isinstance(v, (int, float))}
    value = D.expected_value(prob, context["monthly_charges"], econ)

    # SHAP needs the uncalibrated tree pipeline; the calibrated wrapper hides it.
    attribution = None
    if _store.has_model("uncalibrated"):
        try:
            base = _store.model("uncalibrated")
            prep = base.named_steps["prep"]
            names = list(prep.get_feature_names_out())
            row_t = pd.DataFrame(prep.transform(frame), columns=names)
            attribution = local_explanation(base, row_t, row_t)
        except Exception:
            attribution = None

    band = ("high" if prob >= 0.66 else "medium" if prob >= 0.33 else "low")
    return {
        "churn_probability": round(prob, 4),
        "risk_band": band,
        "decision_threshold": threshold,
        "flagged_for_outreach": bool(prob >= threshold),
        "economics": value,
        "context": context,
        "attribution": attribution,
        "note": (
            f"Threshold {threshold:.2f} minimises expected cost under the stated "
            f"economics; it is deliberately not 0.50. Probability is "
            f"{_store.provenance().get('calibration', 'calibrated') if _store.provenance() else 'calibrated'}-"
            f"calibrated so it can be multiplied by margin."
        ),
    }


@router.get("/form-schema")
def form_schema() -> dict:
    """Valid values and ranges, so the UI can build correct controls."""
    _store.require()
    extras = _store.payload("extras") or {}
    return {
        "domains": extras.get("feature_domains", {}),
        "personas": extras.get("personas", []),
        "economics": extras.get("economics", D.ECONOMICS),
        "deployed_threshold": _threshold(),
    }


@router.post("/what-if")
def what_if(body: dict) -> dict:
    """Sweep one field across its range, holding everything else fixed.

    This is the counterfactual view a retention analyst actually wants: "if this
    customer moved to a one-year contract, how much does their risk fall?"
    """
    _store.require()
    field = body.get("field", "tenure")
    base = dict(body.get("customer") or {})
    model = _store.model("model")

    extras = _store.payload("extras") or {}
    domains = extras.get("feature_domains", {})
    ranges = domains.get("_numeric_ranges", {})

    if field in ranges:
        lo, hi = ranges[field]
        values = list(np.linspace(float(lo), float(hi), 24).round(2))
    elif field in domains:
        values = list(domains[field])
    else:
        raise HTTPException(422, f"Unknown field {field!r}. "
                                 f"Choose from {sorted(set(domains) | set(ranges))}.")

    frames, labels = [], []
    for v in values:
        f, _ = D.build_inference_row({**base, field: v})
        frames.append(f)
        labels.append(v)
    probs = model.predict_proba(pd.concat(frames, ignore_index=True))[:, 1]

    curve = [{"value": (float(v) if isinstance(v, (int, float, np.floating)) else str(v)),
              "churn_probability": round(float(p), 4)} for v, p in zip(labels, probs)]
    lowest = min(curve, key=lambda r: r["churn_probability"])
    highest = max(curve, key=lambda r: r["churn_probability"])
    return {
        "field": field,
        "curve": curve,
        "swing": round(highest["churn_probability"] - lowest["churn_probability"], 4),
        "best_value": lowest["value"],
        "worst_value": highest["value"],
        "threshold": _threshold(),
        "note": ("A counterfactual sweep over one field with all others held fixed. "
                 "It shows the model's learned sensitivity, which is not the same as "
                 "a causal effect -- changing a customer's contract in reality also "
                 "changes the kind of customer they are."),
    }


@router.post("/campaign")
def campaign(body: dict) -> dict:
    """Recompute the campaign value curve under user-supplied economics."""
    _store.require()
    ev = _store.payload("evaluation") or {}
    sim = ev.get("campaign_simulation")
    if not sim:
        raise HTTPException(503, "Campaign simulation is not in the artifacts.")

    econ = {**D.ECONOMICS, **{k: float(v) for k, v in (body or {}).items()
                              if isinstance(v, (int, float))}}
    stored = ev.get("economics", {})
    clv = (stored.get("mean_monthly_charges", 65.0)
           * econ["monthly_margin_pct"] * econ["expected_remaining_months"])

    rows = []
    for r in sim["curve"]:
        cost = r["customers_contacted"] * econ["retention_offer_cost"]
        saved = r["churners_reached"] * econ["offer_acceptance_rate"] * clv
        rows.append({**r, "campaign_cost": round(cost, 2),
                     "revenue_saved": round(saved, 2),
                     "net_value": round(saved - cost, 2),
                     "roi": round((saved - cost) / cost, 3) if cost else None})
    best = max(rows, key=lambda r: r["net_value"])
    return {
        "economics": econ,
        "customer_lifetime_margin": round(clv, 2),
        "curve": rows,
        "best": best,
        "note": (f"Under these assumptions the optimum is to contact the top "
                 f"{best['targeted_pct']}% by predicted risk, netting "
                 f"${best['net_value']:,.0f} on the hold-out."),
    }


@router.get("/fairness")
def fairness_report() -> dict:
    _store.require()
    extras = _store.payload("extras") or {}
    report = extras.get("fairness")
    if not report:
        raise HTTPException(503, "Fairness audit is not in the artifacts.")
    return report


__all__ = ["META", "SLUG", "predict", "router", "train", "D"]
