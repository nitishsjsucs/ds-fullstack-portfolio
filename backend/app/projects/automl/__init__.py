"""AutoML Stacking Tournament -- project package."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from ...core.artifacts import ArtifactStore
from ...core.explain import local_explanation
from ...registry import get as _get
from . import data as D
from .train import run as train

META = _get("automl")
SLUG = "automl"
_store = ArtifactStore(SLUG)

router = APIRouter()


def predict(payload: dict) -> dict:
    """Score one census record with both the champion and the distilled tree.

    Returning both is the point of this project: it shows the accuracy cost of
    choosing the auditable model instead of asserting that the complex one is
    required.
    """
    _store.require()
    model = _store.model("model")
    frame, context = D.build_inference_row(payload)

    prob = float(model.predict_proba(frame)[:, 1][0])
    out = {
        "probability_above_50k": round(prob, 4),
        "prediction": ">50K" if prob >= 0.5 else "<=50K",
        "context": context,
    }

    if _store.has_model("distilled"):
        student = _store.model("distilled")
        s_prob = float(student.predict_proba(frame)[:, 1][0])
        out["distilled_tree"] = {
            "probability_above_50k": round(s_prob, 4),
            "prediction": ">50K" if s_prob >= 0.5 else "<=50K",
            "agrees_with_champion": bool((s_prob >= 0.5) == (prob >= 0.5)),
            "note": ("A depth-4 tree trained to imitate the ensemble. Where it agrees, "
                     "the ensemble's answer is explainable in four questions."),
        }

    try:
        prep = model.named_steps["prep"]
        names = list(prep.get_feature_names_out())
        row_t = pd.DataFrame(prep.transform(frame), columns=names)
        out["attribution"] = local_explanation(model, row_t, row_t)
    except Exception:
        out["attribution"] = None

    out["disclaimer"] = (
        "Trained on a 1994 US census extract. This is a methodology benchmark and must "
        "not inform any decision about any real person."
    )
    return out


@router.get("/form-schema")
def form_schema() -> dict:
    _store.require()
    extras = _store.payload("extras") or {}
    return {"domains": extras.get("feature_domains", {}),
            "profiles": extras.get("profiles", []),
            "dropped_columns": extras.get("dropped_columns", {})}


@router.get("/tournament")
def tournament() -> dict:
    """The full leaderboard with the CV-to-hold-out gap per family."""
    _store.require()
    lb = _store.payload("leaderboard") or {}
    ev = _store.payload("evaluation") or {}
    return {
        "leaderboard": lb.get("leaderboard", []),
        "protocol": lb.get("protocol"),
        "search_overfitting": ev.get("search_overfitting", {}),
        "stacking": ev.get("stacking", {}),
        "ensemble": lb.get("ensemble", {}),
        "deployed": ev.get("deployed"),
        "total_trials": lb.get("total_trials"),
    }


@router.get("/trajectory")
def trajectory(family: str | None = None, limit: int = 400) -> dict:
    """Every hill-climbing trial in order -- the search replayed."""
    _store.require()
    lb = _store.payload("leaderboard") or {}
    trials = lb.get("trajectory", [])
    if family:
        trials = [t for t in trials if t.get("family") == family]
    accepted = [t for t in trials if t.get("accepted")]
    return {
        "trials": trials[:limit],
        "n_trials": len(trials),
        "n_accepted": len(accepted),
        "families": sorted({t["family"] for t in lb.get("trajectory", [])}),
        "scoring": lb.get("scoring"),
        "note": ("Each point is one cross-validated fit. Accepted steps are the moves "
                 "hill climbing kept; the rest were tried and rejected. The rejected "
                 "trials are shown because a search that only reports its successes "
                 "hides how much of the improvement was luck."),
    }


@router.get("/distillation")
def distillation() -> dict:
    """The depth-4 tree that imitates the ensemble, with its rules in full."""
    _store.require()
    ev = _store.payload("evaluation") or {}
    d = ev.get("distillation")
    if not d:
        raise HTTPException(503, "Distillation is not in the artifacts.")
    return d


@router.get("/fairness")
def fairness_report() -> dict:
    _store.require()
    extras = _store.payload("extras") or {}
    report = extras.get("fairness")
    if not report:
        raise HTTPException(503, "Fairness audit is not in the artifacts.")
    return report


__all__ = ["META", "SLUG", "predict", "router", "train", "D"]
