"""Market Basket Association Mining -- project package."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...core.artifacts import ArtifactStore
from ...registry import get as _get
from . import data as D
from .train import run as train

META = _get("basket")
SLUG = "basket"
_store = ArtifactStore(SLUG)

router = APIRouter()


def _rules(confirmed_only: bool = True) -> list[dict]:
    extras = _store.payload("extras") or {}
    rules = extras.get("rules_index", [])
    if confirmed_only:
        confirmed = [r for r in rules if r.get("holdout_status") == "confirmed"]
        # Fall back to everything if the hold-out could not measure anything, so the
        # endpoint degrades to "less validated" rather than "empty".
        return confirmed or rules
    return rules


def predict(payload: dict) -> dict:
    """Complete a basket: given items in the cart, suggest what goes with them.

    Only rules that survived the temporal hold-out are served, so the
    recommendations are patterns that held on invoices the miner never saw rather
    than last season's promotion.
    """
    _store.require()
    basket = payload.get("basket") or payload.get("items") or []
    if isinstance(basket, str):
        basket = [basket]
    basket = {str(i).strip().upper() for i in basket if str(i).strip()}
    if not basket:
        raise ValueError("Provide a non-empty 'basket' array of item descriptions.")

    confirmed_only = bool(payload.get("confirmed_only", True))
    top_k = int(payload.get("top_k", 8))

    matched, suggestions = [], {}
    for r in _rules(confirmed_only):
        ants = set(r["antecedents"])
        if not ants or not ants <= basket:
            continue
        matched.append(r)
        for item in r["consequents"]:
            if item in basket:
                continue
            prev = suggestions.get(item)
            # Keep the strongest rule that recommends this item.
            if prev is None or r["lift"] > prev["lift"]:
                suggestions[item] = {
                    "item": item, "lift": r["lift"], "confidence": r["confidence"],
                    "support": r["support"],
                    "because_of": r["antecedents"],
                    "interpretation": r["interpretation"],
                    "holdout_lift": r.get("holdout_lift"),
                    "lift_retained": r.get("lift_retained"),
                }

    ranked = sorted(suggestions.values(), key=lambda s: -s["lift"])[:top_k]
    return {
        "basket": sorted(basket),
        "recommendations": ranked,
        "rules_matched": len(matched),
        "served_from": ("out-of-sample-confirmed rules only" if confirmed_only
                        else "all rules above threshold"),
        "note": (
            "Ranked by lift: how many times more likely the item is in this basket "
            "than in a random one. No recommendations means no confirmed rule fires "
            "for this combination -- which is the honest answer, not a failure."
            if not ranked else
            "Ranked by lift. These are population-level co-occurrences, not "
            "personalised predictions."
        ),
    }


@router.get("/items")
def items(q: str | None = None, limit: int = 60) -> dict:
    """Item vocabulary with basket frequency, for the autocomplete control."""
    _store.require()
    extras = _store.payload("extras") or {}
    freq = extras.get("item_frequency", [])
    if q:
        needle = q.strip().upper()
        freq = [f for f in freq if needle in f["item"]]
    return {"count": len(freq), "items": freq[:limit]}


@router.get("/rules")
def rules(min_lift: float = 1.2, status: str | None = None,
          contains: str | None = None, limit: int = 120) -> dict:
    """Filterable rule table -- the analyst's main view."""
    _store.require()
    rows = _rules(confirmed_only=False)
    out = [r for r in rows if r["lift"] >= min_lift]
    if status:
        out = [r for r in out if r.get("holdout_status") == status]
    if contains:
        needle = contains.strip().upper()
        out = [r for r in out
               if any(needle in i for i in r["antecedents"] + r["consequents"])]
    ev = _store.payload("evaluation") or {}
    return {
        "count": len(out),
        "rules": out[:limit],
        "filters": {"min_lift": min_lift, "status": status, "contains": contains},
        "holdout_validation": ev.get("holdout_validation", {}),
        "thresholds": ev.get("thresholds", {}),
    }


@router.get("/network")
def network(min_lift: float = 1.5) -> dict:
    """Node-link data for the force-directed rule graph."""
    _store.require()
    ev = _store.payload("evaluation") or {}
    net = ev.get("network", {})
    links = [l for l in net.get("links", []) if l["lift"] >= min_lift]
    keep = {l["source"] for l in links} | {l["target"] for l in links}
    nodes = [n for n in net.get("nodes", []) if n["id"] in keep]
    return {"nodes": nodes, "links": links, "min_lift": min_lift,
            "note": net.get("note")}


@router.get("/algorithm-comparison")
def algorithm_comparison() -> dict:
    """Apriori vs FP-Growth: identical output, very different runtime."""
    _store.require()
    ev = _store.payload("evaluation") or {}
    comp = ev.get("algorithm_comparison")
    if not comp:
        raise HTTPException(503, "Algorithm comparison is not in the artifacts.")
    return {**comp, "support_sensitivity":
            (_store.payload("eda") or {}).get("support_sensitivity", [])}


__all__ = ["META", "SLUG", "predict", "router", "train", "D"]
