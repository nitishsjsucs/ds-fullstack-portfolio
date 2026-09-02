"""NanoGPT: A Transformer From Scratch -- project package."""

from __future__ import annotations

import torch
from fastapi import APIRouter, HTTPException

from ...core.artifacts import ArtifactStore
from ...registry import get as _get
from . import data as D
from .model import ModelConfig, NanoGPT
from .train import run as train

META = _get("nanollm")
SLUG = "nanollm"
_store = ArtifactStore(SLUG)

router = APIRouter()

_cache: dict = {}


def _load():
    """Load the checkpoint once and keep it resident."""
    if "model" in _cache:
        return _cache["model"], _cache["tokenizer"]
    path = _store.dir / "model.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Run `python scripts/train_all.py --only {SLUG}`."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = {k: v for k, v in ckpt["config"].items() if k != "head_dim"}
    cfg = ModelConfig(**cfg_dict)
    model = NanoGPT(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tok = D.CharTokenizer("")
    tok.chars = ckpt["chars"]
    tok.stoi = {c: i for i, c in enumerate(tok.chars)}
    tok.itos = {i: c for c, i in tok.stoi.items()}

    _cache["model"], _cache["tokenizer"] = model, tok
    return model, tok


def predict(payload: dict) -> dict:
    """Generate a continuation. Alias for the /generate endpoint."""
    return generate(payload)


@router.post("/generate")
def generate(body: dict) -> dict:
    """Sample a continuation with live temperature and top-k control."""
    try:
        model, tok = _load()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc

    prompt = str(body.get("prompt") or "ROMEO:\n")
    max_new = int(body.get("max_new_tokens", 240))
    if not 1 <= max_new <= 800:
        raise HTTPException(422, "max_new_tokens must be between 1 and 800.")
    temperature = float(body.get("temperature", 0.8))
    if not 0 <= temperature <= 2.0:
        raise HTTPException(422, "temperature must be between 0 and 2.")
    top_k = body.get("top_k", 40)
    top_k = int(top_k) if top_k else None
    seed = int(body.get("seed", 42))

    ids = tok.encode(prompt)
    dropped = len(prompt) - len(ids)
    if not ids:
        ids = tok.encode("\n") or [0]

    torch.manual_seed(seed)
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new, temperature=temperature, top_k=top_k)
    text = tok.decode(out[0].tolist())

    return {
        "prompt": prompt,
        "generated": text[len(tok.decode(ids)):],
        "full_text": text,
        "settings": {"temperature": temperature, "top_k": top_k,
                     "max_new_tokens": max_new, "seed": seed},
        "characters_dropped_from_prompt": dropped,
        "prompt_note": (
            f"{dropped} character(s) in the prompt are outside the 65-symbol vocabulary "
            f"and were dropped." if dropped else None
        ),
        "caveat": (
            "This model has no knowledge and cannot answer questions. It predicts the "
            "next character from a 1.1M-character Shakespeare corpus, so it produces "
            "Shakespeare-shaped text and nothing more."
        ),
    }


@router.get("/architecture")
def architecture() -> dict:
    """Component-by-component parameter breakdown and design rationale."""
    _store.require()
    extras = _store.payload("extras") or {}
    arch = extras.get("architecture")
    if not arch:
        raise HTTPException(503, "Architecture summary is not in the artifacts.")
    return {**arch, "vocabulary": extras.get("vocabulary", {})}


@router.get("/attention")
def attention() -> dict:
    """Attention heat-maps read from the trained weights."""
    _store.require()
    ex = _store.payload("explain") or {}
    att = ex.get("attention")
    if not att:
        raise HTTPException(503, "Attention maps are not in the artifacts.")
    return att


@router.get("/training-curve")
def training_curve() -> dict:
    """Train and validation loss together, with the divergence point marked."""
    _store.require()
    ev = _store.payload("evaluation") or {}
    return {
        "history": ev.get("history", []),
        "best": ev.get("best"),
        "baselines": ev.get("baselines", {}),
        "overfitting": ev.get("overfitting", {}),
        "perplexity": ev.get("perplexity"),
        "improvement_over_unigram": ev.get("improvement_over_unigram"),
        "metric_note": ev.get("metric_note"),
    }


@router.get("/samples")
def samples() -> dict:
    """Pre-generated samples across the temperature range."""
    _store.require()
    ev = _store.payload("evaluation") or {}
    extras = _store.payload("extras") or {}
    return {
        "samples": ev.get("samples", []),
        "temperature_guide": extras.get("temperature_guide", []),
        "prompts": extras.get("prompts", []),
        "note": ("All five use the same prompt and the same seed; only temperature "
                 "changes. Samples illustrate what the loss corresponds to -- they are "
                 "not evidence of it."),
    }


__all__ = ["META", "SLUG", "predict", "router", "train", "D"]
