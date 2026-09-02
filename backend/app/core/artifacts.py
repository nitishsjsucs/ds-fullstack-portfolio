"""Artifact persistence: the contract between training and serving.

Training is offline and slow; serving must boot in under a second. The seam
between them is this module. ``TrainingArtifacts.save`` writes a fixed set of JSON
payloads plus a pickled pipeline; ``ArtifactStore`` lazily memoises them for the
API.

Why JSON for the payloads rather than recomputing on request: an endpoint that
recomputes SHAP on every call is both slow and non-deterministic across restarts,
and it hides the fact that the numbers in the dashboard should be a *record of a
specific training run*. Pinning them makes the run auditable -- the report you
read is the report that was produced, with the git commit and timestamp attached.
"""

from __future__ import annotations

import pickle
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import paths
from .serialization import read_json, write_json

# The payload names every project writes. Missing ones simply serve `null`, so a
# project without, say, clustering does not need a placeholder file.
PAYLOADS = (
    "overview", "eda", "crispdm", "leaderboard", "evaluation",
    "explain", "model_card", "audit", "extras",
)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def run_provenance(extra: dict | None = None) -> dict:
    """Stamp every artifact with enough context to reproduce or discredit it."""
    return {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        **(extra or {}),
    }


@dataclass
class TrainingArtifacts:
    """Collected outputs of one project's training run."""

    slug: str
    payloads: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, payload: Any) -> "TrainingArtifacts":
        self.payloads[name] = payload
        return self

    def add_model(self, name: str, obj: Any) -> "TrainingArtifacts":
        self.models[name] = obj
        return self

    def save(self, *, provenance: dict | None = None) -> Path:
        out = paths.artifact_dir(self.slug)
        prov = run_provenance(provenance)
        for name, payload in self.payloads.items():
            if isinstance(payload, dict):
                payload = {**payload, "_provenance": prov}
            write_json(out / f"{name}.json", payload)
        for name, obj in self.models.items():
            with (out / f"{name}.pkl").open("wb") as fh:
                pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(out / "_manifest.json", {
            "slug": self.slug,
            "payloads": sorted(self.payloads),
            "models": sorted(self.models),
            "provenance": prov,
        })
        return out


class ArtifactStore:
    """Read-side accessor with in-process memoisation."""

    def __init__(self, slug: str) -> None:
        self.slug = slug
        self.dir = paths.ARTIFACTS / slug
        self._json_cache: dict[str, Any] = {}
        self._model_cache: dict[str, Any] = {}

    # -- availability ------------------------------------------------------ #
    @property
    def is_trained(self) -> bool:
        return (self.dir / "_manifest.json").exists()

    def require(self) -> None:
        if not self.is_trained:
            raise FileNotFoundError(
                f"No artifacts for '{self.slug}'. Run `python scripts/train_all.py "
                f"--only {self.slug}` to produce them."
            )

    # -- payloads ---------------------------------------------------------- #
    def payload(self, name: str, default: Any = None) -> Any:
        if name in self._json_cache:
            return self._json_cache[name]
        path = self.dir / f"{name}.json"
        if not path.exists():
            return default
        value = read_json(path)
        self._json_cache[name] = value
        return value

    def model(self, name: str = "model") -> Any:
        if name in self._model_cache:
            return self._model_cache[name]
        path = self.dir / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Model '{name}' for '{self.slug}' is not on disk ({path}). "
                f"Run `python scripts/train_all.py --only {self.slug}`."
            )
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        self._model_cache[name] = obj
        return obj

    def has_model(self, name: str = "model") -> bool:
        return (self.dir / f"{name}.pkl").exists()

    def provenance(self) -> dict | None:
        m = self.payload("_manifest")
        return (m or {}).get("provenance")

    def invalidate(self) -> None:
        self._json_cache.clear()
        self._model_cache.clear()
