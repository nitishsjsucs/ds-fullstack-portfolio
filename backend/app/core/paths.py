"""Canonical filesystem layout.

Every module resolves paths through here so the repository can be cloned to any
location and the training scripts, the API server and the test suite all agree on
where data and artifacts live.
"""

from __future__ import annotations

from pathlib import Path

APP = Path(__file__).resolve().parents[1]
BACKEND = APP.parent
ROOT = BACKEND.parent

DATA = ROOT / "data"
CURATED = DATA / "curated"
RAW = DATA / "raw"
MANIFEST = DATA / "MANIFEST.json"

ARTIFACTS = BACKEND / "artifacts"
DOCS = ROOT / "docs"
SCREENSHOTS = DOCS / "screenshots"
PAPERS = DOCS / "papers"


def artifact_dir(slug: str) -> Path:
    """Directory holding one project's trained outputs; created on demand."""
    d = ARTIFACTS / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def curated(name: str) -> Path:
    """Resolve a curated dataset by stem, accepting either committed format."""
    for suffix in (".parquet", ".csv.gz", ".txt"):
        p = CURATED / f"{name}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Curated dataset {name!r} not found in {CURATED}. "
        "Run `python scripts/fetch_data.py` first."
    )
