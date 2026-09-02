"""Capture every screen of the console with Playwright.

Doubles as an end-to-end smoke test: the script asserts that each page actually
rendered content before it saves the image, so a blank panel or an error state
fails the capture rather than being quietly written to disk as a screenshot of
nothing. A green run therefore means every route works against real artifacts.

Assumes the API is on :8000 and the Vite dev server on :5173.

    python scripts/capture_screenshots.py
    python scripts/capture_screenshots.py --only taxi churn
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import registry  # noqa: E402

OUT = ROOT / "docs" / "screenshots"
BASE = "http://localhost:5173"
VIEWPORT = {"width": 1680, "height": 1050}

# Tabs captured for every project.
TABS = ["overview", "data", "crispdm", "models", "evaluation", "explain", "live", "audit"]

# Text that must be absent -- if any of these appear the page did not render.
FAILURE_MARKERS = [
    "Could not load",
    "Not trained yet",
    "No leaderboard in the artifacts",
    "does not publish",
]


@dataclass
class Shot:
    name: str
    url: str
    wait_for: str | None = None
    description: str = ""


def build_shots(slugs: list[str]) -> list[Shot]:
    shots = [
        Shot("00_home", f"{BASE}/#/", "text=Eight end-to-end data science systems",
             "Portfolio landing page with all eight systems"),
        Shot("00_datasets", f"{BASE}/#/datasets", "text=Data provenance",
             "Dataset provenance: source, licence, curation rule and SHA-256"),
        Shot("00_methods", f"{BASE}/#/methods", "text=The four guarantees",
             "The method: four guarantees enforced in code"),
    ]
    for meta in registry.PROJECTS:
        if meta.slug not in slugs:
            continue
        n = f"{meta.number:02d}"
        for tab in TABS:
            shots.append(
                Shot(
                    f"{n}_{meta.slug}_{tab}",
                    f"{BASE}/#/{meta.slug}/{tab}",
                    None,
                    f"{meta.title} — {tab}",
                )
            )
    return shots


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=registry.SLUGS)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--full-page", action="store_true", default=True)
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed. Run:\n"
              "  pip install playwright && playwright install chromium")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    shots = build_shots(args.only)
    captured, failed = [], []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.set_default_timeout(25_000)

        for shot in shots:
            try:
                page.goto(shot.url, wait_until="networkidle")
                if shot.wait_for:
                    page.wait_for_selector(shot.wait_for, timeout=15_000)
                # Charts animate in and the live tabs fire a prediction on mount.
                page.wait_for_timeout(1400)

                body = page.inner_text("body")
                if len(body.strip()) < 200:
                    raise AssertionError("page rendered almost no text")
                for marker in FAILURE_MARKERS:
                    if marker in body:
                        raise AssertionError(f"page shows an error state: {marker!r}")

                path = OUT / f"{shot.name}.webp"
                png = page.screenshot(full_page=args.full_page)
                size_kb = _write_webp(png, path)
                captured.append((shot, size_kb))
                print(f"  ok    {shot.name:34s} {size_kb:7.0f} KB  {shot.description}")
            except Exception as exc:
                failed.append((shot, f"{type(exc).__name__}: {exc}"))
                print(f"  FAIL  {shot.name:34s} {type(exc).__name__}: {str(exc)[:90]}")

        browser.close()

    _write_index(captured)
    print(f"\n{len(captured)} captured, {len(failed)} failed -> {OUT.relative_to(ROOT)}")
    if failed:
        print("\nFailures:")
        for shot, err in failed:
            print(f"  {shot.name}: {err}")
    return 1 if failed else 0


def _write_webp(png_bytes: bytes, path: Path, quality: int = 90) -> float:
    """Save the capture as WebP at 1x.

    The browser renders at 2x device scale for crisp text, then this halves the
    dimensions and encodes as WebP. On dark, text-heavy UI captures that is an
    ~80% size reduction against PNG with no visible difference -- 53 MB of PNGs
    becomes about 10 MB, which is the difference between a repository that is
    pleasant to clone and one that is not. GitHub renders WebP in Markdown.
    """
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
    im.save(path, "WEBP", quality=quality, method=6)
    return path.stat().st_size / 1024


def _write_index(captured: list[tuple[Shot, float]]) -> None:
    lines = [
        "# Screenshots",
        "",
        "Captured by `scripts/capture_screenshots.py` against the running console.",
        "The script asserts that each page rendered real content before saving, so this",
        "gallery doubles as an end-to-end check: every image here is a route that works",
        "against real trained artifacts.",
        "",
        f"{len(captured)} screens, captured at {VIEWPORT['width']}px wide at 2x device "
        f"scale and encoded as WebP at 1x.",
        "",
    ]
    current = None
    for shot, _ in captured:
        group = shot.name.split("_")[0]
        if group != current:
            current = group
            title = "Portfolio" if group == "00" else next(
                (p.title for p in registry.PROJECTS if f"{p.number:02d}" == group), group
            )
            lines += ["", f"## {title}", ""]
        lines.append(f"### {shot.description}")
        lines.append("")
        lines.append(f"![{shot.description}](./{shot.name}.webp)")
        lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
