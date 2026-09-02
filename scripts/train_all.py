"""Train every project and persist its artifacts.

    python scripts/train_all.py                  # all projects, full run
    python scripts/train_all.py --only taxi      # one project
    python scripts/train_all.py --quick          # small samples, few trials

``--quick`` exists for iteration: it shrinks the search budget and the row counts
so the whole portfolio trains in a couple of minutes. Numbers produced under
``--quick`` are marked as such in the artifact provenance so they can never be
mistaken for the published results.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="restrict to these project slugs")
    ap.add_argument("--skip", nargs="*", default=[], help="skip these project slugs")
    ap.add_argument("--quick", action="store_true", help="fast, low-fidelity run")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going if one project fails")
    args = ap.parse_args()

    slugs = args.only or registry.SLUGS
    slugs = [s for s in slugs if s not in args.skip]
    unknown = [s for s in slugs if s not in registry.SLUGS]
    if unknown:
        print(f"Unknown project(s): {', '.join(unknown)}")
        print(f"Known: {', '.join(registry.SLUGS)}")
        return 2

    print(f"\nTraining {len(slugs)} project(s): {', '.join(slugs)}"
          f"{'  [QUICK MODE]' if args.quick else ''}\n")
    t0 = time.perf_counter()
    ok, failed = [], []

    for slug in slugs:
        print(f"=== {slug} " + "=" * (56 - len(slug)))
        started = time.perf_counter()
        try:
            mod = importlib.import_module(f"app.projects.{slug}.train")
            mod.run(quick=args.quick)
            ok.append((slug, time.perf_counter() - started))
        except Exception as exc:
            failed.append((slug, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
            if not args.continue_on_error:
                print(f"\nAborting after failure in '{slug}'. "
                      f"Use --continue-on-error to train the rest.")
                break
        print()

    total = time.perf_counter() - t0
    print("=" * 64)
    for slug, secs in ok:
        print(f"  ok    {slug:12s} {secs:7.1f}s")
    for slug, err in failed:
        print(f"  FAIL  {slug:12s} {err}")
    print(f"\n{len(ok)} succeeded, {len(failed)} failed in {total:.0f}s total.\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
