"""Generate the written record from the trained artifacts.

Every number in every document this script writes is read out of the artifacts a
training run produced. Nothing is transcribed by hand, which is the only way to
guarantee that the report and the model still agree six commits later. If a
retrain changes a metric, re-running this script updates the paper.

Produces, per project:

    docs/papers/<slug>/paper.md      technical write-up with the full protocol
    docs/papers/<slug>/abstract.md   one-page summary
    docs/papers/<slug>/article.md    the readable version

and updates the project table inside README.md between its marker comments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import registry  # noqa: E402
from app.core import paths  # noqa: E402
from app.core.artifacts import ArtifactStore  # noqa: E402

TABLE_START = "<!-- PROJECT-TABLE:START -->"
TABLE_END = "<!-- PROJECT-TABLE:END -->"


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def fmt(v, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        if abs(v) >= 10_000:
            return f"{v:,.0f}"
        return f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return str(v)


def human(key: str) -> str:
    return key.replace("_", " ").replace("pct", "%").strip().capitalize()


def headline_table(headline: dict) -> list[str]:
    rows = ["| Measure | Value |", "|---|---|"]
    for k, v in headline.items():
        rows.append(f"| {human(k)} | {fmt(v)} |")
    return rows


# --------------------------------------------------------------------------- #
# document builders
# --------------------------------------------------------------------------- #
def build_paper(meta, art: dict) -> str:
    overview = art["overview"]
    crisp = art["crispdm"]
    card = art["model_card"]
    audit = art["audit"]
    evaluation = art["evaluation"]
    leaderboard = art["leaderboard"]
    eda = art["eda"]
    prov = overview.get("_provenance", {})

    L: list[str] = []
    a = L.append

    a(f"# {meta.title}")
    a("")
    a(f"*{meta.tagline}*")
    a("")
    a(f"**Task** {meta.task}  ·  **Domain** {meta.domain}  ·  "
      f"**Primary metric** {meta.primary_metric}")
    a("")
    a(f"> Generated from the training run of `{prov.get('trained_at', 'unknown')}` "
      f"(commit `{prov.get('git_commit') or 'n/a'}`, seed {prov.get('seed', 'n/a')}). "
      f"Every figure below is read from that run's artifacts.")
    a("")

    a("## Abstract")
    a("")
    a(abstract_body(meta, art))
    a("")

    a("## 1. Business understanding")
    a("")
    a(f"**Question.** {crisp['business_question']}")
    a("")
    a("**Success criteria, fixed before modelling:**")
    a("")
    for c in crisp["success_criteria"]:
        a(f"- {c}")
    a("")
    a(f"**Why {meta.primary_metric}.** {meta.metric_reason}")
    a("")

    a("## 2. Data")
    a("")
    a(f"- **Source** — {meta.dataset_title}")
    a(f"- **Rows** — {fmt(meta.dataset_rows)} in the curated file"
      + (f"; {fmt(overview.get('rows_modelled'))} after preparation"
         if overview.get("rows_modelled") else ""))
    a(f"- **Licence** — {card.get('training_data', {}).get('licence', 'see data/MANIFEST.json')}")
    if eda.get("quality"):
        q = eda["quality"]
        a(f"- **Quality** — grade {q['grade']} at {q['overall_score']:.1%} across six dimensions")
    if eda.get("missingness"):
        m = eda["missingness"]
        a(f"- **Missingness** — {fmt(m['missing_cells'])} of {fmt(m['total_cells'])} cells "
          f"({m['missing_pct']:.2%}); {m['complete_rows_pct']:.1%} of rows complete")
    a("")
    a("The curation rule and a SHA-256 for this file are recorded in "
      "[`data/MANIFEST.json`](../../../data/MANIFEST.json). Ingestion applies only "
      "column selection, dtype coercion, deterministic subsampling and stable "
      "sorting — no imputation, scaling or target-aware filtering.")
    a("")

    for key, number in (("data_understanding", "3"), ("data_preparation", "4")):
        phase = next((p for p in crisp["phases"] if p["key"] == key), None)
        if not phase:
            continue
        a(f"## {number}. {phase['title'].split('. ', 1)[-1]}")
        a("")
        a(phase["summary"])
        a("")
        if phase["findings"]:
            a("**Findings.**")
            a("")
            for f in phase["findings"]:
                a(f"- *{f['statement']}* — {f['evidence']}."
                  + (f" {f['implication']}" if f["implication"] else ""))
            a("")
        if phase["decisions"]:
            a("**Decisions.**")
            a("")
            for d in phase["decisions"]:
                a(f"- **{d['decision']}** {d['rationale']}"
                  + (f" *Rejected:* {d['alternative_rejected']} — {d['rejection_reason']}"
                     if d["alternative_rejected"] else ""))
            a("")

    a("## 5. Modelling")
    a("")
    modelling = next((p for p in crisp["phases"] if p["key"] == "modeling"), None)
    if modelling:
        a(modelling["summary"])
        a("")
    rows = leaderboard.get("leaderboard", [])
    if rows:
        a(f"**Leaderboard** — scored by {leaderboard.get('scoring', 'the primary metric')}.")
        a("")
        cols = _leaderboard_columns(rows)
        a("| " + " | ".join(c[0] for c in cols) + " |")
        a("|" + "|".join("---" for _ in cols) + "|")
        for r in rows:
            a("| " + " | ".join(fmt(c[1](r)) for c in cols) + " |")
        a("")
    if leaderboard.get("protocol") and isinstance(leaderboard["protocol"], str):
        a(f"> {leaderboard['protocol']}")
        a("")

    a("## 6. Evaluation")
    a("")
    evaluation_phase = next((p for p in crisp["phases"] if p["key"] == "evaluation"), None)
    if evaluation_phase:
        a(evaluation_phase["summary"])
        a("")
    a("**Headline results**")
    a("")
    L.extend(headline_table(overview.get("headline", {})))
    a("")
    baselines = evaluation.get("baselines")
    if isinstance(baselines, list) and baselines:
        a("**Baselines the model had to beat**")
        a("")
        a("| Baseline | Score | What it does |")
        a("|---|---|---|")
        for b in baselines:
            score = b.get("mae") or b.get("pr_auc") or b.get("roc_auc")
            a(f"| {b.get('name')} | {fmt(score)} | {b.get('description', '')} |")
        a("")

    a("## 7. Deployment")
    a("")
    deployment = next((p for p in crisp["phases"] if p["key"] == "deployment"), None)
    if deployment:
        a(deployment["summary"])
        a("")
    a(f"Served at `POST /api/projects/{meta.slug}/predict`. The fitted pipeline is "
      "pickled whole, so preprocessing travels with the model and training and "
      "serving cannot drift apart.")
    a("")

    a("## 8. Limitations")
    a("")
    for lim in card.get("limitations", []):
        a(f"- {lim}")
    a("")
    a("**Out of scope**")
    a("")
    for s in card.get("out_of_scope", []):
        a(f"- {s}")
    a("")
    if card.get("ethical_considerations"):
        a("**Ethical considerations**")
        a("")
        for e in card["ethical_considerations"]:
            a(f"- {e}")
        a("")

    a("## 9. Audit")
    a("")
    a(f"{audit['passed']} of {audit['total']} checks pass (grade {audit['grade']}). "
      f"*Scope:* {audit['scope']}")
    a("")
    a("| Check | Status | Evidence |")
    a("|---|---|---|")
    for c in audit["checks"]:
        ev = c["evidence"].replace("|", "\\|").replace("\n", " ")
        a(f"| {c['check']} | {c['status']} | {ev} |")
    a("")

    gates_passed = crisp["gates_passed"]
    gates_total = crisp["gates_total"]
    a(f"**Phase gates:** {gates_passed} of {gates_total} passed.")
    a("")
    for phase in crisp["phases"]:
        mark = "PASS" if phase["gate"]["passed"] else "**FAIL**"
        a(f"- {mark} — {phase['title']}"
          + (f": {phase['gate']['notes']}" if phase["gate"]["notes"] else ""))
    a("")

    if crisp.get("iteration_notes"):
        a("## 10. Where this project looped back")
        a("")
        a("CRISP-DM is iterative. These are the points where a later phase sent the "
          "work back to an earlier one — normally the part deleted before publication.")
        a("")
        for n in crisp["iteration_notes"]:
            a(f"- {n}")
        a("")

    a("## Reproduction")
    a("")
    a("```bash")
    a("python scripts/fetch_data.py")
    a(f"python scripts/train_all.py --only {meta.slug}")
    a("```")
    a("")
    a("Every estimator, split and sampler is seeded, so a rerun on the same data "
      "reproduces this leaderboard exactly.")
    a("")
    return "\n".join(L)


def _leaderboard_columns(rows: list[dict]):
    """Pick the columns that are actually populated for this project."""
    def has(k):
        return any(r.get(k) is not None for r in rows)

    cols = [("Model", lambda r: r.get("model") or r.get("algorithm"))]
    for key, label in [
        ("cv_score", "CV"),
        ("cv_mase_mean", "CV MASE"),
        ("holdout_score", "Hold-out"),
        ("holdout_mase", "Hold-out MASE"),
        ("generalisation_gap", "Gap"),
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
        ("silhouette", "Silhouette"),
        ("val_bpc", "Bits/char"),
        ("stability_ari", "Bootstrap ARI"),
        ("seconds", "Seconds"),
        ("n_trials", "Trials"),
    ]:
        if has(key):
            cols.append((label, (lambda k: (lambda r: r.get(k)))(key)))
    return cols


def abstract_body(meta, art: dict) -> str:
    overview = art["overview"]
    crisp = art["crispdm"]
    audit = art["audit"]
    headline = overview.get("headline", {})
    top = ", ".join(
        f"{human(k).lower()} {fmt(v)}" for k, v in list(headline.items())[:4]
    )
    evaluation_phase = next((p for p in crisp["phases"] if p["key"] == "evaluation"), None)
    verdict = evaluation_phase["summary"] if evaluation_phase else ""

    return (
        f"{meta.tagline} The system follows the full CRISP-DM cycle on "
        f"{fmt(meta.dataset_rows)} rows of {meta.dataset_title}, selecting "
        f"{overview.get('champion', 'a model')} by "
        f"{'hill-climbing search' if art['leaderboard'].get('trajectory') else 'comparative evaluation'} "
        f"on cross-validated scores with the hold-out reserved for a single final "
        f"measurement. Headline results: {top}. {verdict} "
        f"A static leakage audit of the training path passes {audit['passed']} of "
        f"{audit['total']} checks, and {crisp['gates_passed']} of "
        f"{crisp['gates_total']} CRISP-DM phase gates are met."
    )


def build_abstract(meta, art: dict) -> str:
    overview = art["overview"]
    card = art["model_card"]
    L = [
        f"# {meta.title} — Abstract",
        "",
        f"**Task** {meta.task}  ·  **Data** {meta.dataset_title}  ·  "
        f"**Metric** {meta.primary_metric}",
        "",
        abstract_body(meta, art),
        "",
        "## Results",
        "",
    ]
    L.extend(headline_table(overview.get("headline", {})))
    L += ["", "## What this model must not be used for", ""]
    L += [f"- {s}" for s in card.get("out_of_scope", [])]
    L += ["", "## Known limitations", ""]
    L += [f"- {s}" for s in card.get("limitations", [])]
    L += ["", f"Full write-up: [`paper.md`](./paper.md).", ""]
    return "\n".join(L)


def build_article(meta, art: dict) -> str:
    overview = art["overview"]
    crisp = art["crispdm"]
    card = art["model_card"]
    evaluation = art["evaluation"]

    interesting = _interesting_finding(crisp)
    loop = crisp.get("iteration_notes", [])

    L = [
        f"# {meta.title}",
        "",
        f"*{meta.tagline}*",
        "",
        "## The question",
        "",
        crisp["business_question"],
        "",
        "## Why this metric and not accuracy",
        "",
        meta.metric_reason,
        "",
        "## What the data actually looked like",
        "",
    ]
    du = next((p for p in crisp["phases"] if p["key"] == "data_understanding"), None)
    if du:
        L += [du["summary"], ""]
        for f in du["findings"][:3]:
            L.append(f"**{f['statement']}** — {f['evidence']}. {f['implication']}")
            L.append("")

    L += ["## The modelling", ""]
    modelling = next((p for p in crisp["phases"] if p["key"] == "modeling"), None)
    if modelling:
        L += [modelling["summary"], ""]
        for d in modelling["decisions"][:2]:
            L.append(f"> **{d['decision']}** {d['rationale']}")
            if d.get("alternative_rejected"):
                L.append(f">")
                L.append(f"> *We rejected {d['alternative_rejected']}:* {d['rejection_reason']}")
            L.append("")

    L += ["## Results", ""]
    L.extend(headline_table(overview.get("headline", {})))
    L.append("")

    if interesting:
        L += ["## The finding worth reading twice", "",
              f"**{interesting['statement']}**", "",
              f"{interesting['evidence']}. {interesting['implication']}", ""]

    if loop:
        L += ["## What went wrong first", "",
              "Every project here records where a later phase sent the work back to an "
              "earlier one. That is normally the part deleted before publication, and "
              "it is usually the most useful part.", ""]
        for n in loop:
            L += [f"- {n}"]
        L.append("")

    L += ["## What it cannot do", ""]
    L += [f"- {s}" for s in card.get("out_of_scope", [])[:3]]
    L += ["", "## Try it", "",
          f"The live inference playground for this project is on the **Live inference** "
          f"tab of the console, or call it directly:", "",
          "```bash",
          f"curl -X POST localhost:8000/api/projects/{meta.slug}/predict \\",
          "  -H 'Content-Type: application/json' -d '{}'",
          "```", ""]
    return "\n".join(L)


def _interesting_finding(crisp: dict) -> dict | None:
    """The most severe finding across all phases -- usually the real story."""
    best = None
    rank = {"critical": 2, "watch": 1, "info": 0}
    for phase in crisp["phases"]:
        for f in phase["findings"]:
            if best is None or rank.get(f["severity"], 0) > rank.get(best["severity"], 0):
                best = f
    return best


# --------------------------------------------------------------------------- #
def readme_table() -> str:
    lines = [
        "| # | Project | Data | Task | Headline result |",
        "|---|---|---|---|---|",
    ]
    for meta in registry.PROJECTS:
        store = ArtifactStore(meta.slug)
        headline = ""
        if store.is_trained:
            h = (store.payload("overview") or {}).get("headline", {})
            parts = [f"{human(k).lower()} **{fmt(v)}**" for k, v in list(h.items())[:2]]
            headline = ", ".join(parts)
        else:
            headline = "*not trained*"
        rows = f"{meta.dataset_rows:,}"
        lines.append(
            f"| {meta.number:02d} "
            f"| [{meta.title}](docs/papers/{meta.slug}/paper.md)<br/>"
            f"<sub>{meta.tagline}</sub> "
            f"| {meta.dataset_title.split('(')[0].strip()}<br/><sub>{rows} rows</sub> "
            f"| {meta.task} "
            f"| {headline} |"
        )
    return "\n".join(lines)


def update_readme() -> bool:
    path = ROOT / "README.md"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if TABLE_START not in text or TABLE_END not in text:
        return False
    head, rest = text.split(TABLE_START, 1)
    _, tail = rest.split(TABLE_END, 1)
    path.write_text(
        f"{head}{TABLE_START}\n{readme_table()}\n{TABLE_END}{tail}", encoding="utf-8"
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    slugs = args.only or registry.SLUGS
    written = 0
    skipped = []

    for slug in slugs:
        meta = registry.get(slug)
        store = ArtifactStore(slug)
        if not store.is_trained:
            skipped.append(slug)
            continue
        art = {
            name: store.payload(name) or {}
            for name in ("overview", "crispdm", "evaluation", "leaderboard",
                         "model_card", "audit", "eda")
        }
        out = paths.PAPERS / slug
        out.mkdir(parents=True, exist_ok=True)
        (out / "paper.md").write_text(build_paper(meta, art), encoding="utf-8")
        (out / "abstract.md").write_text(build_abstract(meta, art), encoding="utf-8")
        (out / "article.md").write_text(build_article(meta, art), encoding="utf-8")
        written += 3
        print(f"  {slug:12s} -> docs/papers/{slug}/{{paper,abstract,article}}.md")

    if update_readme():
        print("  README.md project table refreshed")

    if skipped:
        print(f"\n  skipped (not trained): {', '.join(skipped)}")
    print(f"\n{written} documents written from artifacts.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
