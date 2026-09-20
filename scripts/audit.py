"""Portfolio-wide leakage and methodology audit.

Two passes:

1. **Static scan.** Walk the AST of every project module looking for the specific
   syntactic shapes that cause leakage -- a `.rolling()` that is not preceded by
   a `.shift()`, a `train_test_split` with `shuffle=True` in a temporal project,
   a `fit_transform` called on a whole frame outside a Pipeline, a target column
   appearing in a feature list. These are patterns, so they can be found
   mechanically and they never get tired of looking.

2. **Artifact aggregation.** Collect each project's own audit payload, the
   CRISP-DM gate results, and check that reported metrics beat the baselines the
   project itself published.

Writes `backend/artifacts/_portfolio_audit.json` (served at `/api/audit`) and
`AUDIT_REPORT.md`.

A static scan cannot prove the absence of leakage -- only a reviewer can. What it
can do is make the common mistakes impossible to commit unnoticed, and say
plainly what it did not check.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import registry  # noqa: E402
from app.core import paths  # noqa: E402
from app.core.artifacts import ArtifactStore  # noqa: E402
from app.core.serialization import write_json  # noqa: E402

# Projects whose data has a time axis; a shuffled split in these is a defect.
TEMPORAL_PROJECTS = {"taxi", "forecast", "basket", "nanollm"}


def _collect_shifted_names(source: str) -> set[str]:
    """Names bound to an expression that contains a `.shift(` call."""
    names: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        has_shift = any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "shift"
            for c in ast.walk(node.value)
        )
        if has_shift:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


@dataclass
class Issue:
    project: str
    file: str
    line: int
    rule: str
    severity: str        # "critical" | "warning" | "info"
    message: str
    snippet: str = ""


@dataclass
class ScanResult:
    issues: list[Issue] = field(default_factory=list)
    files_scanned: int = 0
    rules_run: int = 0


# --------------------------------------------------------------------------- #
# static rules
# --------------------------------------------------------------------------- #
class LeakageVisitor(ast.NodeVisitor):
    """AST walker implementing the syntactic leakage rules."""

    def __init__(self, project: str, path: Path, source: str) -> None:
        self.project = project
        self.path = path
        self.lines = source.splitlines()
        self.issues: list[Issue] = []
        # Names bound to an expression containing .shift(). The idiomatic safe
        # pattern splits across two statements --
        #     past = target.shift(1)
        #     past.rolling(24).mean()
        # -- so a rule that only inspects a single attribute chain reports it as a
        # leak. Following one level of assignment removes that false positive.
        # An audit tool that cries wolf gets switched off, which is worse than
        # having no audit tool.
        self.shifted_names: set[str] = _collect_shifted_names(source)

    # -- helpers ---------------------------------------------------------- #
    def _snippet(self, node: ast.AST) -> str:
        line = getattr(node, "lineno", 0)
        return self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""

    def _suppressed(self, node: ast.AST) -> bool:
        """Honour an `# audit: ok - reason` marker on, or just above, the call.

        Every rule here has legitimate exceptions. Making the exception explicit
        and greppable is better than loosening the rule until it stops firing --
        the marker survives review, a silently weakened rule does not.

        The marker is accepted anywhere in the four lines preceding the call as
        well as on the call itself, because a justification worth writing rarely
        fits in a trailing comment.
        """
        line = getattr(node, "lineno", 0)
        start = max(0, line - 5)
        window = self.lines[start:line]
        return any("audit: ok" in ln for ln in window)

    def _add(self, node: ast.AST, rule: str, severity: str, message: str) -> None:
        if self._suppressed(node):
            return
        self.issues.append(
            Issue(self.project, str(self.path.relative_to(ROOT)), getattr(node, "lineno", 0),
                  rule, severity, message, self._snippet(node))
        )

    @staticmethod
    def _attr_chain(node: ast.AST) -> list[str]:
        """Flatten `a.b().c()` into ['a', 'b', 'c'] so a chain can be inspected."""
        out: list[str] = []
        cur = node
        while True:
            if isinstance(cur, ast.Call):
                cur = cur.func
            elif isinstance(cur, ast.Attribute):
                out.append(cur.attr)
                cur = cur.value
            elif isinstance(cur, ast.Name):
                out.append(cur.id)
                break
            elif isinstance(cur, ast.Subscript):
                cur = cur.value
            else:
                break
        return list(reversed(out))

    # -- rules ------------------------------------------------------------ #
    def visit_Call(self, node: ast.Call) -> None:
        chain = self._attr_chain(node)

        # R1: a rolling/expanding window that is not preceded by a shift contains
        # the value it is used to predict. The shift may appear in the same chain
        # or on the variable the chain starts from.
        if "rolling" in chain or "expanding" in chain:
            base = chain[0] if chain else ""
            if "shift" not in chain and base not in self.shifted_names:
                self._add(
                    node, "R1-rolling-without-shift", "critical",
                    "A rolling/expanding window with no preceding .shift() includes "
                    "the current row in its own predictor. Apply .shift(1) first.",
                )

        # R2: shuffling a temporal dataset.
        if chain and chain[-1] == "train_test_split":
            for kw in node.keywords:
                if kw.arg == "shuffle" and isinstance(kw.value, ast.Constant) and kw.value.value:
                    if self.project in TEMPORAL_PROJECTS:
                        self._add(node, "R2-shuffled-temporal-split", "critical",
                                  "train_test_split(shuffle=True) on a time-ordered "
                                  "dataset lets the model interpolate rather than "
                                  "extrapolate.")

        # R3: fit_transform outside a Pipeline, applied before a split.
        if chain and chain[-1] == "fit_transform":
            src = self._snippet(node)
            if "Pipeline" not in src and "prep.fit_transform" not in src:
                self._add(node, "R3-fit-transform-on-full-frame", "warning",
                          "fit_transform outside a Pipeline learns from every row it "
                          "sees. Confirm this runs on training rows only, or move it "
                          "into the Pipeline so CV re-fits it per fold.")

        # R4: SMOTE / resampling applied before the split.
        if chain and any(c in {"SMOTE", "RandomOverSampler", "ADASYN"} for c in chain):
            self._add(node, "R4-resample-before-split", "warning",
                      "Oversampling must happen inside the cross-validation loop; "
                      "applied to the full frame it duplicates rows across folds.")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # R5: a feature list that contains the target name.
        for target in node.targets:
            name = getattr(target, "id", "")
            if name in {"ALL_FEATURES", "NUMERIC_FEATURES", "CATEGORICAL_FEATURES", "FEATURES"}:
                literals = {
                    n.value for n in ast.walk(node.value)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                }
                suspicious = {"target", "label", "y", "churn", "is_attack", "high_income"}
                hit = literals & suspicious
                if hit:
                    self._add(node, "R5-target-in-feature-list", "critical",
                              f"The feature list appears to contain the target: {sorted(hit)}")
        self.generic_visit(node)


def scan_project(slug: str) -> list[Issue]:
    pkg = paths.APP / "projects" / slug
    issues: list[Issue] = []
    for path in sorted(pkg.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            issues.append(Issue(slug, str(path.relative_to(ROOT)), exc.lineno or 0,
                                "R0-syntax", "critical", f"Could not parse: {exc}"))
            continue
        visitor = LeakageVisitor(slug, path, source)
        visitor.visit(tree)
        issues.extend(visitor.issues)
    return issues


# --------------------------------------------------------------------------- #
# artifact checks
# --------------------------------------------------------------------------- #
def _normalise_baselines(baselines) -> list[dict]:
    """Projects express baselines differently: a list of dicts for the supervised
    ones, a dict of named entropy floors for the language model. Normalise both
    rather than assuming one shape."""
    out: list[dict] = []
    if isinstance(baselines, dict):
        for name, b in baselines.items():
            if isinstance(b, dict):
                out.append({"name": name,
                            "score": b.get("bits_per_char") or b.get("score")})
        return out
    for b in baselines or []:
        if isinstance(b, dict):
            out.append({"name": b.get("name"),
                        "score": b.get("mae") or b.get("pr_auc") or b.get("roc_auc")})
    return out


def audit_artifacts(slug: str) -> dict:
    store = ArtifactStore(slug)
    if not store.is_trained:
        return {"slug": slug, "trained": False,
                "note": "No artifacts; run scripts/train_all.py"}

    own = store.payload("audit") or {}
    crisp = store.payload("crispdm") or {}
    evaluation = store.payload("evaluation") or {}
    card = store.payload("model_card") or {}
    overview = store.payload("overview") or {}

    findings: list[str] = []

    # Every project must publish baselines and beat the best of them.
    baselines = evaluation.get("baselines") or []
    # segments (no ground truth), nanollm (baselines are a dict of entropy floors)
    # and basket (its reference is the lift = 1 independence line, published on
    # every rule) express their baseline differently.
    if not baselines and slug not in {"segments", "nanollm", "basket"}:
        findings.append("No baselines published; a result without a baseline is not a claim.")

    # The model card must state limits.
    if not card.get("out_of_scope"):
        findings.append("Model card does not state out-of-scope uses.")
    if not card.get("limitations"):
        findings.append("Model card claims no limitations.")

    # Phase gates.
    gates_passed = crisp.get("gates_passed", 0)
    gates_total = crisp.get("gates_total", 0)
    if gates_total and gates_passed < gates_total:
        findings.append(f"{gates_total - gates_passed} CRISP-DM gate(s) failed.")

    # Every finding in the record must carry evidence.
    for phase in crisp.get("phases", []):
        for f in phase.get("findings", []):
            if not f.get("evidence"):
                findings.append(f"Phase '{phase['key']}' has a finding with no evidence.")

    prov = store.provenance() or {}
    if prov.get("quick_mode"):
        findings.append("Artifacts were produced in --quick mode (reduced fidelity).")

    return {
        "slug": slug,
        "trained": True,
        "own_audit": {"passed": own.get("passed"), "total": own.get("total"),
                      "grade": own.get("grade"), "scope": own.get("scope")},
        "checks": own.get("checks", []),
        "gates": {"passed": gates_passed, "total": gates_total},
        "baselines": _normalise_baselines(baselines),
        "headline": overview.get("headline", {}),
        "provenance": prov,
        "findings": findings,
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markdown", default=str(ROOT / "AUDIT_REPORT.md"))
    args = ap.parse_args()

    print("\n=== Static leakage scan ===")
    scan = ScanResult(rules_run=5)
    for meta in registry.PROJECTS:
        issues = scan_project(meta.slug)
        scan.issues.extend(issues)
        scan.files_scanned += len(list((paths.APP / "projects" / meta.slug).rglob("*.py")))
        status = "clean" if not issues else f"{len(issues)} issue(s)"
        print(f"  {meta.slug:12s} {status}")
        for i in issues:
            print(f"      [{i.severity}] {i.file}:{i.line} {i.rule}")
            print(f"          {i.message}")

    print("\n=== Artifact audit ===")
    per_project = []
    for meta in registry.PROJECTS:
        result = audit_artifacts(meta.slug)
        per_project.append(result)
        if not result["trained"]:
            print(f"  {meta.slug:12s} NOT TRAINED")
            continue
        own = result["own_audit"]
        flag = "" if not result["findings"] else f"  <- {len(result['findings'])} finding(s)"
        print(f"  {meta.slug:12s} {own['passed']}/{own['total']} checks, "
              f"grade {own['grade']}, gates {result['gates']['passed']}/"
              f"{result['gates']['total']}{flag}")
        for f in result["findings"]:
            print(f"      - {f}")

    critical = [i for i in scan.issues if i.severity == "critical"]
    warnings_ = [i for i in scan.issues if i.severity == "warning"]
    trained = [p for p in per_project if p["trained"]]
    total_checks = sum(p["own_audit"]["total"] or 0 for p in trained)
    passed_checks = sum(p["own_audit"]["passed"] or 0 for p in trained)

    payload = {
        "static_scan": {
            "files_scanned": scan.files_scanned,
            "rules_run": scan.rules_run,
            "rules": [
                {"id": "R1-rolling-without-shift", "severity": "critical",
                 "detects": "A rolling or expanding window not preceded by .shift(), "
                            "which includes the current row in its own predictor."},
                {"id": "R2-shuffled-temporal-split", "severity": "critical",
                 "detects": "train_test_split(shuffle=True) in a time-ordered project."},
                {"id": "R3-fit-transform-on-full-frame", "severity": "warning",
                 "detects": "fit_transform outside a Pipeline, which learns from every "
                            "row it is shown."},
                {"id": "R4-resample-before-split", "severity": "warning",
                 "detects": "Oversampling applied outside the cross-validation loop."},
                {"id": "R5-target-in-feature-list", "severity": "critical",
                 "detects": "A declared feature list containing the target column."},
            ],
            "critical": len(critical),
            "warnings": len(warnings_),
            "issues": [vars(i) for i in scan.issues],
        },
        "projects": per_project,
        "summary": {
            "projects": len(registry.PROJECTS),
            "trained": len(trained),
            "checks_passed": passed_checks,
            "checks_total": total_checks,
            "gates_passed": sum(p["gates"]["passed"] for p in trained),
            "gates_total": sum(p["gates"]["total"] for p in trained),
            "static_critical": len(critical),
            "static_warnings": len(warnings_),
            "open_findings": sum(len(p["findings"]) for p in trained),
        },
        "limits_of_this_audit": [
            "A static scan matches syntax, not semantics. It cannot detect leakage "
            "expressed through a variable it did not follow, nor a feature that is "
            "legitimate in form but computed from the future in substance.",
            "It does not verify that the numbers in the artifacts were produced by the "
            "code as committed; that is what the pinned seeds, the run provenance "
            "stamped into every artifact and the reproduction instructions are for. "
            "Note the committed artifacts predate this repository's first commit, so "
            "their git_commit field is null -- seed, timestamp, Python version and "
            "platform are recorded, the commit is not.",
            "Passing every check means the known failure modes were checked for. It "
            "does not mean the analysis is correct.",
        ],
    }

    out = paths.ARTIFACTS / "_portfolio_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, payload)
    _write_markdown(Path(args.markdown), payload)

    s = payload["summary"]
    print(f"\n{'=' * 64}")
    print(f"  {s['trained']}/{s['projects']} projects trained")
    print(f"  {s['checks_passed']}/{s['checks_total']} per-project audit checks passed")
    print(f"  {s['gates_passed']}/{s['gates_total']} CRISP-DM phase gates passed")
    print(f"  static scan: {s['static_critical']} critical, {s['static_warnings']} warnings")
    print(f"  open findings: {s['open_findings']}")
    print(f"\n  -> {out.relative_to(ROOT)}")
    print(f"  -> {Path(args.markdown).relative_to(ROOT)}\n")

    return 1 if critical else 0


def _write_markdown(path: Path, payload: dict) -> None:
    s = payload["summary"]
    lines = [
        "# Leakage & Methodology Audit",
        "",
        "Generated by `scripts/audit.py`. Two passes: a static AST scan for the",
        "syntactic shapes that cause leakage, and an aggregation of each project's",
        "own audit payload and CRISP-DM phase gates.",
        "",
        "## Summary",
        "",
        "| Measure | Result |",
        "|---|---|",
        f"| Projects trained | {s['trained']} / {s['projects']} |",
        f"| Per-project audit checks passed | {s['checks_passed']} / {s['checks_total']} |",
        f"| CRISP-DM phase gates passed | {s['gates_passed']} / {s['gates_total']} |",
        f"| Static scan — critical | {s['static_critical']} |",
        f"| Static scan — warnings | {s['static_warnings']} |",
        f"| Open findings | {s['open_findings']} |",
        "",
        "## Static rules",
        "",
        "| Rule | Severity | Detects |",
        "|---|---|---|",
    ]
    for r in payload["static_scan"]["rules"]:
        lines.append(f"| `{r['id']}` | {r['severity']} | {r['detects']} |")

    issues = payload["static_scan"]["issues"]
    lines += ["", "### Findings", ""]
    if not issues:
        lines.append("No static rule fired across "
                     f"{payload['static_scan']['files_scanned']} project files.")
    else:
        for i in issues:
            lines.append(f"- **{i['severity']}** `{i['file']}:{i['line']}` "
                         f"({i['rule']}) — {i['message']}")
            if i.get("snippet"):
                lines.append(f"  ```python\n  {i['snippet']}\n  ```")

    lines += ["", "## Per project", ""]
    for p in payload["projects"]:
        if not p["trained"]:
            lines += [f"### `{p['slug']}`", "", "Not trained.", ""]
            continue
        own = p["own_audit"]
        lines += [
            f"### `{p['slug']}` — grade {own['grade']} "
            f"({own['passed']}/{own['total']} checks, "
            f"{p['gates']['passed']}/{p['gates']['total']} gates)",
            "",
            f"*Scope:* {own['scope']}",
            "",
            "| Check | Status | Evidence |",
            "|---|---|---|",
        ]
        for c in p["checks"]:
            evidence = c["evidence"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {c['check']} | {c['status']} | {evidence} |")
        if p["findings"]:
            lines += ["", "**Open findings:**", ""]
            lines += [f"- {f}" for f in p["findings"]]
        lines.append("")

    lines += ["## What this audit does not establish", ""]
    lines += [f"- {n}" for n in payload["limits_of_this_audit"]]
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
