"""A structured, evidence-bearing CRISP-DM record.

CRISP-DM is usually rendered in portfolios as six decorative headings. Here it is
a data structure with teeth:

* every **finding** carries the measured number that supports it, so the phase
  narrative cannot drift from what the training run actually produced;
* every **decision** must name the alternative it rejected and why, which is the
  part of the methodology that actually transfers to real work;
* every phase has an explicit **exit gate** with pass/fail criteria, mirroring the
  phase-gate reviews CRISP-DM prescribes and most tutorials omit.

The training script builds these objects while it has the real numbers in hand and
serialises them next to the model, so the report and the model can never disagree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

PhaseKey = Literal[
    "business_understanding",
    "data_understanding",
    "data_preparation",
    "modeling",
    "evaluation",
    "deployment",
]

PHASE_TITLES: dict[str, str] = {
    "business_understanding": "1. Business Understanding",
    "data_understanding": "2. Data Understanding",
    "data_preparation": "3. Data Preparation",
    "modeling": "4. Modeling",
    "evaluation": "5. Evaluation",
    "deployment": "6. Deployment",
}

PHASE_PURPOSE: dict[str, str] = {
    "business_understanding": (
        "Translate a business question into a measurable data-mining goal and fix "
        "the success criteria before seeing any results."
    ),
    "data_understanding": (
        "Acquire the data, describe it honestly, and surface quality problems "
        "early enough that they can change the plan."
    ),
    "data_preparation": (
        "Construct the modelling dataset: selection, cleaning, feature engineering "
        "and the train/test protocol that makes evaluation trustworthy."
    ),
    "modeling": (
        "Select techniques, search their hyperparameters, and record what was tried "
        "-- including what failed."
    ),
    "evaluation": (
        "Assess results against the business criteria from phase 1, not just the "
        "loss function, and review the whole process for defects."
    ),
    "deployment": (
        "Ship the model behind an interface, with monitoring, documented limits, "
        "and a plan for the day the data drifts."
    ),
}


@dataclass
class Finding:
    """An observation backed by a number produced during the run."""

    statement: str
    evidence: str                      # the measured value, formatted for display
    implication: str = ""              # what it changed downstream
    severity: Literal["info", "watch", "critical"] = "info"


@dataclass
class Decision:
    """A methodological choice and the road not taken."""

    decision: str
    rationale: str
    alternative_rejected: str = ""
    rejection_reason: str = ""


@dataclass
class Gate:
    """Phase-exit criteria, evaluated rather than asserted."""

    criteria: list[str] = field(default_factory=list)
    passed: bool = True
    notes: str = ""


@dataclass
class Phase:
    key: str
    summary: str
    activities: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    gate: Gate = field(default_factory=Gate)

    @property
    def title(self) -> str:
        return PHASE_TITLES.get(self.key, self.key)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "purpose": PHASE_PURPOSE.get(self.key, ""),
            "summary": self.summary,
            "activities": list(self.activities),
            "findings": [asdict(f) for f in self.findings],
            "decisions": [asdict(d) for d in self.decisions],
            "metrics": dict(self.metrics),
            "artifacts": list(self.artifacts),
            "gate": asdict(self.gate),
        }


@dataclass
class CrispDmRecord:
    """The full six-phase record for one project."""

    project: str
    business_question: str
    success_criteria: list[str] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)
    iteration_notes: list[str] = field(default_factory=list)

    def phase(self, key: str) -> Phase | None:
        return next((p for p in self.phases if p.key == key), None)

    def to_dict(self) -> dict:
        ordered = [p for k in PHASE_TITLES for p in self.phases if p.key == k]
        gates_passed = sum(1 for p in ordered if p.gate.passed)
        return {
            "project": self.project,
            "business_question": self.business_question,
            "success_criteria": list(self.success_criteria),
            "phases": [p.to_dict() for p in ordered],
            "gates_passed": gates_passed,
            "gates_total": len(ordered),
            "all_gates_passed": gates_passed == len(ordered),
            "iteration_notes": list(self.iteration_notes),
            "methodology": (
                "CRISP-DM (Cross-Industry Standard Process for Data Mining), "
                "Chapman et al. 2000. Phases are iterative; the notes below record "
                "where this project looped back rather than proceeding linearly."
            ),
        }


def gate(passed: bool, *criteria: str, notes: str = "") -> Gate:
    """Terse constructor so training scripts stay readable."""
    return Gate(criteria=list(criteria), passed=bool(passed), notes=notes)
