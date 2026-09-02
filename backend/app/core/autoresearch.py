"""AutoResearch: a hill-climbing model tournament with greedy ensembling.

The reference portfolio this project replicates advertises "autoresearch hill
climbing" but ships a static leaderboard. This module implements the search for
real, and records every trial so the trajectory can be replayed in the UI.

The procedure
-------------
1. **Seed.** Each model family starts from a hand-chosen sane baseline.
2. **Coordinate ascent.** Repeatedly pick one hyperparameter, try its neighbours
   on the discrete grid, and move if cross-validated score improves. This is
   plain hill climbing: cheap, deterministic given a seed, and -- unlike random
   search -- it produces a *story* ("depth 6 -> 8 bought +0.004 PR-AUC") which is
   the point of showing it to a learner.
3. **Patience.** Stop a family after `patience` consecutive non-improving sweeps.
4. **Greedy ensembling.** Caruana et al. (2004) forward selection *with
   replacement* over the out-of-fold predictions of every family's best model.
   Selection runs on OOF predictions only; the test set is untouched until the
   final scoring call.

Anti-overfitting notes
----------------------
The search optimises a cross-validated score, never the hold-out score. The
hold-out set is scored exactly once per family at the end. Because hill climbing
with many trials can still overfit the CV folds, the leaderboard reports both the
CV score and the hold-out score side by side -- a large gap is the diagnostic,
and the audit flags it automatically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict, cross_val_score


# --------------------------------------------------------------------------- #
# search space
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    """One model family plus the discrete neighbourhood the climber may explore."""

    name: str
    family: str
    factory: Callable[[dict], Any]
    seed_params: dict
    grid: dict[str, list]
    # Human-facing rationale shown in the tournament UI.
    hypothesis: str = ""
    # Families that cannot benefit from search (e.g. plain OLS) skip the climb.
    searchable: bool = True

    def build(self, params: dict | None = None):
        return self.factory({**self.seed_params, **(params or {})})


@dataclass
class Trial:
    step: int
    family: str
    params: dict
    score: float
    accepted: bool
    changed: str | None
    elapsed_s: float
    note: str = ""


@dataclass
class FamilyResult:
    name: str
    family: str
    best_params: dict
    cv_score: float
    seed_score: float
    n_trials: int
    trials: list[Trial] = field(default_factory=list)
    hypothesis: str = ""
    holdout: dict | None = None
    fit_seconds: float = 0.0

    @property
    def improvement(self) -> float:
        return self.cv_score - self.seed_score


# --------------------------------------------------------------------------- #
# the climber
# --------------------------------------------------------------------------- #
class AutoResearch:
    """Coordinate-ascent hyperparameter search across several model families."""

    def __init__(self, *, scoring: str, cv, greater_is_better: bool = True,
                 max_trials_per_family: int = 26, patience: int = 2,
                 seed: int = 42, verbose: bool = True) -> None:
        self.scoring = scoring
        self.cv = cv
        self.sign = 1.0 if greater_is_better else -1.0
        self.max_trials = max_trials_per_family
        self.patience = patience
        self.seed = seed
        self.verbose = verbose
        self._trial_counter = 0

    # -- scoring ---------------------------------------------------------- #
    def _score(self, estimator, X, y) -> float:
        scores = cross_val_score(
            estimator, X, y, scoring=self.scoring, cv=self.cv, n_jobs=1,
            error_score="raise",
        )
        return float(np.mean(scores))

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"    {msg}", flush=True)

    # -- one family ------------------------------------------------------- #
    def climb(self, spec: ModelSpec, X, y) -> FamilyResult:
        t0 = time.perf_counter()
        trials: list[Trial] = []
        current = dict(spec.seed_params)

        base_score = self._score(spec.build(current), X, y)
        seed_score = base_score
        self._trial_counter += 1
        trials.append(Trial(self._trial_counter, spec.family, dict(current), base_score,
                            True, None, time.perf_counter() - t0, "seed configuration"))
        self._log(f"{spec.name:22s} seed   {self.scoring}={base_score:+.5f}")

        if spec.searchable and spec.grid:
            stale = 0
            keys = list(spec.grid)
            rng = np.random.default_rng(self.seed)
            while len(trials) < self.max_trials and stale < self.patience:
                improved_this_sweep = False
                # Deterministic-but-varied coordinate order per sweep.
                for key in rng.permutation(keys):
                    if len(trials) >= self.max_trials:
                        break
                    options = [v for v in spec.grid[key] if v != current.get(key)]
                    for value in options:
                        cand = {**current, key: value}
                        ts = time.perf_counter()
                        try:
                            score = self._score(spec.build(cand), X, y)
                        except Exception as exc:
                            self._trial_counter += 1
                            trials.append(Trial(self._trial_counter, spec.family, cand,
                                                float("nan"), False, str(key),
                                                time.perf_counter() - ts,
                                                f"rejected: {type(exc).__name__}"))
                            continue
                        accepted = self.sign * score > self.sign * base_score + 1e-9
                        self._trial_counter += 1
                        trials.append(Trial(
                            self._trial_counter, spec.family, dict(cand), score, accepted,
                            str(key), time.perf_counter() - ts,
                            f"{key}: {current.get(key)} -> {value}",
                        ))
                        if accepted:
                            delta = score - base_score
                            self._log(f"{spec.name:22s} accept {key}={value} "
                                      f"({self.scoring}={score:+.5f}, {delta:+.5f})")
                            base_score, current = score, cand
                            improved_this_sweep = True
                            break  # move to the next coordinate
                stale = 0 if improved_this_sweep else stale + 1

        return FamilyResult(
            name=spec.name, family=spec.family, best_params=current,
            cv_score=base_score, seed_score=seed_score, n_trials=len(trials),
            trials=trials, hypothesis=spec.hypothesis,
            fit_seconds=time.perf_counter() - t0,
        )

    # -- the tournament --------------------------------------------------- #
    def run(self, specs: Sequence[ModelSpec], X, y) -> list[FamilyResult]:
        results = []
        for spec in specs:
            self._log(f"-- climbing {spec.name}")
            results.append(self.climb(spec, X, y))
        results.sort(key=lambda r: -self.sign * r.cv_score)
        return results


# --------------------------------------------------------------------------- #
# Caruana greedy ensemble selection
# --------------------------------------------------------------------------- #
def greedy_ensemble(oof: dict[str, np.ndarray], y_true: np.ndarray,
                    score_fn: Callable[[np.ndarray, np.ndarray], float],
                    *, n_rounds: int = 25, greater_is_better: bool = True,
                    init_top: int = 1) -> dict:
    """Forward selection with replacement over out-of-fold predictions.

    Selection *with replacement* is what makes this work: a strong model can be
    picked repeatedly, which is how the procedure assigns it a larger weight
    without ever solving an optimisation problem that could overfit.
    """
    names = list(oof)
    if not names:
        return {"weights": {}, "score": None, "trajectory": []}
    sign = 1.0 if greater_is_better else -1.0

    solo = sorted(names, key=lambda n: -sign * score_fn(y_true, oof[n]))
    chosen = solo[:max(1, init_top)]
    trajectory = []

    def blend(members: list[str]) -> np.ndarray:
        return np.mean([oof[m] for m in members], axis=0)

    best = score_fn(y_true, blend(chosen))
    trajectory.append({"round": 0, "added": chosen[0], "score": float(best),
                       "members": list(chosen)})

    for r in range(1, n_rounds + 1):
        cand_scores = {}
        for n in names:
            cand_scores[n] = score_fn(y_true, blend(chosen + [n]))
        best_name = max(cand_scores, key=lambda n: sign * cand_scores[n])
        if sign * cand_scores[best_name] <= sign * best + 1e-10:
            break  # no addition helps; stop rather than pad the ensemble
        chosen.append(best_name)
        best = cand_scores[best_name]
        trajectory.append({"round": r, "added": best_name, "score": float(best),
                           "members": list(chosen)})

    counts = {n: chosen.count(n) for n in set(chosen)}
    total = sum(counts.values())
    weights = {n: round(c / total, 4) for n, c in sorted(counts.items(), key=lambda kv: -kv[1])}
    return {
        "weights": weights,
        "score": float(best),
        "n_rounds_used": len(trajectory) - 1,
        "trajectory": trajectory,
        "best_single": solo[0],
        "best_single_score": float(score_fn(y_true, oof[solo[0]])),
        "method": "Caruana et al. (2004) greedy forward selection with replacement",
    }


def build_oof(estimators: dict[str, Any], X, y, cv, *, method: str = "predict_proba",
              n_jobs: int = 1) -> dict[str, np.ndarray]:
    """Out-of-fold predictions for ensembling.

    Every prediction here is made by a model that never saw that row in training,
    which is the property that makes ensemble weight selection honest.
    """
    oof: dict[str, np.ndarray] = {}
    for name, est in estimators.items():
        try:
            pred = cross_val_predict(clone(est), X, y, cv=cv, method=method, n_jobs=n_jobs)
            if method == "predict_proba":
                pred = np.asarray(pred)[:, 1]
            oof[name] = np.asarray(pred, dtype=float)
        except Exception as exc:  # a family that cannot produce OOF is simply absent
            print(f"    [oof] skipped {name}: {type(exc).__name__}: {exc}", flush=True)
    return oof


# --------------------------------------------------------------------------- #
# serialisation for the API
# --------------------------------------------------------------------------- #
def leaderboard_payload(results: Sequence[FamilyResult], *, scoring: str,
                        ensemble: dict | None = None,
                        overfit_gap_threshold: float = 0.03) -> dict:
    """Turn tournament results into the payload the leaderboard UI consumes."""
    rows = []
    for rank, r in enumerate(results, start=1):
        hold = r.holdout or {}
        primary_holdout = hold.get("primary")
        gap = (r.cv_score - primary_holdout) if primary_holdout is not None else None
        rows.append({
            "rank": rank,
            "model": r.name,
            "family": r.family,
            "cv_score": r.cv_score,
            "seed_score": r.seed_score,
            "improvement_from_search": r.improvement,
            "holdout_score": primary_holdout,
            "generalisation_gap": gap,
            "overfit_flag": bool(gap is not None and gap > overfit_gap_threshold),
            "best_params": r.best_params,
            "n_trials": r.n_trials,
            "fit_seconds": round(r.fit_seconds, 2),
            "hypothesis": r.hypothesis,
            "holdout_metrics": hold.get("metrics"),
        })
    trajectory = [
        {"step": t.step, "family": t.family, "score": (None if np.isnan(t.score) else t.score),
         "accepted": t.accepted, "changed": t.changed, "note": t.note,
         "params": t.params, "elapsed_s": round(t.elapsed_s, 3)}
        for r in results for t in r.trials
    ]
    return {
        "scoring": scoring,
        "leaderboard": rows,
        "trajectory": trajectory,
        "total_trials": len(trajectory),
        "total_seconds": round(sum(r.fit_seconds for r in results), 1),
        "ensemble": ensemble,
        "protocol": (
            "Hyperparameters were selected by coordinate-ascent hill climbing on the "
            f"cross-validated {scoring}. The hold-out set was scored once per family "
            "after the search finished and never fed back into selection. "
            "'generalisation_gap' = CV minus hold-out; a large positive gap means the "
            "search overfitted the folds."
        ),
    }
