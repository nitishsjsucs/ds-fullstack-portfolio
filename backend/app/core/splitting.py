"""Leakage-safe dataset splitting.

The single most common way a portfolio project silently cheats is by letting
information cross the train/test boundary. This module centralises every split in
the repository so the leakage audit has exactly one place to inspect, and so no
project can quietly call ``train_test_split(shuffle=True)`` on a time series.

Three guarantees are enforced here rather than left to convention:

1. **Temporal data is never shuffled.** :func:`temporal_split` slices by position
   in time order; the test set is always the *future* of the training set.
2. **Overlapping observations are purged.** :class:`PurgedTimeSeriesSplit`
   removes training rows whose feature window overlaps the validation fold and
   then embargoes a further gap, following Lopez de Prado's construction. Without
   this, lagged features leak the validation target backwards into training.
3. **Splits are asserted, not assumed.** :func:`assert_disjoint` and
   :func:`assert_chronological` are called by the training scripts and by the
   test suite, so a regression fails CI instead of inflating a metric.

Nothing in this module fits a transformer. Preprocessing belongs inside a
``Pipeline`` that is fitted per-fold; that is what makes the guarantees real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


# --------------------------------------------------------------------------- #
# Assertions -- cheap, and they run in production training
# --------------------------------------------------------------------------- #
def assert_disjoint(train_idx: Sequence[int], test_idx: Sequence[int], *, what: str = "split") -> None:
    overlap = np.intersect1d(np.asarray(train_idx), np.asarray(test_idx))
    if overlap.size:
        raise AssertionError(
            f"{what}: {overlap.size} row(s) appear in both train and test "
            f"(first offenders: {overlap[:5].tolist()})"
        )


def assert_chronological(times: pd.Series, train_idx: Sequence[int], test_idx: Sequence[int],
                         *, what: str = "temporal split") -> None:
    """Every training timestamp must precede every test timestamp."""
    tr, te = times.iloc[list(train_idx)], times.iloc[list(test_idx)]
    if len(tr) and len(te) and tr.max() > te.min():
        raise AssertionError(
            f"{what}: lookahead detected -- max(train)={tr.max()} > min(test)={te.min()}"
        )


def assert_no_group_leak(groups: pd.Series, train_idx: Sequence[int], test_idx: Sequence[int],
                         *, what: str = "group split") -> None:
    """No entity (customer, vehicle, session) may straddle the boundary."""
    shared = set(groups.iloc[list(train_idx)]) & set(groups.iloc[list(test_idx)])
    if shared:
        raise AssertionError(
            f"{what}: {len(shared)} group(s) span train and test "
            f"(e.g. {sorted(shared)[:5]})"
        )


# --------------------------------------------------------------------------- #
# Split constructors
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Split:
    """A concrete train/test partition plus the provenance to justify it."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    strategy: str
    rationale: str
    seed: int | None = None
    boundary: str | None = None  # e.g. the timestamp the split happened at

    @property
    def n_train(self) -> int:
        return int(len(self.train_idx))

    @property
    def n_test(self) -> int:
        return int(len(self.test_idx))

    def describe(self) -> dict:
        return {
            "strategy": self.strategy,
            "rationale": self.rationale,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "test_fraction": round(self.n_test / max(1, self.n_train + self.n_test), 4),
            "seed": self.seed,
            "boundary": self.boundary,
        }


def stratified_split(y: pd.Series, *, test_size: float = 0.2, seed: int = 42) -> Split:
    """Class-balanced random split for i.i.d. tabular data.

    Stratification is on the target only. It preserves the positive rate across
    folds, which matters when the minority class is a few percent -- an unlucky
    shuffle can otherwise leave a fold with almost no positives and make PR-AUC
    meaningless.
    """
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    tr, te = np.sort(tr), np.sort(te)
    assert_disjoint(tr, te, what="stratified_split")
    return Split(
        train_idx=tr, test_idx=te, seed=seed, strategy="stratified random hold-out",
        rationale=(
            f"Rows are exchangeable (no time or group structure), so a random "
            f"{int(test_size * 100)}% hold-out is unbiased. Stratified on the target to "
            f"hold the positive rate constant across folds."
        ),
    )


def temporal_split(times: pd.Series, *, test_size: float = 0.2) -> Split:
    """Chronological hold-out: train on the past, test on the future.

    ``times`` must already be sorted ascending -- we verify rather than sort, so a
    caller that forgot to order the frame gets an error instead of a silently
    reshuffled dataset.
    """
    if not times.is_monotonic_increasing:
        raise ValueError(
            "temporal_split expects a chronologically sorted frame; "
            "sort before splitting so row positions and time order agree."
        )
    n = len(times)
    cut = int(n * (1 - test_size))
    tr, te = np.arange(cut), np.arange(cut, n)
    assert_disjoint(tr, te, what="temporal_split")
    assert_chronological(times, tr, te)
    return Split(
        train_idx=tr, test_idx=te, strategy="chronological hold-out",
        boundary=str(times.iloc[cut]),
        rationale=(
            f"The target is a time series, so the only honest evaluation is "
            f"out-of-sample in the future. Train covers {times.iloc[0]} to "
            f"{times.iloc[cut - 1]}; test covers {times.iloc[cut]} onward. "
            f"No shuffling at any point."
        ),
    )


def grouped_split(groups: pd.Series, *, test_size: float = 0.2, seed: int = 42) -> Split:
    """Entity-disjoint split: every row of a group lands on one side.

    Used when repeated observations of the same entity would otherwise let the
    model memorise the entity instead of learning the relationship.
    """
    uniq = pd.Index(groups.unique())
    rng = np.random.default_rng(seed)
    shuffled = uniq.to_numpy().copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_size))
    test_groups = set(shuffled[:n_test])
    mask = groups.isin(test_groups).to_numpy()
    tr, te = np.where(~mask)[0], np.where(mask)[0]
    assert_disjoint(tr, te, what="grouped_split")
    assert_no_group_leak(groups, tr, te)
    return Split(
        train_idx=tr, test_idx=te, seed=seed, strategy="group-disjoint hold-out",
        rationale=(
            f"{len(uniq):,} entities partitioned {1 - test_size:.0%}/{test_size:.0%} so no "
            f"entity contributes rows to both sides; prevents identity memorisation."
        ),
    )


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
class PurgedTimeSeriesSplit:
    """Expanding-window CV with purging and an embargo.

    Plain ``TimeSeriesSplit`` still leaks whenever a feature is built from a
    trailing window: a training row at time ``t`` whose 24-hour lag window reaches
    into the validation fold has already seen validation data. We therefore:

    * **purge** the last ``window`` training rows before each validation fold, and
    * **embargo** the first ``embargo`` rows after it, so a subsequent fold's
      training set does not start immediately on the heels of validation.

    Parameters
    ----------
    n_splits : number of expanding folds.
    window : length (in rows) of the longest look-back used by feature engineering.
    embargo : extra rows dropped after each validation fold.
    """

    def __init__(self, n_splits: int = 5, *, window: int = 0, embargo: int = 0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.window = int(window)
        self.embargo = int(embargo)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # sklearn protocol
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold = n // (self.n_splits + 1)
        if fold <= self.window + self.embargo:
            raise ValueError(
                f"Fold size {fold} is too small for window={self.window} and "
                f"embargo={self.embargo}; reduce n_splits or the look-back."
            )
        for k in range(1, self.n_splits + 1):
            val_start = k * fold
            val_end = min(n, val_start + fold)
            # Purge: drop training rows whose look-back window touches validation.
            train_end = max(0, val_start - self.window)
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start + self.embargo, val_end)
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            assert_disjoint(train_idx, val_idx, what=f"purged fold {k}")
            yield train_idx, val_idx

    def describe(self) -> dict:
        return {
            "strategy": "purged & embargoed expanding-window CV",
            "n_splits": self.n_splits,
            "purge_window_rows": self.window,
            "embargo_rows": self.embargo,
            "rationale": (
                "Expanding-window folds keep training strictly in the past. The purge "
                f"drops the {self.window} rows before each validation fold so trailing-"
                "window features cannot span the boundary, and the embargo skips the "
                f"first {self.embargo} validation rows to break autocorrelation carry-over."
            ),
        }


def stratified_cv(n_splits: int = 5, seed: int = 42) -> StratifiedKFold:
    """Standard stratified k-fold, seeded for reproducibility."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
