"""Split-conformal calibration of prediction intervals.

Quantile regression gives you a band. It does not give you a band that *covers*:
gradient-boosted quantile models are routinely over-confident, producing nominal
80% intervals that contain 70% of outcomes. An interval that lies about its own
coverage is worse than no interval, because downstream planning treats it as a
bound.

Conformalized Quantile Regression (Romano, Patterson & Candes, NeurIPS 2019)
repairs this with one idea. Hold out a calibration set the quantile models never
saw. For each calibration point measure the *conformity score*

    E_i = max(lo_i - y_i,  y_i - hi_i)

-- how far outside the predicted band the true value fell (negative when it fell
comfortably inside). Take the appropriate empirical quantile of those scores and
widen every future interval by it.

The resulting band carries a finite-sample guarantee: coverage is at least
1 - alpha, under exchangeability of the calibration and test points, with no
assumption whatsoever about the model or the noise distribution.

**The exchangeability caveat is real and is stated wherever this is used.** For
time series the calibration set is the most recent block before the test period,
which is the closest available approximation; if the process drifts, the
guarantee weakens to an approximation. That is still far better than an
uncalibrated band, and unlike the uncalibrated band it is falsifiable -- we
measure empirical coverage on the test set and report the gap.
"""

from __future__ import annotations

import numpy as np


class ConformalInterval:
    """Calibrate a quantile-model band to its nominal coverage."""

    def __init__(self, alpha: float = 0.2) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1); 0.2 gives an 80% interval.")
        self.alpha = alpha
        self.q_: float | None = None
        self.n_calibration_: int = 0

    def calibrate(self, y_cal, lo_cal, hi_cal) -> "ConformalInterval":
        y = np.asarray(y_cal, dtype=float)
        lo = np.asarray(lo_cal, dtype=float)
        hi = np.asarray(hi_cal, dtype=float)
        if not (len(y) == len(lo) == len(hi)):
            raise ValueError("Calibration arrays must be the same length.")
        n = len(y)
        if n < 20:
            raise ValueError(f"Need at least 20 calibration points, got {n}.")

        scores = np.maximum(lo - y, y - hi)
        # The finite-sample correction: the (1-alpha)(1 + 1/n) empirical quantile
        # is what makes the guarantee exact rather than asymptotic.
        level = min(1.0, (1 - self.alpha) * (1 + 1 / n))
        self.q_ = float(np.quantile(scores, level, method="higher"))
        self.n_calibration_ = n
        self.calibration_coverage_ = float(np.mean(scores <= 0))
        return self

    def apply(self, lo, hi) -> tuple[np.ndarray, np.ndarray]:
        if self.q_ is None:
            raise RuntimeError("Call calibrate() before apply().")
        lo = np.asarray(lo, dtype=float) - self.q_
        hi = np.asarray(hi, dtype=float) + self.q_
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def report(self) -> dict:
        return {
            "method": "Conformalized Quantile Regression (Romano et al., 2019)",
            "alpha": self.alpha,
            "nominal_coverage": round(1 - self.alpha, 3),
            "n_calibration": self.n_calibration_,
            "adjustment": None if self.q_ is None else round(self.q_, 4),
            "raw_calibration_coverage": (
                None if not hasattr(self, "calibration_coverage_")
                else round(self.calibration_coverage_, 4)
            ),
            "direction": (
                "widened" if (self.q_ or 0) > 0 else
                "narrowed" if (self.q_ or 0) < 0 else "unchanged"
            ),
            "guarantee": (
                f"Coverage is at least {1 - self.alpha:.0%} under exchangeability of "
                f"the calibration and test sets, with no assumption about the model or "
                f"the noise distribution."
            ),
            "caveat": (
                "For time series, exchangeability is approximate: the calibration block "
                "is the most recent data before the test period, so a drifting process "
                "weakens the guarantee. Empirical coverage is therefore measured on the "
                "test set and reported against nominal rather than assumed."
            ),
        }
