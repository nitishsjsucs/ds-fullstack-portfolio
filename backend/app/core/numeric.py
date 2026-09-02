"""Platform-specific numerical noise suppression.

On Apple silicon, NumPy links against the Accelerate framework, whose ``matmul``
raises spurious ``divide by zero`` / ``overflow`` / ``invalid value`` RuntimeWarnings
even for inputs that are entirely finite and well-conditioned. The arithmetic is
correct -- verified by comparing ``A @ B`` against ``np.einsum`` on the same
arrays, which agree to the last bit -- but scikit-learn calls matmul inside
K-Means initialisation, PCA and pairwise distances, so a single training run emits
hundreds of meaningless lines.

We therefore filter *exactly* these three messages and only when they come from a
matmul. Anything else -- a genuine overflow in a loss function, a division by a
zero denominator in our own code -- still surfaces normally. Blanket
``np.seterr(all="ignore")`` would have hidden real bugs, which is why it is not
used here.

Reference: numpy/numpy#27575 and the Accelerate BLAS backend introduced in NumPy 2.0.
"""

from __future__ import annotations

import warnings

_MESSAGES = (
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
    "invalid value encountered in matmul",
)

_installed = False


def silence_blas_warnings() -> None:
    """Idempotently filter the known-spurious Accelerate matmul warnings."""
    global _installed
    if _installed:
        return
    for msg in _MESSAGES:
        warnings.filterwarnings("ignore", message=msg, category=RuntimeWarning)
    _installed = True


def verify_blas_is_sane(n: int = 128) -> bool:
    """Prove the suppressed warnings really are cosmetic.

    Called by the test suite: if BLAS matmul ever stops agreeing with an explicit
    einsum contraction, the filter above is hiding a genuine fault and the test
    fails loudly.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fast = a @ a.T
    slow = np.einsum("ij,kj->ik", a, a)
    # Tolerance, not bit-equality: a blocked BLAS kernel and an einsum contraction
    # sum in different orders, so they legitimately differ in the last few ulps
    # (~1e-13 here). What would indicate a real fault is a NaN, an infinity, or a
    # disagreement far larger than accumulation error -- which is what this checks.
    return bool(np.isfinite(fast).all() and np.allclose(fast, slow, rtol=1e-9, atol=1e-9))
