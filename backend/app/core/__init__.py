"""Shared data-science machinery used by all eight projects.

Importing this package installs the platform warning filter described in
:mod:`app.core.numeric`, so training scripts and the API server both get clean
output without any module having to remember to do it.
"""

from .numeric import silence_blas_warnings

silence_blas_warnings()

__all__ = ["silence_blas_warnings"]
