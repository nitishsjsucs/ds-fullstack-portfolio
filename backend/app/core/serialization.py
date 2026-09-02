"""NumPy/pandas -> JSON coercion.

FastAPI cannot serialise ``np.float32``, ``np.int64``, ``NaN`` or ``Timestamp``,
and every one of those leaks out of scikit-learn at some point. ``jsonable``
normalises a whole payload once, at the boundary, so no downstream module has to
remember to call ``float()`` on a metric.

NaN and +/-Inf become ``None`` rather than the JavaScript-invalid literals that
``json.dumps`` emits by default -- the frontend then has exactly one "missing"
sentinel to handle.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def jsonable(obj: Any) -> Any:
    """Recursively convert *obj* into something ``json.dumps`` accepts."""
    # Order matters: numpy scalars are instances of several ABCs.
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)) and not isinstance(obj, np.generic):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Series):
        return [jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [jsonable(rec) for rec in obj.to_dict(orient="records")]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):  # 0-d numpy leftovers
        try:
            return jsonable(obj.item())
        except Exception:  # pragma: no cover - defensive
            pass
    return str(obj)


def round_floats(obj: Any, ndigits: int = 6) -> Any:
    """Shrink payloads by rounding; charts never need 17 significant digits."""
    if isinstance(obj, float):
        return round(obj, ndigits) if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def write_json(path: Path, payload: Any, *, ndigits: int = 6) -> Path:
    """Persist a training artifact as compact, deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = round_floats(jsonable(payload), ndigits)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
