"""Feature preparation for hourly bike-share demand forecasting.

Time series is where leakage is easiest to introduce and hardest to see, because
nothing errors -- you just get a suspiciously good score. Three rules govern this
module.

**Lags only ever look backwards.** Every lag and rolling statistic is computed
with ``shift(1)`` applied first, so the feature for hour *t* is built exclusively
from data available strictly before *t*. Writing ``rolling(24).mean()`` without
the shift includes the current hour in its own predictor -- a leak that typically
halves the reported error.

**Gaps are made explicit.** The archive is missing 165 hours. Reindexing to a
complete hourly grid turns those into visible NaNs, so a "24-hour lag" really is
24 hours ago rather than "24 rows ago, whenever those happened to be". Silently
treating rows as evenly spaced is the second-most-common time-series bug.

**The target is never used to build a feature about itself at the same timestamp.**
``casual`` and ``registered`` sum exactly to ``cnt``; either one would predict the
target perfectly and neither is known in advance. They are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core import paths

TARGET = "cnt"
TIME = "timestamp"

# Known at forecast time because they are calendar facts or published forecasts.
EXOGENOUS = ["season", "holiday", "weekday", "workingday", "weathersit",
             "temp", "atemp", "hum", "windspeed"]

# Lags chosen from the ACF: 1h (persistence), 24h (same hour yesterday),
# 168h (same hour last week -- the strongest signal in a commuting series).
LAGS = [1, 2, 3, 24, 25, 48, 168]
ROLLING_WINDOWS = [24, 168]

# The longest look-back, in hours. The purge window in cross-validation must be at
# least this large or a training row's feature window will reach into validation.
MAX_LOOKBACK = max(max(LAGS), max(ROLLING_WINDOWS))

FORBIDDEN = ["casual", "registered"]


def load_raw() -> pd.DataFrame:
    df = pd.read_parquet(paths.curated("bike_sharing_hourly"))
    df[TIME] = pd.to_datetime(df[TIME])
    return df.sort_values(TIME).reset_index(drop=True)


def reindex_complete(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Put the series on a gap-free hourly grid so lags mean what they say."""
    full = pd.date_range(df[TIME].min(), df[TIME].max(), freq="h")
    out = df.set_index(TIME).reindex(full)
    out.index.name = TIME
    missing = int(out[TARGET].isna().sum())
    info = {
        "observed_hours": int(len(df)),
        "expected_hours": int(len(full)),
        "missing_hours": missing,
        "missing_pct": round(missing / len(full), 5),
        "policy": (
            f"The archive omits {missing} hours. They are reindexed in as explicit NaNs "
            f"rather than left as absent rows, so a 24-lag is genuinely 24 hours back. "
            f"Rows whose target is missing are excluded from training and scoring; they "
            f"are never imputed, because inventing demand would corrupt both the lag "
            f"features and the error metric."
        ),
    }
    return out.reset_index(), info


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features plus strictly-backward-looking lags and rolling stats."""
    out = df.copy()
    ts = out[TIME]
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["year"] = ts.dt.year
    out["dayofyear"] = ts.dt.dayofyear
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    for col, period in (("hour", 24), ("dayofweek", 7), ("month", 12)):
        theta = 2 * np.pi * out[col] / period
        out[f"{col}_sin"] = np.sin(theta)
        out[f"{col}_cos"] = np.cos(theta)

    # shift(1) FIRST, then lag/roll. This is the line that makes the whole project
    # honest: without it every rolling window contains the value it is predicting.
    past = out[TARGET].shift(1)
    for lag in LAGS:
        out[f"lag_{lag}"] = out[TARGET].shift(lag)
    for w in ROLLING_WINDOWS:
        out[f"roll_mean_{w}"] = past.rolling(w, min_periods=max(2, w // 4)).mean()
        out[f"roll_std_{w}"] = past.rolling(w, min_periods=max(2, w // 4)).std()
    out["roll_max_24"] = past.rolling(24, min_periods=6).max()
    # Change over the last day, a cheap trend proxy.
    out["delta_24"] = out["lag_1"] - out["lag_25"]

    for col in FORBIDDEN:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    calendar = ["hour", "dayofweek", "month", "dayofyear", "is_weekend",
                "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
                "month_sin", "month_cos"]
    lags = [f"lag_{l}" for l in LAGS]
    rolls = [f"roll_mean_{w}" for w in ROLLING_WINDOWS] + \
            [f"roll_std_{w}" for w in ROLLING_WINDOWS] + ["roll_max_24", "delta_24"]
    exo = [c for c in EXOGENOUS if c in df.columns]
    return [c for c in calendar + lags + rolls + exo if c in df.columns]


def prepare() -> tuple[pd.DataFrame, list[str], dict]:
    raw = load_raw()
    gridded, gap_info = reindex_complete(raw)
    eng = engineer(gridded)
    cols = feature_columns(eng)

    for bad in FORBIDDEN:
        assert bad not in cols, f"{bad} sums to the target and must not be a feature"

    # Drop the warm-up period where the longest lag is undefined, and any hour whose
    # target is missing. Both are counted rather than silently dropped.
    n_before = len(eng)
    usable = eng.dropna(subset=[TARGET]).copy()
    n_target_missing = n_before - len(usable)
    usable = usable.dropna(subset=cols)
    n_warmup = n_before - n_target_missing - len(usable)
    usable = usable.reset_index(drop=True)

    meta = {
        **gap_info,
        "rows_modelled": int(len(usable)),
        "rows_dropped_missing_target": int(n_target_missing),
        "rows_dropped_warmup": int(n_warmup),
        "warmup_reason": (f"The {MAX_LOOKBACK}-hour look-back is undefined at the start "
                          f"of the series; those rows cannot be scored fairly."),
        "features": len(cols),
        "max_lookback_hours": MAX_LOOKBACK,
        "forbidden_features": FORBIDDEN,
        "forbidden_reason": ("casual + registered == cnt exactly. Either would give a "
                             "perfect score and neither is known before the hour "
                             "happens."),
        "span": {"start": str(usable[TIME].min()), "end": str(usable[TIME].max())},
    }
    return usable, cols, meta


def seasonal_naive(y: pd.Series, season: int = 168) -> np.ndarray:
    """Same hour last week. The baseline every forecast must beat to be worth using."""
    return y.shift(season).to_numpy()


def stl_decomposition(series: pd.Series, period: int = 24) -> dict:
    """Trend / seasonal / residual split, for the decomposition view."""
    from statsmodels.tsa.seasonal import STL

    s = series.astype(float).interpolate(limit_direction="both")
    res = STL(s, period=period, robust=True).fit()
    step = max(1, len(s) // 900)
    idx = np.arange(0, len(s), step)
    var_total = float(np.var(s))
    return {
        "period": period,
        "points": [
            {"i": int(i),
             "observed": round(float(s.iloc[i]), 2),
             "trend": round(float(res.trend.iloc[i]), 2),
             "seasonal": round(float(res.seasonal.iloc[i]), 2),
             "residual": round(float(res.resid.iloc[i]), 2)}
            for i in idx
        ],
        "variance_share": {
            "trend": round(float(np.var(res.trend) / var_total), 4) if var_total else None,
            "seasonal": round(float(np.var(res.seasonal) / var_total), 4) if var_total else None,
            "residual": round(float(np.var(res.resid) / var_total), 4) if var_total else None,
        },
        "note": ("STL with a 24-hour period. The seasonal share tells you how much of "
                 "demand is pure daily rhythm -- and therefore how much a model can "
                 "possibly add beyond knowing the hour."),
    }


def correlogram(series: pd.Series, nlags: int = 72) -> dict:
    """ACF and PACF with significance bands -- how the lag set was chosen."""
    from statsmodels.tsa.stattools import acf, pacf

    s = series.astype(float).dropna()
    a = acf(s, nlags=nlags, fft=True)
    p = pacf(s, nlags=min(nlags, len(s) // 2 - 1), method="ywm")
    band = 1.96 / np.sqrt(len(s))
    return {
        "nlags": nlags,
        "confidence_band": round(float(band), 4),
        "acf": [{"lag": i, "value": round(float(v), 4),
                 "significant": bool(abs(v) > band)} for i, v in enumerate(a)],
        "pacf": [{"lag": i, "value": round(float(v), 4),
                  "significant": bool(abs(v) > band)} for i, v in enumerate(p)],
        "note": ("Peaks at lag 24 and 168 are what motivated those lags as features: "
                 "yesterday's same hour and last week's same hour carry the most "
                 "information in a commuting series."),
    }


def stationarity(series: pd.Series) -> dict:
    """ADF and KPSS, which test opposite nulls and should be read together."""
    from statsmodels.tsa.stattools import adfuller, kpss

    s = series.astype(float).dropna()
    sample = s.iloc[-6000:] if len(s) > 6000 else s
    out = {}
    try:
        adf = adfuller(sample, autolag="AIC")
        out["adf"] = {"statistic": round(float(adf[0]), 4), "p_value": round(float(adf[1]), 5),
                      "null": "unit root (non-stationary)",
                      "reject_null": bool(adf[1] < 0.05)}
    except Exception as exc:
        out["adf"] = {"error": str(exc)}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = kpss(sample, regression="c", nlags="auto")
        out["kpss"] = {"statistic": round(float(k[0]), 4), "p_value": round(float(k[1]), 5),
                       "null": "stationary", "reject_null": bool(k[1] < 0.05)}
    except Exception as exc:
        out["kpss"] = {"error": str(exc)}
    out["interpretation"] = (
        "ADF and KPSS test opposite nulls, so they are only convincing when they "
        "agree. Hourly bike demand is strongly seasonal rather than trending: the "
        "level is stable but the daily cycle is enormous, which is why the models "
        "here use explicit seasonal lags rather than differencing."
    )
    return out
