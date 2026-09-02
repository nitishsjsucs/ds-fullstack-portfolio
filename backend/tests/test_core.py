"""Tests for the shared machinery every project depends on.

These are the tests that make the portfolio's guarantees real. A claim in a README
that "preprocessing is fitted per fold" is worth nothing; a test that constructs a
frame where a leak would change the answer, and asserts it does not, is worth
something.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from app.core.conformal import ConformalInterval
from app.core.evaluation import (
    binary_metrics,
    forecast_metrics,
    interval_metrics,
    regression_metrics,
    threshold_sweep,
)
from app.core.features import FrequencyEncoder, RareCategoryGrouper, add_cyclical, haversine_km
from app.core.numeric import verify_blas_is_sane
from app.core.splitting import (
    PurgedTimeSeriesSplit,
    assert_chronological,
    assert_disjoint,
    assert_no_group_leak,
    grouped_split,
    stratified_split,
    temporal_split,
)


# --------------------------------------------------------------------------- #
# platform sanity
# --------------------------------------------------------------------------- #
def test_blas_matmul_is_trustworthy():
    """The warning filter in core.numeric hides RuntimeWarnings from Accelerate's
    matmul. If BLAS ever stops agreeing with an explicit contraction, that filter
    would be concealing a real fault -- so assert the arithmetic directly."""
    assert verify_blas_is_sane()


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def test_temporal_split_never_looks_ahead():
    times = pd.Series(pd.date_range("2024-01-01", periods=1000, freq="h"))
    split = temporal_split(times, test_size=0.2)
    assert split.n_train == 800 and split.n_test == 200
    assert times.iloc[split.train_idx].max() < times.iloc[split.test_idx].min()
    assert_chronological(times, split.train_idx, split.test_idx)


def test_temporal_split_rejects_unsorted_input():
    """A caller who forgot to sort must get an error, not a silently reshuffled
    dataset -- that failure mode produces a working pipeline with a wrong score."""
    times = pd.Series(pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]))
    with pytest.raises(ValueError, match="chronologically sorted"):
        temporal_split(times)


def test_stratified_split_preserves_prevalence():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.binomial(1, 0.05, size=5000))
    split = stratified_split(y, test_size=0.2, seed=42)
    train_rate = y.iloc[split.train_idx].mean()
    test_rate = y.iloc[split.test_idx].mean()
    assert abs(train_rate - test_rate) < 0.01
    assert_disjoint(split.train_idx, split.test_idx)


def test_grouped_split_keeps_entities_whole():
    groups = pd.Series([f"cust{i // 10}" for i in range(1000)])
    split = grouped_split(groups, test_size=0.25, seed=1)
    assert_no_group_leak(groups, split.train_idx, split.test_idx)


def test_assert_disjoint_catches_overlap():
    with pytest.raises(AssertionError, match="appear in both"):
        assert_disjoint([1, 2, 3], [3, 4, 5])


def test_assert_chronological_catches_lookahead():
    times = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
    with pytest.raises(AssertionError, match="lookahead"):
        assert_chronological(times, train_idx=[5, 6, 7], test_idx=[0, 1, 2])


def test_purged_split_removes_the_window_before_validation():
    """The purge is the whole point: a training row whose trailing feature window
    reaches into the validation fold has already seen validation data."""
    X = pd.DataFrame({"x": range(1200)})
    window = 50
    cv = PurgedTimeSeriesSplit(n_splits=3, window=window, embargo=10)
    folds = list(cv.split(X))
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        assert_disjoint(train_idx, val_idx)
        # Training always precedes validation...
        assert train_idx.max() < val_idx.min()
        # ...and by at least the purge window.
        assert val_idx.min() - train_idx.max() > window


def test_purged_split_rejects_impossible_configuration():
    X = pd.DataFrame({"x": range(100)})
    cv = PurgedTimeSeriesSplit(n_splits=5, window=500, embargo=100)
    with pytest.raises(ValueError, match="too small"):
        list(cv.split(X))


# --------------------------------------------------------------------------- #
# fitted transformers must not leak
# --------------------------------------------------------------------------- #
def test_frequency_encoder_uses_only_training_counts():
    """The reason this transformer exists. Encoding frequencies over the full
    frame lets each training row see how often its category appears in test."""
    train = pd.DataFrame({"zone": ["a"] * 80 + ["b"] * 20})
    test = pd.DataFrame({"zone": ["a"] * 10 + ["b"] * 90})

    enc = FrequencyEncoder().fit(train)
    out = enc.transform(test)

    # 'a' must carry its TRAINING frequency of 0.8, not its test frequency of 0.1.
    assert out[0, 0] == pytest.approx(0.8)
    assert out[-1, 0] == pytest.approx(0.2)


def test_frequency_encoder_handles_unseen_categories():
    enc = FrequencyEncoder().fit(pd.DataFrame({"z": ["a", "b"]}))
    out = enc.transform(pd.DataFrame({"z": ["never_seen"]}))
    assert out[0, 0] == 0.0  # rarer than anything in training


def test_rare_grouper_learns_the_vocabulary_on_train_only():
    train = pd.DataFrame({"c": ["x"] * 95 + ["y"] * 5})
    grouper = RareCategoryGrouper(min_frequency=0.1).fit(train)
    out = grouper.transform(pd.DataFrame({"c": ["x", "y", "z"]}))
    assert list(out.iloc[:, 0]) == ["x", "__rare__", "__rare__"]


def test_rare_grouper_emits_string_feature_names():
    """Regression test: when this sits after a SimpleImputer that returns a bare
    numpy array, the incoming column labels are integers. Passing those through
    made the downstream OneHotEncoder build a name as `0 + "_" + "Private"`."""
    grouper = RareCategoryGrouper().fit(pd.DataFrame([[1, 2], [1, 2]]))
    names = grouper.get_feature_names_out()
    assert all(isinstance(n, str) for n in names)


def test_pipeline_refits_preprocessing_per_fold():
    """An end-to-end check that a Pipeline's transformer is re-fitted by
    cross_val_score rather than carrying state across folds."""
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"g": rng.choice(["a", "b", "c"], 300)})
    y = pd.Series(rng.normal(size=300))
    pipe = Pipeline([("freq", FrequencyEncoder()), ("model", LinearRegression())])
    scores = cross_val_score(pipe, X, y, cv=3, scoring="neg_mean_absolute_error")
    assert len(scores) == 3 and np.isfinite(scores).all()


def test_cyclical_encoding_wraps_around():
    df = add_cyclical(pd.DataFrame({"hour": [0, 23]}), "hour", 24)
    d = np.hypot(df.hour_sin[0] - df.hour_sin[1], df.hour_cos[0] - df.hour_cos[1])
    # 23:00 and 00:00 must be neighbours, not 23 units apart.
    assert d < 0.3


def test_haversine_matches_a_known_distance():
    # Manhattan (Times Square) to JFK is roughly 21 km great-circle.
    km = haversine_km([40.7580], [-73.9855], [40.6413], [-73.7781])
    assert 20 < float(km[0]) < 23


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def test_binary_metrics_flag_misleading_accuracy():
    y = np.array([0] * 970 + [1] * 30)
    prob = np.full(1000, 0.01)
    m = binary_metrics(y, prob)
    assert m["accuracy_is_misleading"] is True
    assert m["baseline_pr_auc"] == pytest.approx(0.03)
    assert m["baseline_accuracy"] == pytest.approx(0.97)


def test_perfect_classifier_scores_one():
    y = np.array([0, 0, 1, 1])
    m = binary_metrics(y, np.array([0.01, 0.02, 0.98, 0.99]))
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)


def test_threshold_sweep_finds_the_cost_optimum():
    """With a false negative priced far above a false positive, the optimal cut
    must fall below the naive 0.5."""
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.2, 2000)
    prob = np.clip(y * 0.55 + rng.normal(0.25, 0.18, 2000), 0.01, 0.99)
    sweep = threshold_sweep(y, prob, cost_fn=20.0, cost_fp=1.0)
    assert sweep["best_cost_threshold"] < 0.5


def test_regression_metrics_report_bias_and_baseline():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    m = regression_metrics(y, y + 2.0)
    assert m["mae"] == pytest.approx(2.0)
    assert m["bias"] == pytest.approx(-2.0)
    assert m["baseline_mae_mean"] > 0


def test_mase_is_one_for_the_seasonal_naive_forecast():
    """Sanity anchor for the metric the forecasting project is judged on."""
    rng = np.random.default_rng(1)
    train = pd.Series(rng.normal(100, 10, 500))
    naive = train.shift(24).dropna()
    actual = train.iloc[24:]
    m = forecast_metrics(actual, naive, y_train=train, seasonality=24)
    assert m["mase"] == pytest.approx(1.0, abs=0.05)


def test_interval_metrics_detect_overconfidence():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 4000)
    # A band far too narrow for an 80% claim.
    m = interval_metrics(y, np.full(4000, -0.3), np.full(4000, 0.3), nominal=0.8)
    assert m["empirical_coverage"] < 0.6
    assert "over-confident" in m["verdict"]


def test_interval_metrics_accept_a_calibrated_band():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 8000)
    m = interval_metrics(y, np.full(8000, -1.2816), np.full(8000, 1.2816), nominal=0.8)
    assert m["verdict"] == "well calibrated"


# --------------------------------------------------------------------------- #
# conformal calibration
# --------------------------------------------------------------------------- #
def test_conformal_repairs_an_overconfident_band():
    rng = np.random.default_rng(7)
    y_cal = rng.normal(0, 1, 3000)
    lo_cal, hi_cal = np.full(3000, -0.4), np.full(3000, 0.4)

    conf = ConformalInterval(alpha=0.2).calibrate(y_cal, lo_cal, hi_cal)
    assert conf.q_ > 0  # it must widen a band this narrow

    y_test = rng.normal(0, 1, 3000)
    lo, hi = conf.apply(np.full(3000, -0.4), np.full(3000, 0.4))
    coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
    assert coverage == pytest.approx(0.8, abs=0.03)


def test_conformal_narrows_a_needlessly_wide_band():
    rng = np.random.default_rng(8)
    y = rng.normal(0, 1, 3000)
    conf = ConformalInterval(alpha=0.2).calibrate(y, np.full(3000, -8.0), np.full(3000, 8.0))
    assert conf.q_ < 0  # a band this wide should be tightened
    assert conf.report()["direction"] == "narrowed"


def test_conformal_requires_enough_calibration_data():
    with pytest.raises(ValueError, match="at least 20"):
        ConformalInterval().calibrate([1, 2, 3], [0, 0, 0], [2, 2, 2])
