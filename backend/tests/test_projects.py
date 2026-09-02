"""Per-project leakage and correctness tests.

Each test here targets the specific way *that* project could cheat. Generic tests
live in test_core.py; these are the ones that need domain knowledge to write.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from app import registry
from app.core import paths


# --------------------------------------------------------------------------- #
# data provenance
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not paths.MANIFEST.exists(), reason="run scripts/fetch_data.py first")
def test_curated_files_match_their_recorded_checksums():
    """The manifest claims a SHA-256 for every curated file. If ingestion is not
    deterministic, or a file was edited by hand, this is where it surfaces."""
    manifest = json.loads(paths.MANIFEST.read_text())
    for entry in manifest["datasets"]:
        path = paths.ROOT / entry["file"]
        assert path.exists(), f"{entry['name']} is missing from disk"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], (
            f"{entry['name']} does not match its recorded checksum -- the committed "
            f"file and the manifest disagree."
        )


@pytest.mark.skipif(not paths.MANIFEST.exists(), reason="run scripts/fetch_data.py first")
def test_every_registered_project_has_its_dataset():
    manifest = json.loads(paths.MANIFEST.read_text())
    names = {d["name"] for d in manifest["datasets"]}
    for meta in registry.PROJECTS:
        stem = meta.dataset.replace("tiny_shakespeare", "tiny_shakespeare_stats")
        assert stem in names, f"{meta.slug} references an unknown dataset {meta.dataset}"


# --------------------------------------------------------------------------- #
# taxi: post-outcome columns must never become features
# --------------------------------------------------------------------------- #
def test_taxi_excludes_columns_known_only_after_the_trip():
    from app.projects.taxi import data as D

    forbidden = {"tip_amount", "tolls_amount", "total_amount", "congestion_surcharge",
                 "dropoff_datetime", "duration_min", "fare_amount", "avg_speed_mph"}
    assert not (set(D.ALL_FEATURES) & forbidden), (
        "A feature known only during or after the trip would leak the outcome."
    )


def test_taxi_filters_never_reference_the_target_distribution():
    """Filtering on 'more than 3 standard deviations from the mean duration' would
    delete exactly the hard cases the model is scored on. Every rule must be a
    physical bound instead."""
    from app.projects.taxi import data as D

    for _name, expr, _reason in D.FILTERS:
        lowered = expr.lower()
        for statistic in ("mean", "std", "quantile", "median", "percentile"):
            assert statistic not in lowered, f"Filter {expr!r} references {statistic}"


@pytest.mark.slow
def test_taxi_prepared_frame_is_chronological():
    from app.projects.taxi import data as D

    prepared = D.prepare()
    assert prepared.frame["pickup_datetime"].is_monotonic_increasing
    assert prepared.n_kept < prepared.n_raw  # filters did something
    assert prepared.exclusion_rate < 0.20    # ...but not too much


# --------------------------------------------------------------------------- #
# churn: identifiers and sensitive attributes
# --------------------------------------------------------------------------- #
def test_churn_excludes_the_identifier_and_sensitive_attributes():
    from app.projects.churn import data as D

    assert D.ID_COLUMN not in D.ALL_FEATURES, (
        "customerID is unique per row; a tree given it memorises the training set."
    )
    for col in D.SENSITIVE:
        assert col not in D.ALL_FEATURES, f"{col} is reserved for the fairness audit"


def test_churn_leaves_missing_values_for_the_pipeline():
    """The 11 blank TotalCharges must reach the model as NaN. Filling them at load
    time computes a statistic over the full dataset, including test rows."""
    from app.projects.churn import data as D

    df = D.prepare()
    assert df["TotalCharges"].isna().sum() > 0, (
        "TotalCharges was imputed before the split; that median saw the test rows."
    )


def test_churn_expected_value_crosses_zero_at_breakeven():
    from app.projects.churn import data as D

    ev = D.expected_value(prob=0.5, monthly_charges=70.0)
    at_breakeven = D.expected_value(prob=ev["breakeven_probability"], monthly_charges=70.0)
    assert abs(at_breakeven["expected_value_of_offer"]) < 1.0


# --------------------------------------------------------------------------- #
# forecast: the lookahead trap
# --------------------------------------------------------------------------- #
def test_forecast_rolling_features_are_shifted():
    """The single most important test in the repository.

    A rolling mean computed without shifting first contains the value it is used
    to predict. We construct a series where the target jumps, then assert the
    rolling feature at the jump does NOT reflect it.
    """
    from app.projects.forecast import data as D

    n = 400
    frame = pd.DataFrame({
        D.TIME: pd.date_range("2011-01-01", periods=n, freq="h"),
        D.TARGET: [10.0] * (n - 1) + [10_000.0],  # a huge spike in the final hour
    })
    out = D.engineer(frame)

    spike_row = out.iloc[-1]
    assert spike_row["roll_mean_24"] < 100, (
        "The rolling mean at the spike includes the spike itself -- the shift(1) "
        "before .rolling() is missing, and every score is inflated."
    )
    assert spike_row["lag_1"] == pytest.approx(10.0)


def test_forecast_excludes_columns_that_sum_to_the_target():
    from app.projects.forecast import data as D

    frame = pd.DataFrame({
        D.TIME: pd.date_range("2011-01-01", periods=50, freq="h"),
        D.TARGET: np.arange(50.0),
        "casual": np.arange(50.0) / 2,
        "registered": np.arange(50.0) / 2,
    })
    out = D.engineer(frame)
    for col in D.FORBIDDEN:
        assert col not in out.columns, f"{col} sums to the target and gives a perfect score"


def test_forecast_purge_window_covers_the_longest_lookback():
    """If the CV purge is shorter than the longest feature window, a training row's
    window reaches into validation."""
    from app.projects.forecast import data as D

    assert D.MAX_LOOKBACK >= max(D.LAGS)
    assert D.MAX_LOOKBACK >= max(D.ROLLING_WINDOWS)


# --------------------------------------------------------------------------- #
# anomaly: the label must never reach a detector
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_anomaly_label_is_absent_from_the_feature_matrix():
    from app.projects.anomaly import data as D

    X, y, family, meta = D.prepare()
    assert D.LABEL not in X.columns
    assert D.LABEL_TEXT not in X.columns
    assert len(X) == len(y) == len(family)


def test_anomaly_contamination_is_an_assumption_not_a_measurement():
    """Passing the observed attack rate as contamination is leakage through a
    hyperparameter: in deployment that number is unknown."""
    from app.projects.anomaly import data as D

    assert D.ASSUMED_CONTAMINATION == 0.02
    # The true rate in the curated sample is ~3.37%; the two must not coincide.
    assert abs(D.ASSUMED_CONTAMINATION - 0.0337) > 0.005


def test_anomaly_alert_budget_is_monotone_in_recall():
    from app.projects.anomaly import data as D

    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.03, 5000)
    scores = y * 0.6 + rng.normal(0, 0.2, 5000)
    rows = D.alert_budget_simulation(scores, y)
    recalls = [r["recall_at_k"] for r in rows]
    assert recalls == sorted(recalls), "Recall cannot fall as the queue grows"


# --------------------------------------------------------------------------- #
# automl: dataset-specific traps
# --------------------------------------------------------------------------- #
def test_automl_drops_the_survey_weight_and_the_duplicate_column():
    from app.projects.automl import data as D

    assert "fnlwgt" not in D.ALL_FEATURES, "fnlwgt is a survey-design artefact"
    assert "education" not in D.ALL_FEATURES, "education duplicates education_num"
    for col in ("sex", "race"):
        assert col not in D.ALL_FEATURES, f"{col} is reserved for the fairness audit"


# --------------------------------------------------------------------------- #
# segments: the transform that makes clustering work
# --------------------------------------------------------------------------- #
def test_segments_log_transform_reduces_skew():
    """Without it, K-Means puts 96% of customers in one cluster."""
    from app.projects.segments import data as D

    rng = np.random.default_rng(0)
    heavy = pd.DataFrame({c: rng.lognormal(4, 2, 2000) for c in D.FEATURES})
    transformed = D.build_preprocessor().fit_transform(heavy)
    raw_skew = float(heavy[D.FEATURES[2]].skew())
    new_skew = float(pd.Series(transformed[:, 2]).skew())
    assert abs(new_skew) < abs(raw_skew)


def test_segment_names_are_distinct_across_the_taxonomy():
    """Phase 1 required every segment to carry a distinct action. Two clusters
    with the same label break that, so the taxonomy must be injective."""
    from app.projects.segments import data as D

    overall = {c: 1.0 for c in D.PROFILE_COLUMNS}
    names = set()
    for r in (0.3, 0.6, 1.0, 2.5):
        for f in (0.5, 1.5, 3.0):
            for m in (0.2, 0.8, 1.5, 3.0):
                profile = {**overall, "recency_days": r, "frequency": f, "monetary": m}
                name, action = D.name_segment(profile, overall)
                assert action, f"{name} has no recommended action"
                names.add(name)
    assert len(names) >= 8, "The taxonomy collapses too many centroids onto one name"


# --------------------------------------------------------------------------- #
# nanollm: causal masking is the generative equivalent of leakage
# --------------------------------------------------------------------------- #
def test_transformer_cannot_attend_forward():
    """A model that can see the next token achieves a spectacular training loss by
    copying. The mask is asserted here rather than trusted."""
    torch = pytest.importorskip("torch")
    from app.projects.nanollm.model import ModelConfig, NanoGPT

    cfg = ModelConfig(vocab_size=16, n_layer=2, n_head=2, n_embd=32, block_size=12)
    model = NanoGPT(cfg).eval()
    idx = torch.randint(0, 16, (1, 12))
    with torch.no_grad():
        _, _, attns = model(idx, return_attention=True)

    for layer, att in enumerate(attns):
        a = att[0].numpy()
        upper = a[:, np.triu_indices(a.shape[-1], k=1)[0], np.triu_indices(a.shape[-1], k=1)[1]]
        assert np.allclose(upper, 0.0), f"Layer {layer} attends to future positions"
        rows = a.sum(axis=-1)
        assert np.allclose(rows, 1.0, atol=1e-5), "Attention rows must be a distribution"


def test_transformer_output_depends_only_on_the_prefix():
    """A stronger statement than the mask test: changing a later token must not
    change the logits at an earlier position."""
    torch = pytest.importorskip("torch")
    from app.projects.nanollm.model import ModelConfig, NanoGPT

    cfg = ModelConfig(vocab_size=16, n_layer=2, n_head=2, n_embd=32, block_size=10, dropout=0.0)
    model = NanoGPT(cfg).eval()
    a = torch.randint(0, 16, (1, 10))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 5) % 16  # change only the final token

    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5), (
        "Editing the last token changed earlier logits -- information flowed backwards."
    )


def test_tokenizer_roundtrips():
    from app.projects.nanollm.data import CharTokenizer

    tok = CharTokenizer("hello world\n")
    text = "hello world"
    assert tok.decode(tok.encode(text)) == text
    # Out-of-vocabulary characters are dropped, not mapped to a sentinel.
    assert "z" not in tok.decode(tok.encode("zzz hello"))


def test_language_model_split_is_contiguous_and_unshuffled():
    from app.projects.nanollm import data as D

    if not (paths.CURATED / "tiny_shakespeare.txt").exists():
        pytest.skip("run scripts/fetch_data.py first")
    ds = D.build_dataset(val_fraction=0.1)
    meta = ds["meta"]
    assert meta["train_tokens"] + meta["val_tokens"] == meta["tokens"]
    # The validation block must be the tail of the corpus, not a random sample.
    assert meta["split_at_char"] == meta["train_tokens"]


# --------------------------------------------------------------------------- #
# basket
# --------------------------------------------------------------------------- #
def test_rule_quality_identifies_independence():
    """Lift 1.0 means the items are independent -- the rule restates base rates."""
    from app.projects.basket import data as D

    q = D.rule_quality(support_ab=0.06, support_a=0.2, support_b=0.3)
    assert q["lift"] == pytest.approx(1.0)
    assert q["leverage"] == pytest.approx(0.0, abs=1e-9)
    assert "independent" in D.interpret_lift(q["lift"])


def test_rule_quality_detects_substitution():
    from app.projects.basket import data as D

    q = D.rule_quality(support_ab=0.01, support_a=0.3, support_b=0.4)
    assert q["lift"] < 1
    assert q["zhangs_metric"] < 0  # negative means the items substitute
    assert "substitute" in D.interpret_lift(q["lift"])
