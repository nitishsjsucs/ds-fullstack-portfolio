"""API contract tests.

These exercise the surface the frontend actually calls, so a broken payload fails
here rather than as an empty panel in the browser.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import registry
from app.core.artifacts import ArtifactStore
from app.main import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


TRAINED = [p.slug for p in registry.PROJECTS if ArtifactStore(p.slug).is_trained]
UNTRAINED_REASON = "project has no artifacts; run scripts/train_all.py"


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["projects"] == len(registry.PROJECTS)


def test_project_index_lists_every_registered_project(client):
    body = client.get("/api/projects").json()
    assert body["count"] == len(registry.PROJECTS)
    slugs = {p["slug"] for p in body["projects"]}
    assert slugs == set(registry.SLUGS)
    for p in body["projects"]:
        # The navigation depends on all of these being present.
        for key in ("title", "tagline", "task", "primary_metric", "accent", "number"):
            assert p.get(key), f"{p['slug']} is missing {key}"


def test_unknown_project_returns_404(client):
    assert client.get("/api/projects/not-a-project").status_code == 404


def test_unknown_payload_returns_404(client):
    if not TRAINED:
        pytest.skip(UNTRAINED_REASON)
    assert client.get(f"/api/projects/{TRAINED[0]}/nonsense").status_code == 404


@pytest.mark.parametrize("slug", TRAINED or ["__none__"])
@pytest.mark.parametrize("payload", ["overview", "crispdm", "evaluation", "model_card", "audit"])
def test_core_payloads_are_served(client, slug, payload):
    if slug == "__none__":
        pytest.skip(UNTRAINED_REASON)
    res = client.get(f"/api/projects/{slug}/{payload}")
    assert res.status_code == 200, f"{slug}/{payload}: {res.text[:200]}"
    assert isinstance(res.json(), dict)


@pytest.mark.parametrize("slug", TRAINED or ["__none__"])
def test_crispdm_record_is_complete(client, slug):
    if slug == "__none__":
        pytest.skip(UNTRAINED_REASON)
    d = client.get(f"/api/projects/{slug}/crispdm").json()
    assert d["gates_total"] == 6, "CRISP-DM has six phases"
    assert len(d["phases"]) == 6
    assert d["business_question"]
    assert d["success_criteria"]
    for phase in d["phases"]:
        assert phase["summary"], f"{slug}/{phase['key']} has no summary"
        assert phase["gate"]["criteria"], f"{slug}/{phase['key']} has no exit criteria"
        for finding in phase["findings"]:
            # A finding without evidence is an assertion, which is what this
            # structure exists to prevent.
            assert finding["evidence"], f"{slug}/{phase['key']} finding lacks evidence"


@pytest.mark.parametrize("slug", TRAINED or ["__none__"])
def test_audit_reports_every_check_with_evidence(client, slug):
    if slug == "__none__":
        pytest.skip(UNTRAINED_REASON)
    d = client.get(f"/api/projects/{slug}/audit").json()
    assert d["total"] >= 8, "an audit with fewer than eight checks is not an audit"
    assert d["passed"] <= d["total"]
    for check in d["checks"]:
        assert check["status"] in {"pass", "warn", "fail"}
        assert len(check["evidence"]) > 20, f"{slug}: '{check['check']}' has thin evidence"


@pytest.mark.parametrize("slug", TRAINED or ["__none__"])
def test_model_card_states_its_limits(client, slug):
    if slug == "__none__":
        pytest.skip(UNTRAINED_REASON)
    card = client.get(f"/api/projects/{slug}/model_card").json()
    assert card["out_of_scope"], f"{slug} does not say what it must not be used for"
    assert card["limitations"], f"{slug} claims no limitations"
    assert card["ethical_considerations"]
    assert card["intended_use"]


def test_datasets_manifest_is_served(client):
    res = client.get("/api/datasets")
    if res.status_code == 404:
        pytest.skip("run scripts/fetch_data.py first")
    body = res.json()
    assert body["datasets"]
    for d in body["datasets"]:
        assert d["sha256"] and len(d["sha256"]) == 64
        assert d["source"]["url"].startswith("http")
        assert d["source"]["licence"]


# --------------------------------------------------------------------------- #
# live inference
# --------------------------------------------------------------------------- #
INFERENCE_CASES = {
    "taxi": {"pu_location_id": 161, "do_location_id": 132, "passenger_count": 2},
    "churn": {"tenure": 3, "MonthlyCharges": 89.0, "Contract": "Month-to-month"},
    "segments": {"recency_days": 20, "frequency": 8, "monetary": 3000},
    "basket": {"basket": ["WHITE HANGING HEART T-LIGHT HOLDER"]},
    "automl": {"age": 45, "education_num": 13, "hours_per_week": 45},
    "nanollm": {"prompt": "ROMEO:\n", "max_new_tokens": 40, "temperature": 0.8},
    "forecast": {"timestamp": "2012-10-15T08:00:00", "weathersit": 1},
}


@pytest.mark.parametrize("slug,payload", list(INFERENCE_CASES.items()))
def test_live_inference_returns_a_usable_answer(client, slug, payload):
    if slug not in TRAINED:
        pytest.skip(UNTRAINED_REASON)
    res = client.post(f"/api/projects/{slug}/predict", json=payload)
    assert res.status_code == 200, f"{slug}: {res.text[:300]}"
    body = res.json()
    assert isinstance(body, dict) and body
    assert "latency_ms" in body


def test_taxi_prediction_is_physically_plausible(client):
    if "taxi" not in TRAINED:
        pytest.skip(UNTRAINED_REASON)
    body = client.post(
        "/api/projects/taxi/predict",
        json={
            "pu_location_id": 161,
            "do_location_id": 132,
            "pickup_datetime": "2024-01-26T17:00:00",
            "passenger_count": 2,
        },
    ).json()
    # Midtown to JFK at Friday rush hour: somewhere between 20 and 90 minutes.
    assert 20 < body["duration_minutes"] < 90
    interval = body["duration_interval_80"]
    assert interval["lower"] <= body["duration_minutes"] <= interval["upper"], (
        "The point estimate must lie inside its own interval."
    )
    assert body["fare_usd"] > 20


def test_taxi_attribution_is_additive(client):
    """SHAP's defining property: base value plus contributions equals the
    prediction. If the attribution and the model ever disagree, the explanation
    is decorative."""
    if "taxi" not in TRAINED:
        pytest.skip(UNTRAINED_REASON)
    body = client.post(
        "/api/projects/taxi/predict",
        json={"pu_location_id": 236, "do_location_id": 87, "passenger_count": 1},
    ).json()
    attr = body.get("attribution")
    if not attr:
        pytest.skip("attribution unavailable for this estimator")
    total = attr["base_value"] + sum(c["shap"] for c in attr["contributions"])
    total += attr.get("residual_other_features", 0.0)
    assert total == pytest.approx(attr["prediction"], abs=0.01)


def test_churn_returns_a_decision_not_just_a_score(client):
    if "churn" not in TRAINED:
        pytest.skip(UNTRAINED_REASON)
    body = client.post(
        "/api/projects/churn/predict",
        json={"tenure": 2, "MonthlyCharges": 95.0, "Contract": "Month-to-month"},
    ).json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["economics"]["recommended_action"]
    assert body["decision_threshold"] != 0.5, (
        "The deployed threshold should come from the cost matrix, not the default."
    )


def test_segments_reports_assignment_confidence(client):
    if "segments" not in TRAINED:
        pytest.skip(UNTRAINED_REASON)
    body = client.post(
        "/api/projects/segments/predict",
        json={"recency_days": 15, "frequency": 12, "monetary": 6000},
    ).json()
    assert body["confidence"] in {"high", "medium", "borderline"}
    assert len(body["distances"]) >= 2
    distances = [d["distance"] for d in body["distances"]]
    assert distances == sorted(distances), "Distances must be returned nearest-first"


def test_segments_rejects_impossible_input(client):
    if "segments" not in TRAINED:
        pytest.skip(UNTRAINED_REASON)
    res = client.post(
        "/api/projects/segments/predict",
        json={"recency_days": 15, "frequency": 0, "monetary": 0},
    )
    assert res.status_code == 422


def test_nanollm_generation_respects_its_settings(client):
    if "nanollm" not in TRAINED:
        pytest.skip(UNTRAINED_REASON)
    body = client.post(
        "/api/projects/nanollm/generate",
        json={"prompt": "ROMEO:\n", "max_new_tokens": 50, "temperature": 0.7, "seed": 1},
    ).json()
    assert len(body["generated"]) > 0
    assert body["settings"]["temperature"] == 0.7
    assert body["caveat"], "the model's limits must travel with every response"


def test_nanollm_rejects_out_of_range_temperature(client):
    if "nanollm" not in TRAINED:
        pytest.skip(UNTRAINED_REASON)
    res = client.post("/api/projects/nanollm/generate", json={"temperature": 9})
    assert res.status_code == 422


def test_basket_recommendations_never_repeat_the_basket(client):
    if "basket" not in TRAINED:
        pytest.skip(UNTRAINED_REASON)
    items = client.get("/api/projects/basket/items?limit=5").json()["items"]
    if not items:
        pytest.skip("no items in the artifact")
    seed = items[0]["item"]
    body = client.post("/api/projects/basket/predict", json={"basket": [seed]}).json()
    for rec in body["recommendations"]:
        assert rec["item"] != seed, "Recommending an item already in the basket"
        assert rec["lift"] >= 1.0
