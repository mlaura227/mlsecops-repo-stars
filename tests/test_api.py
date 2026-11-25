"""FastAPI tests: OpenAPI, validation, auth, error mappings, and responses."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app, get_settings
from src.api_schemas import PredictResponse, TrainResponse
from src.api_service import ApiSettings, ChampionUnavailable, InvalidPrediction


@pytest.fixture
def client() -> TestClient:
    """Use a fresh testing override for every test."""
    app.dependency_overrides[get_settings] = lambda: ApiSettings(train_api_key="secret")
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def valid_features() -> dict[str, object]:
    return {
        "language": "Python", "forks": 101, "open_issues": 3, "age_days": 500,
        "has_coc": True, "has_contributing": False,
        "has_workflows": True, "has_readme": True,
    }


def test_health_and_openapi(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    definition = client.get("/openapi.json").json()
    assert "/train" in definition["paths"] and "/predict" in definition["paths"]
    assert client.get("/docs").status_code == 200


def test_predict_success(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                         valid_features: dict[str, object]) -> None:
    import src.api as api

    def fake_predict(payload: object, settings: ApiSettings) -> PredictResponse:
        assert settings.model_name == "github-stars-regressor"
        assert getattr(payload, "language") == "Python"
        return PredictResponse(request_id="r-1", prediction=3000.0,
                               model_name=settings.model_name, model_version=4, audit_run_id="audit-1")

    monkeypatch.setattr(api, "predict_stars", fake_predict)
    response = client.post("/predict", json=valid_features)
    assert response.status_code == 200
    assert response.json()["prediction"] == 3000.0
    assert response.json()["audit_run_id"] == "audit-1"


@pytest.mark.parametrize("change", [
    {"stars": 2000}, {"forks": -1}, {"forks": 1.3},
    {"forks": "100"}, {"language": "  "}, {"language": None},
    {"has_coc": "yes"}, {"age_days": -4},
])
def test_predict_rejects_invalid_input(client: TestClient, valid_features: dict[str, object],
                                       change: dict[str, object]) -> None:
    response = client.post("/predict", json={**valid_features, **change})
    assert response.status_code == 422


def test_predict_missing_feature(client: TestClient, valid_features: dict[str, object]) -> None:
    del valid_features["open_issues"]
    assert client.post("/predict", json=valid_features).status_code == 422


@pytest.mark.parametrize(("error", "code"), [
    (ChampionUnavailable("No champion"), 503),
    (InvalidPrediction("Bad model"), 502),
])
def test_predict_service_error(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                               valid_features: dict[str, object], error: Exception, code: int) -> None:
    import src.api as api

    def failure(payload: object, settings: ApiSettings) -> PredictResponse:
        raise error

    monkeypatch.setattr(api, "predict_stars", failure)
    response = client.post("/predict", json=valid_features)
    assert response.status_code == code
    assert response.json()["detail"] == str(error)


def test_train_auth_and_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.api as api

    calls: list[tuple[object, ApiSettings]] = []

    def fake_train(payload: object, settings: ApiSettings) -> TrainResponse:
        calls.append((payload, settings))
        return TrainResponse(
            baseline_run_id="baseline", challenger_run_id="challenger",
            baseline_metrics={"mae": 10.0}, challenger_metrics={"mae": 9.0},
            registered_model_version=5, promotion_accepted=True,
            promotion_reason="Gate passed", audit_run_id="train-audit",
        )

    monkeypatch.setattr(api, "train_models", fake_train)
    assert client.post("/train", json={}).status_code == 401
    assert client.post("/train", headers={"X-API-Key": "wrong"}, json={}).status_code == 401
    assert client.post("/train", headers={"X-API-Key": "secret"}, json={"random_state": -1}).status_code == 422
    response = client.post("/train", headers={"X-API-Key": "secret"}, json={})
    assert response.status_code == 200
    assert response.json()["promotion_accepted"] is True
    assert response.json()["audit_run_id"] == "train-audit"
    assert len(calls) == 1
    assert getattr(calls[0][0], "auto_promote") is True


def test_train_open_when_key_unset(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.api as api

    app.dependency_overrides[get_settings] = lambda: ApiSettings(train_api_key=None)
    monkeypatch.setattr(api, "train_models", lambda payload, settings: TrainResponse(
        baseline_run_id="a", challenger_run_id="b", baseline_metrics={}, challenger_metrics={},
        registered_model_version=1, promotion_accepted=None, promotion_reason=None, audit_run_id="c",
    ))
    response = client.post("/train", json={"auto_promote": False})
    assert response.status_code == 200
    assert response.json()["promotion_accepted"] is None


def test_train_rejects_client_paths_and_bad_gate(client: TestClient) -> None:
    for input_data in ({"train_path": str(Path("/tmp/other.csv"))},
                       {"min_mae_improvement": 1.0},
                       {"min_mae_improvement": -0.1},
                       {"random_state": 3.5},
                       {"auto_promote": "yes"}):
        assert client.post("/train", headers={"X-API-Key": "secret"}, json=input_data).status_code == 422


def test_model_status_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.api as api
    from src.api_schemas import ModelStatusResponse

    def fake_status(settings: ApiSettings) -> ModelStatusResponse:
        return ModelStatusResponse(
            serving_model_name="github-stars-regressor-validation-v1",
            serving_model_version=11,
            training_model_name=settings.model_name,
            validation_protocol_id="d" * 64,
        )

    monkeypatch.setattr(api, "model_status", fake_status)
    response = client.get("/model-status")
    assert response.status_code == 200
    assert response.json() == {
        "serving_model_name": "github-stars-regressor-validation-v1",
        "serving_model_version": 11,
        "training_model_name": "github-stars-regressor",
        "validation_protocol_id": "d" * 64,
    }


def test_model_status_endpoint_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.api as api

    def unavailable(settings: ApiSettings) -> object:
        raise ChampionUnavailable("No champion alias exists")

    monkeypatch.setattr(api, "model_status", unavailable)
    response = client.get("/model-status")
    assert response.status_code == 503
    assert response.json()["detail"] == "No champion alias exists"
