"""Regression tests for validation-based /train and rollback-safe routing."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api import app, get_settings
from src.api_schemas import TrainRequest
from src.api_service import ApiSettings, model_status, train_models
from src.validation_promotion import MODEL_NAME, ValidationDecision
from src.validation_training import ValidationTrainingResult


def candidate(decision: ValidationDecision | None = None) -> ValidationTrainingResult:
    """A model whose baseline was measured in the same validation run."""
    return ValidationTrainingResult(
        run_id="validation-run", model_version=14, protocol_id="a" * 64,
        fit_rows=640, validation_rows=160,
        candidate_metrics={"mae": 11.0, "rmse": 16.0},
        baseline_metrics={"mae": 20.0, "rmse": 29.0},
        best_params={"regressor__n_estimators": 100}, decision=decision,
    )


@pytest.fixture
def audit(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, object]]:
    import src.api_service as service

    seen: list[tuple[str, object]] = []
    monkeypatch.setattr(service.mlflow, "set_tracking_uri", lambda uri: seen.append(("uri", uri)))
    monkeypatch.setattr(service.mlflow, "set_experiment", lambda name: seen.append(("experiment", name)))
    monkeypatch.setattr(service.mlflow, "set_tag", lambda key, val: seen.append((key, val)))
    monkeypatch.setattr(service.mlflow, "log_dict", lambda data, path: seen.append((path, data)))
    monkeypatch.setattr(
        service.mlflow, "start_run",
        lambda **kwargs: nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="api-audit"))),
    )
    return seen


def test_validation_train_does_not_use_legacy_or_final_test(
    monkeypatch: pytest.MonkeyPatch, audit: list[tuple[str, object]],
) -> None:
    import src.api_service as service

    called: list[dict[str, object]] = []

    def validated(**kwargs: object) -> ValidationTrainingResult:
        called.append(kwargs)
        return candidate()

    monkeypatch.setattr(service, "train_validation_candidate", validated)
    monkeypatch.setattr(service, "train_experiment", lambda **kwargs: pytest.fail("legacy must not run"))
    monkeypatch.setattr(service, "promote_candidate", lambda **kwargs: pytest.fail("old gate must not run"))
    settings = ApiSettings(
        training_workflow="validation", train_path=Path("train.csv"),
        test_path=Path("INVALID_FINAL_TEST.csv"), tracking_uri="http://mlflow:5000",
    )
    result = train_models(TrainRequest(auto_promote=False, random_state=11), settings)
    assert called == [{
        "train_path": Path("train.csv"), "tracking_uri": "http://mlflow:5000",
        "model_name": MODEL_NAME, "random_state": 11, "auto_promote": False,
        "min_relative_improvement": 0.01,
    }]
    assert result.training_workflow == "validation"
    assert result.registered_model_name == MODEL_NAME
    assert result.baseline_run_id is None
    assert result.challenger_run_id == "validation-run"
    assert result.validation_protocol_id == "a" * 64
    assert result.promotion_accepted is None
    assert result.audit_run_id == "api-audit"
    artifact = next(value for key, value in audit if key == "api/train.json")
    assert isinstance(artifact, dict)
    assert artifact["baseline"] == {"run_id": None, "metrics": candidate().baseline_metrics}
    assert artifact["promotion"] is None
    assert artifact["fit_rows"] == 640 and artifact["validation_rows"] == 160
    assert "INVALID_FINAL_TEST.csv" not in json.dumps(artifact)
    assert ("training.workflow", "validation") in audit
    assert ("validation.protocol_id", "a" * 64) in audit


@pytest.mark.parametrize("accepted", [True, False])
def test_decision_is_preserved_for_audit(
    accepted: bool, monkeypatch: pytest.MonkeyPatch, audit: list[tuple[str, object]],
) -> None:
    import src.api_service as service

    decision = ValidationDecision(
        accepted=accepted, reason="passed" if accepted else "rejected",
        candidate_version=14, previous_champion_version=9, protocol_id="a" * 64,
        candidate_metrics={"mae": 11.0}, baseline_metrics={"mae": 20.0},
        champion_metrics={"mae": 12.0},
    )
    monkeypatch.setattr(service, "train_validation_candidate", lambda **kwargs: candidate(decision))
    result = train_models(TrainRequest(auto_promote=True), ApiSettings(training_workflow="validation"))
    assert result.promotion_accepted is accepted
    assert result.promotion_reason == decision.reason
    artifact = next(value for key, value in audit if key == "api/train.json")
    assert isinstance(artifact, dict)
    assert artifact["promotion"] == {
        "accepted": accepted, "reason": decision.reason, "previous_champion_version": 9,
    }


def test_legacy_remains_available_and_does_not_call_validation(
    monkeypatch: pytest.MonkeyPatch, audit: list[tuple[str, object]],
) -> None:
    import src.api_service as service
    from src.train import TrainingRun

    monkeypatch.setattr(service, "train_validation_candidate", lambda **kwargs: pytest.fail("validation not selected"))
    monkeypatch.setattr(service, "train_experiment", lambda **kwargs: [
        TrainingRun("baseline", "b", {"mae": 30.0}),
        TrainingRun("random_forest", "c", {"mae": 20.0}, 8),
    ])
    result = train_models(TrainRequest(auto_promote=False), ApiSettings(training_workflow="legacy"))
    assert result.training_workflow == "legacy"
    assert result.registered_model_name == "github-stars-regressor"
    assert result.validation_protocol_id is None
    assert result.baseline_run_id == "b"
    assert ("api.endpoint", "train") in audit


@pytest.mark.parametrize("raw", ["validation", " VALIDATION "])
def test_validation_workflow_env(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("MLSECOPS_TRAIN_WORKFLOW", raw)
    assert ApiSettings.from_env().training_workflow == "validation"


def test_legacy_workflow_env_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLSECOPS_TRAIN_WORKFLOW", raising=False)
    assert ApiSettings.from_env().training_workflow == "legacy"
    monkeypatch.setenv("MLSECOPS_TRAIN_WORKFLOW", "legacy")
    assert ApiSettings.from_env().training_workflow == "legacy"
    monkeypatch.setenv("MLSECOPS_TRAIN_WORKFLOW", "typo")
    with pytest.raises(ValueError, match="MLSECOPS_TRAIN_WORKFLOW"):
        ApiSettings.from_env()


def test_status_displays_selected_training_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.api_service as service

    monkeypatch.setattr(service, "_serving_champion", lambda settings: ("serving-model", 4, "a" * 64))
    status = model_status(ApiSettings(training_workflow="validation"))
    assert status.training_model_name == MODEL_NAME
    assert status.serving_model_name == "serving-model"
    assert status.validation_protocol_id == "a" * 64


def test_swagger_train_response_shows_validated_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.api_service as service

    monkeypatch.setattr(service, "train_validation_candidate", lambda **kwargs: candidate())
    # The HTTP layer must not bypass the existing X-API-Key check.
    app.dependency_overrides[get_settings] = lambda: ApiSettings(
        training_workflow="validation", train_api_key="secret",
    )
    try:
        with TestClient(app) as client:
            assert client.post("/train", json={}).status_code == 401
            response = client.post("/train", json={}, headers={"X-API-Key": "secret"})
        assert response.status_code == 200
        data = response.json()
        assert data["training_workflow"] == "validation"
        assert data["registered_model_name"] == MODEL_NAME
        assert data["baseline_run_id"] is None
        assert data["validation_protocol_id"] == "a" * 64
    finally:
        app.dependency_overrides.clear()
