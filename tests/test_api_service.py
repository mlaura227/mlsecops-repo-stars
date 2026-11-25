"""Business-layer tests for MLflow event artifacts and champion-only serving."""

import csv
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

from src.api_schemas import PredictRequest, TrainRequest
from src.api_service import (
    ApiSettings,
    ChampionUnavailable,
    InvalidPrediction,
    _champion_version,
    _log_training_event,
    _select_training_runs,
    model_status,
    predict_stars,
    train_models,
)
from src.promotion import PromotionDecision
from src.train import TrainingRun
from src.data import validate_dataset
from src.preprocessing import split_features_target
from src.training import build_pipeline


@pytest.fixture
def payload() -> PredictRequest:
    return PredictRequest(
        language=" Python ", forks=10, open_issues=3, age_days=850,
        has_coc=True, has_contributing=False, has_workflows=True, has_readme=True,
    )


class FakeMLflow:
    """Track what would be committed to MLflow without network or database."""

    def __init__(self) -> None:
        self.uris: list[str] = []
        self.experiments: list[str] = []
        self.run_names: list[str] = []
        self.tags: list[tuple[str, str]] = []
        self.params: list[dict[str, object]] = []
        self.dicts: list[tuple[str, dict[str, object]]] = []
        self.files: list[tuple[str, str, str]] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.uris.append(uri)

    def set_experiment(self, experiment: str) -> None:
        self.experiments.append(experiment)

    def start_run(self, run_name: str) -> object:
        self.run_names.append(run_name)
        return nullcontext(SimpleNamespace(info=SimpleNamespace(run_id=f"audit-{len(self.run_names)}")))

    def set_tag(self, key: str, value: str) -> None:
        self.tags.append((key, value))

    def log_params(self, params: dict[str, object]) -> None:
        self.params.append(params)

    def log_dict(self, contents: dict[str, object], artifact_file: str) -> None:
        self.dicts.append((artifact_file, contents))

    def log_artifact(self, path: str, artifact_path: str) -> None:
        self.files.append((Path(path).name, artifact_path, Path(path).read_text(encoding="utf-8")))


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> FakeMLflow:
    import src.api_service as service

    fake = FakeMLflow()
    for method in ("set_tracking_uri", "set_experiment", "start_run", "set_tag",
                   "log_params", "log_dict", "log_artifact"):
        monkeypatch.setattr(service.mlflow, method, getattr(fake, method))
    return fake


def registry(monkeypatch: pytest.MonkeyPatch, *, version: object | None) -> None:
    import src.api_service as service

    aliases = {} if version is None else {"champion": version}
    monkeypatch.setattr(service, "MlflowClient", lambda: SimpleNamespace(
        get_registered_model=lambda name: SimpleNamespace(aliases=aliases)))


def test_prediction_logged_as_json_and_csv(
    payload: PredictRequest, monkeypatch: pytest.MonkeyPatch, fake_mlflow: FakeMLflow,
) -> None:
    import src.api_service as service

    registry(monkeypatch, version="7")
    seen: list[pd.DataFrame] = []

    def predict(features: pd.DataFrame) -> np.ndarray:
        seen.append(features.copy())
        return np.array([8000.5])

    monkeypatch.setattr(service, "load_sklearn_model", lambda model_uri: SimpleNamespace(predict=predict))
    outcome = predict_stars(payload, ApiSettings(tracking_uri="sqlite:///test.db"))
    assert outcome.model_version == 7
    assert outcome.prediction == 8000.5
    assert outcome.audit_run_id == "audit-1"
    assert str(UUID(outcome.request_id)) == outcome.request_id
    assert list(seen[0].columns) == ["language", "forks", "open_issues", "age_days", "has_coc",
                                     "has_contributing", "has_workflows", "has_readme"]
    assert seen[0]["language"].iloc[0] == "Python"
    assert seen[0]["has_coc"].iloc[0] == 1
    assert fake_mlflow.uris == ["sqlite:///test.db"]
    assert fake_mlflow.experiments == ["github-stars-regression-api"]
    assert fake_mlflow.run_names[0].startswith("predict-")
    assert ("api.endpoint", "predict") in fake_mlflow.tags
    assert fake_mlflow.params == [{"model_name": "github-stars-regressor", "model_version": 7}]
    assert fake_mlflow.dicts[0][0] == "api/predict.json"
    event = fake_mlflow.dicts[0][1]
    assert event["request_id"] == outcome.request_id
    assert event["prediction"] == 8000.5
    assert event["request"]["has_coc"] is True
    assert fake_mlflow.files[0][0:2] == ("predictions.csv", "monitoring")
    monitoring = list(csv.DictReader(fake_mlflow.files[0][2].splitlines()))
    assert len(monitoring) == 1
    assert monitoring[0]["prediction"] == "8000.5"
    assert monitoring[0]["has_coc"] == "1"
    assert monitoring[0]["model_version"] == "7"


def test_prediction_with_real_sklearn_pipeline(
    payload: PredictRequest, sample_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch, fake_mlflow: FakeMLflow,
) -> None:
    import src.api_service as service

    training_frame = validate_dataset(sample_dataset)
    features, target = split_features_target(training_frame)
    model = build_pipeline("random_forest", random_state=42).fit(features, target)
    registry(monkeypatch, version=3)
    monkeypatch.setattr(service, "load_sklearn_model", lambda model_uri: model)
    outcome = predict_stars(payload, ApiSettings())
    assert outcome.prediction > 0
    assert outcome.model_version == 3
    assert fake_mlflow.files[0][1] == "monitoring"


@pytest.mark.parametrize("bad", [None, "0", "-1"])
def test_missing_or_invalid_champion(bad: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    registry(monkeypatch, version=bad)
    with pytest.raises(ChampionUnavailable):
        _champion_version("a-regressor")


@pytest.mark.parametrize("values", [[1.0, 2.0], [float("nan")], [-0.5], [float("inf")]])
def test_invalid_prediction_fails_without_logging(
    values: list[float], monkeypatch: pytest.MonkeyPatch, payload: PredictRequest,
    fake_mlflow: FakeMLflow,
) -> None:
    import src.api_service as service

    registry(monkeypatch, version=4)
    monkeypatch.setattr(service, "load_sklearn_model", lambda model_uri: SimpleNamespace(
        predict=lambda features: values))
    with pytest.raises(InvalidPrediction):
        predict_stars(payload, ApiSettings())
    assert fake_mlflow.run_names == []


def test_missing_model_fails_without_logging(monkeypatch: pytest.MonkeyPatch,
                                             payload: PredictRequest, fake_mlflow: FakeMLflow) -> None:
    import src.api_service as service

    registry(monkeypatch, version=4)
    monkeypatch.setattr(service, "load_sklearn_model", lambda model_uri: None)
    with pytest.raises(InvalidPrediction, match="no model"):
        predict_stars(payload, ApiSettings())
    assert not fake_mlflow.run_names


def test_missing_registered_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.api_service as service

    def missing(name: str) -> object:
        raise MlflowException("Model not found", error_code=RESOURCE_DOES_NOT_EXIST)

    monkeypatch.setattr(service, "MlflowClient", lambda: SimpleNamespace(get_registered_model=missing))
    with pytest.raises(ChampionUnavailable, match="No registered model"):
        _champion_version("a-regressor")


def test_registry_outage_is_not_treated_as_missing_champion(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.api_service as service

    def broken(name: str) -> object:
        raise MlflowException("Registry down", error_code=INTERNAL_ERROR)

    monkeypatch.setattr(service, "MlflowClient", lambda: SimpleNamespace(get_registered_model=broken))
    with pytest.raises(MlflowException, match="Registry down"):
        _champion_version("a-regressor")


def training_runs() -> list[TrainingRun]:
    return [
        TrainingRun("baseline", "run-1", {"mae": 6.0, "rmse": 8.0, "r2": 0.1}),
        TrainingRun("random_forest", "run-2", {"mae": 4.0, "rmse": 7.0, "r2": 0.2}, 13),
    ]


@pytest.mark.parametrize("registry_version", [13, "13"])
def test_train_with_promotion(
    monkeypatch: pytest.MonkeyPatch,
    fake_mlflow: FakeMLflow,
    registry_version: int | str,
) -> None:
    import src.api_service as service

    calls: list[tuple[str, dict[str, object]]] = []
    decision = PromotionDecision(True, "passed", 13, 12, {"mae": 4.0}, {"mae": 6.0}, {"mae": 5.0})

    def fake_train(**kwargs: object) -> list[TrainingRun]:
        calls.append(("train", kwargs))
        runs = training_runs()
        # MLflow can supply "13" at runtime despite the dataclass annotation.
        challenger = runs[1]
        runs[1] = TrainingRun(
            challenger.kind,
            challenger.run_id,
            challenger.metrics,
            cast(int, registry_version),
        )
        return runs

    def fake_promote(**kwargs: object) -> PromotionDecision:
        calls.append(("promote", kwargs))
        return decision

    monkeypatch.setattr(service, "train_experiment", fake_train)
    monkeypatch.setattr(service, "promote_candidate", fake_promote)
    monkeypatch.setattr(service, "log_promotion_decision", lambda run_id, item: calls.append(
        ("log_promotion", {"run_id": run_id, "accepted": item.accepted})))
    settings = ApiSettings(train_path=Path("train.csv"), test_path=Path("test.csv"))
    response = train_models(TrainRequest(), settings)
    assert [item[0] for item in calls] == ["train", "promote", "log_promotion"]
    assert calls[0][1]["train_path"] == Path("train.csv")
    assert calls[1][1]["candidate_version"] == 13
    assert isinstance(calls[1][1]["candidate_version"], int)
    assert calls[1][1]["min_relative_improvement"] == 0.01
    assert response.baseline_run_id == "run-1"
    assert response.promotion_accepted is True
    assert response.promotion_reason == "passed"
    assert response.registered_model_version == 13
    assert response.audit_run_id == "audit-1"
    assert fake_mlflow.dicts[0][0] == "api/train.json"
    assert fake_mlflow.dicts[0][1]["promotion"]["previous_champion_version"] == 12
    assert fake_mlflow.tags == [("api.endpoint", "train")]
    assert fake_mlflow.uris == ["sqlite:///mlflow.db"]


def test_train_without_promotion(monkeypatch: pytest.MonkeyPatch, fake_mlflow: FakeMLflow) -> None:
    import src.api_service as service

    monkeypatch.setattr(service, "train_experiment", lambda **kwargs: training_runs())
    monkeypatch.setattr(service, "promote_candidate", lambda **kwargs: pytest.fail("must not promote"))
    response = train_models(TrainRequest(auto_promote=False, random_state=9), ApiSettings())
    assert response.promotion_accepted is None
    assert response.promotion_reason is None
    assert fake_mlflow.dicts[0][1]["promotion"] is None
    assert fake_mlflow.dicts[0][1]["request"]["random_state"] == 9


@pytest.mark.parametrize("runs", [[], [TrainingRun("baseline", "a", {})],
                                      [TrainingRun("random_forest", "b", {}, None)]])
def test_incomplete_training_results(runs: list[TrainingRun]) -> None:
    with pytest.raises(RuntimeError, match="baseline and registered challenger"):
        _select_training_runs(runs)


def test_training_event_artifact_no_decision(fake_mlflow: FakeMLflow) -> None:
    baseline, challenger = _select_training_runs(training_runs())
    run_id = _log_training_event(ApiSettings(), TrainRequest(), baseline, challenger, None)
    assert run_id == "audit-1"
    assert fake_mlflow.dicts[0][1]["promotion"] is None


def test_env_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLSECOPS_EXPERIMENT", "custom-experiment")
    monkeypatch.setenv("MLSECOPS_MODEL_NAME", "custom-regressor")
    monkeypatch.setenv("MLSECOPS_TRAIN_CSV", "custom-train.csv")
    monkeypatch.setenv("MLSECOPS_TEST_CSV", "custom-test.csv")
    monkeypatch.setenv("MLSECOPS_TRAIN_API_KEY", "my-key")
    settings = ApiSettings.from_env()
    assert settings.tracking_uri == "http://localhost:5000"
    assert settings.experiment_name == "custom-experiment"
    assert settings.model_name == "custom-regressor"
    assert settings.train_path == Path("custom-train.csv")
    assert settings.test_path == Path("custom-test.csv")
    assert settings.train_api_key == "my-key"


def test_train_event_is_json_serializable(fake_mlflow: FakeMLflow) -> None:
    baseline, challenger = _select_training_runs(training_runs())
    _log_training_event(ApiSettings(), TrainRequest(), baseline, challenger, None)
    assert json.loads(json.dumps(fake_mlflow.dicts[0][1]))["challenger"]["registered_model_version"] == 13


def validation_registry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    alias: str | None = "9",
    tags: dict[str, str] | None = None,
) -> None:
    """Simulate the two registry reads required before validated serving."""
    import src.api_service as service

    version_aliases = {} if alias is None else {"champion": alias}
    model_tags = tags if tags is not None else {
        "validation.gate_decision": "accepted",
        "validation.protocol_id": "a" * 64,
    }

    class RegistryClient:
        def get_registered_model(self, model_name: str) -> SimpleNamespace:
            assert model_name == "github-stars-regressor-validation-v1"
            return SimpleNamespace(aliases=version_aliases)

        def get_model_version(self, model_name: str, version: str) -> SimpleNamespace:
            assert model_name == "github-stars-regressor-validation-v1"
            assert version == alias
            return SimpleNamespace(tags=model_tags)

    monkeypatch.setattr(service, "MlflowClient", RegistryClient)


def test_validated_champion_predicted_and_audited(
    payload: PredictRequest, monkeypatch: pytest.MonkeyPatch, fake_mlflow: FakeMLflow,
) -> None:
    import src.api_service as service

    validation_registry(monkeypatch)
    loaded_uris: list[str] = []

    def load(model_uri: str) -> SimpleNamespace:
        loaded_uris.append(model_uri)
        return SimpleNamespace(predict=lambda data: np.array([460.5]))

    monkeypatch.setattr(service, "load_sklearn_model", load)
    settings = ApiSettings(serving_model_name="github-stars-regressor-validation-v1")
    response = predict_stars(payload, settings)
    assert response.model_name == "github-stars-regressor-validation-v1"
    assert response.model_version == 9
    assert loaded_uris == ["models:/github-stars-regressor-validation-v1/9"]
    assert fake_mlflow.params[0]["model_name"] == "github-stars-regressor-validation-v1"
    assert fake_mlflow.dicts[0][1]["model_name"] == "github-stars-regressor-validation-v1"
    assert fake_mlflow.dicts[0][1]["validation_protocol_id"] == "a" * 64
    assert ("validation.protocol_id", "a" * 64) in fake_mlflow.tags


def test_serving_status_shows_both_registries(
    monkeypatch: pytest.MonkeyPatch, fake_mlflow: FakeMLflow,
) -> None:
    validation_registry(monkeypatch)
    response = model_status(ApiSettings(serving_model_name="github-stars-regressor-validation-v1"))
    assert response.serving_model_name == "github-stars-regressor-validation-v1"
    assert response.serving_model_version == 9
    assert response.training_model_name == "github-stars-regressor"
    assert response.validation_protocol_id == "a" * 64
    assert fake_mlflow.uris == ["sqlite:///mlflow.db"]
    assert not fake_mlflow.run_names


def test_legacy_status_retains_default(monkeypatch: pytest.MonkeyPatch, fake_mlflow: FakeMLflow) -> None:
    registry(monkeypatch, version="3")
    response = model_status(ApiSettings())
    assert response.serving_model_name == "github-stars-regressor"
    assert response.serving_model_version == 3
    assert response.training_model_name == "github-stars-regressor"
    assert response.validation_protocol_id is None


@pytest.mark.parametrize("invalid_tags", [
    {"validation.protocol_id": "b" * 64},
    {"validation.gate_decision": "rejected", "validation.protocol_id": "b" * 64},
    {"validation.gate_decision": "accepted", "validation.protocol_id": "bad"},
    {"validation.gate_decision": "accepted", "validation.protocol_id": "G" * 64},
    {"validation.gate_decision": "accepted"},
])
def test_unapproved_candidate_never_loaded(
    invalid_tags: dict[str, str], payload: PredictRequest,
    monkeypatch: pytest.MonkeyPatch, fake_mlflow: FakeMLflow,
) -> None:
    import src.api_service as service

    validation_registry(monkeypatch, tags=invalid_tags)
    monkeypatch.setattr(service, "load_sklearn_model", lambda model_uri: pytest.fail("unsafe load"))
    with pytest.raises(ChampionUnavailable, match="lacks an accepted validation"):
        predict_stars(payload, ApiSettings(serving_model_name="github-stars-regressor-validation-v1"))
    assert not fake_mlflow.run_names


def test_selected_registry_without_champion_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch, fake_mlflow: FakeMLflow,
) -> None:
    validation_registry(monkeypatch, alias=None)
    with pytest.raises(ChampionUnavailable, match="No champion alias"):
        model_status(ApiSettings(serving_model_name="github-stars-regressor-validation-v1"))
    assert not fake_mlflow.run_names


def test_serving_env_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLSECOPS_SERVING_MODEL_NAME", raising=False)
    assert ApiSettings.from_env().serving_model_name is None
    monkeypatch.setenv("MLSECOPS_SERVING_MODEL_NAME", " github-stars-regressor-validation-v1 ")
    assert ApiSettings.from_env().serving_model_name == "github-stars-regressor-validation-v1"
    monkeypatch.setenv("MLSECOPS_SERVING_MODEL_NAME", "  ")
    assert ApiSettings.from_env().serving_model_name is None


def test_api_train_remains_on_legacy_registry(
    monkeypatch: pytest.MonkeyPatch, fake_mlflow: FakeMLflow,
) -> None:
    import src.api_service as service

    observed: list[str] = []

    def fake_train(**kwargs: object) -> list[TrainingRun]:
        observed.append(str(kwargs["registered_model_name"]))
        return training_runs()

    monkeypatch.setattr(service, "train_experiment", fake_train)
    train_models(
        TrainRequest(auto_promote=False),
        ApiSettings(serving_model_name="github-stars-regressor-validation-v1"),
    )
    assert observed == ["github-stars-regressor"]


def test_invalid_legacy_alias_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    registry(monkeypatch, version="not-an-integer")
    with pytest.raises(ChampionUnavailable, match="invalid model version"):
        _champion_version("github-stars-regressor")


def test_validated_serving_with_real_pipeline(
    payload: PredictRequest, sample_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch, fake_mlflow: FakeMLflow,
) -> None:
    """Approved validation registry can serve the actual sklearn pipeline."""
    import src.api_service as service

    clean = validate_dataset(sample_dataset)
    features, target = split_features_target(clean)
    fitted = build_pipeline("random_forest", random_state=42).fit(features, target)
    validation_registry(monkeypatch, tags={
        "validation.gate_decision": "accepted", "validation.protocol_id": "c" * 64,
    })
    monkeypatch.setattr(service, "load_sklearn_model", lambda model_uri: fitted)
    outcome = predict_stars(
        payload, ApiSettings(serving_model_name="github-stars-regressor-validation-v1"),
    )
    assert outcome.prediction > 0
    assert outcome.model_name == "github-stars-regressor-validation-v1"
    assert outcome.model_version == 9
    assert fake_mlflow.dicts[0][1]["validation_protocol_id"] == "c" * 64
