"""End-to-end validation training with sklearn and simulated MLflow I/O."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.validation_training as training
from src.data import validate_dataset
from src.validation_promotion import MODEL_NAME, PROTOCOL_TAG, ValidationDecision

FAST_GRID: dict[str, list[int | float]] = {
    "regressor__n_estimators": [5],
    "regressor__min_samples_leaf": [1],
    "regressor__max_features": [1.0],
}


class FakeTracking:
    """Track artifacts and model registry tags without real MLflow server."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.uri = ""
        self.experiment = ""
        self.artifacts: dict[str, bytes] = {}
        self.params: dict[str, object] = {}
        self.metrics: dict[str, float] = {}
        self.tags: dict[str, str] = {}
        self.saved_models: list[dict[str, object]] = []
        self.model_version: str | None = "14"
        self.registry_tags: list[tuple[str, str, str, str]] = []
        self.dicts: list[tuple[str, dict[str, object]]] = []
        self.run_sessions: list[dict[str, object]] = []

        monkeypatch.setattr(training.mlflow, "set_tracking_uri", self.set_tracking_uri)
        monkeypatch.setattr(training.mlflow, "set_experiment", self.set_experiment)
        monkeypatch.setattr(training.mlflow, "start_run", self.start_run)
        monkeypatch.setattr(training.mlflow, "log_params", self.log_params)
        monkeypatch.setattr(training.mlflow, "log_metrics", self.log_metrics)
        monkeypatch.setattr(training.mlflow, "set_tag", self.set_tag)
        monkeypatch.setattr(training.mlflow, "log_artifact", self.log_artifact)
        monkeypatch.setattr(training.mlflow, "log_dict", self.log_dict)
        monkeypatch.setattr(training, "log_sklearn_model", self.log_model)
        monkeypatch.setattr(training, "MlflowClient", lambda: self)

    def set_tracking_uri(self, uri: str) -> None:
        self.uri = uri

    def set_experiment(self, experiment: str) -> None:
        self.experiment = experiment

    def start_run(self, **kwargs: object) -> object:
        self.run_sessions.append(kwargs)
        return nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="validation-run")))

    def log_params(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_artifact(self, path: str, artifact_path: str) -> None:
        assert artifact_path == "validation"
        self.artifacts[Path(path).name] = Path(path).read_bytes()

    def log_dict(self, contents: dict[str, object], artifact_file: str) -> None:
        self.dicts.append((artifact_file, contents))

    def log_model(self, **kwargs: object) -> SimpleNamespace:
        self.saved_models.append(kwargs)
        return SimpleNamespace(registered_model_version=self.model_version)

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.registry_tags.append((name, version, key, value))


@pytest.fixture
def fake_tracking(monkeypatch: pytest.MonkeyPatch) -> FakeTracking:
    return FakeTracking(monkeypatch)


@pytest.fixture
def train_file(sample_dataset: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "train.csv"
    validate_dataset(sample_dataset).to_csv(path, index=False)
    # The final test is deliberately invalid: no code may read it.
    (tmp_path / "test.csv").write_text("SHOULD NEVER BE READ", encoding="utf-8")
    return path


def test_training_logs_protocol_and_sklearn_model_without_touching_final_test(
    train_file: Path, fake_tracking: FakeTracking,
) -> None:
    result = training.train_validation_candidate(
        train_path=train_file, tracking_uri="http://mlflow:5000",
        param_grid=FAST_GRID,
    )
    assert result.run_id == "validation-run"
    assert result.model_version == 14
    assert result.decision is None
    assert result.fit_rows + result.validation_rows == 32
    assert result.fit_rows == 25
    assert result.validation_rows == 7
    assert len(result.protocol_id) == 64
    assert fake_tracking.uri == "http://mlflow:5000"
    assert fake_tracking.experiment == "github-stars-validation-training"
    assert fake_tracking.params["protocol_id"] == result.protocol_id
    assert fake_tracking.metrics["validation_mae"] == result.candidate_metrics["mae"]
    assert fake_tracking.tags["data.final_test_used_for_selection"] == "false"
    assert fake_tracking.registry_tags == [(MODEL_NAME, "14", PROTOCOL_TAG, result.protocol_id)]
    assert fake_tracking.saved_models[0]["registered_model_name"] == MODEL_NAME
    assert "sklearn.tree._tree.Tree" in fake_tracking.saved_models[0]["skops_trusted_types"]
    assert {"validation_predictions.csv", "cv_results.csv", "validation_protocol.json"} == set(
        fake_tracking.artifacts
    )
    manifest = json.loads(fake_tracking.artifacts["validation_protocol.json"])
    assert manifest["final_test_used_for_selection"] is False
    assert manifest["protocol_id"] == result.protocol_id


def test_retraining_is_reproducible(train_file: Path, fake_tracking: FakeTracking) -> None:
    first = training.train_validation_candidate(train_path=train_file, param_grid=FAST_GRID)
    second = training.train_validation_candidate(train_path=train_file, param_grid=FAST_GRID)
    assert first.protocol_id == second.protocol_id
    assert first.best_params == second.best_params
    assert first.candidate_metrics == second.candidate_metrics
    assert first.baseline_metrics == second.baseline_metrics


def test_auto_promotion_logs_decision(
    train_file: Path, fake_tracking: FakeTracking, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def promote(**kwargs: object) -> ValidationDecision:
        captured.append(kwargs)
        return ValidationDecision(
            True, "accepted", 14, None, str(kwargs["protocol_id"]),
            {"mae": 2.0, "rmse": 3.0}, {"mae": 5.0, "rmse": 7.0}, None,
        )

    monkeypatch.setattr(training, "promote_validation_candidate", promote)
    result = training.train_validation_candidate(
        train_path=train_file, param_grid=FAST_GRID, auto_promote=True,
    )
    assert result.decision is not None and result.decision.accepted
    assert captured[0]["candidate_version"] == 14
    assert captured[0]["validation_frame"].shape[0] == 7
    assert captured[0]["protocol_id"] == result.protocol_id
    assert fake_tracking.dicts[0][0] == "validation/promotion_decision.json"
    assert fake_tracking.tags["validation.promotion_status"] == "accepted"
    assert fake_tracking.run_sessions[-1] == {"run_id": "validation-run"}


def test_auto_promotion_rejection_is_audited(
    train_file: Path, fake_tracking: FakeTracking, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "promote_validation_candidate", lambda **kwargs: ValidationDecision(
        False, "worse", 14, 8, str(kwargs["protocol_id"]),
        {"mae": 10.0, "rmse": 15.0}, {"mae": 5.0, "rmse": 7.0}, None,
    ))
    result = training.train_validation_candidate(train_path=train_file, param_grid=FAST_GRID, auto_promote=True)
    assert result.decision is not None and result.decision.accepted is False
    assert fake_tracking.tags["validation.promotion_status"] == "rejected"


def test_registration_without_version_fails_closed(
    train_file: Path, fake_tracking: FakeTracking,
) -> None:
    fake_tracking.model_version = None
    with pytest.raises(RuntimeError, match="did not register"):
        training.train_validation_candidate(train_path=train_file, param_grid=FAST_GRID)
    assert not fake_tracking.registry_tags


@pytest.mark.parametrize("kwargs, message", [
    ({"model_name": "github-stars-regressor"}, "non-legacy"),
    ({"experiment_name": " "}, "experiment_name"),
    ({"min_relative_improvement": -0.1}, "min_relative_improvement"),
    ({"param_grid": {}}, "param_grid"),
    ({"param_grid": {"x": []}}, "param_grid"),
])
def test_invalid_config_before_mlflow(
    fake_tracking: FakeTracking, train_file: Path, kwargs: dict[str, object], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        training.train_validation_candidate(train_path=train_file, **kwargs)
    assert not fake_tracking.saved_models


def test_unexpected_hyperparameter_type() -> None:
    with pytest.raises(TypeError, match="Unexpected hyperparameter"):
        training._best_params({"x": True})
    with pytest.raises(TypeError, match="Unexpected hyperparameter"):
        training._best_params({"x": "oops"})


def test_invalid_predictions() -> None:
    target = pd.Series([2.0, 3.0])
    with pytest.raises(ValueError, match="Invalid regression"):
        training._metrics(target, np.asarray([5.0], dtype=np.float64))
    with pytest.raises(ValueError, match="Invalid regression"):
        training._metrics(target, np.asarray([float("nan"), 2.0], dtype=np.float64))


def test_cli_help(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["validation_training", "--help"])
    with pytest.raises(SystemExit) as result:
        training.main()
    assert result.value.code == 0
    assert "auto-promote" in capsys.readouterr().out


def test_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake_result = training.ValidationTrainingResult(
        "some-run", 3, "a" * 64, 80, 20, {"mae": 1.0}, {"mae": 2.0}, {}, None,
    )
    monkeypatch.setattr(training, "train_validation_candidate", lambda **kwargs: fake_result)
    monkeypatch.setattr(sys, "argv", [
        "validation_training", "--train", "custom.csv", "--tracking-uri", "http://server:5000",
        "--model-name", "custom-validation", "--seed", "10", "--auto-promote",
        "--min-mae-improvement", "0.03",
    ])
    training.main()
    result = json.loads(capsys.readouterr().out)
    assert result["model_version"] == 3
    assert result["protocol_id"] == "a" * 64
