"""Unit and functional tests for CV tuning and split isolation."""

import io
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.tuning as tuning
from src.data import validate_dataset

FAST_GRID: dict[str, list[int | float]] = {
    "regressor__n_estimators": [6],
    "regressor__min_samples_leaf": [2],
    "regressor__max_features": [1.0],
}


class FakeTracking:
    """Records logged artifacts while leaving real sklearn fitting enabled."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.params: dict[str, object] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: dict[str, bytes] = {}
        self.models: list[dict[str, object]] = []
        self.tags: dict[str, str] = {}
        self.experiment = ""
        self.uri = ""
        monkeypatch.setattr(tuning.mlflow, "set_tracking_uri", self.set_uri)
        monkeypatch.setattr(tuning.mlflow, "set_experiment", self.set_experiment)
        monkeypatch.setattr(tuning.mlflow, "start_run", self.start_run)
        monkeypatch.setattr(tuning.mlflow, "log_params", self.log_params)
        monkeypatch.setattr(tuning.mlflow, "log_metrics", self.log_metrics)
        monkeypatch.setattr(tuning.mlflow, "set_tag", self.set_tag)
        monkeypatch.setattr(tuning.mlflow, "log_artifact", self.log_artifact)
        monkeypatch.setattr(tuning, "log_sklearn_model", self.log_model)

    def set_uri(self, uri: str) -> None:
        self.uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, **kwargs: object) -> object:
        return nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="tuning-run")))

    def log_params(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def log_artifact(self, path: str, artifact_path: str) -> None:
        assert artifact_path == "tuning"
        self.artifacts[Path(path).name] = Path(path).read_bytes()

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_model(self, **kwargs: object) -> object:
        self.models.append(kwargs)
        return object()


@pytest.fixture
def fake_tracking(monkeypatch: pytest.MonkeyPatch) -> FakeTracking:
    return FakeTracking(monkeypatch)


@pytest.fixture
def train_file(sample_dataset: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "train.csv"
    validate_dataset(sample_dataset).to_csv(path, index=False)
    return path


def test_tuning_produces_reproducible_validation_only_evidence(
    fake_tracking: FakeTracking, train_file: Path
) -> None:
    first = tuning.tune_experiment(
        train_path=train_file, tracking_uri="sqlite:///test.db", random_state=8, param_grid=FAST_GRID
    )
    manifest = json.loads(fake_tracking.artifacts["tuning_manifest.json"])
    saved_predictions = pd.read_csv(
        io.BytesIO(fake_tracking.artifacts["validation_predictions.csv"])
    )
    assert first.training_rows + first.validation_rows == 32
    assert first.run_id == "tuning-run"
    assert first.cv_best_mae >= 0
    assert first.validation_metrics["mae"] >= 0
    assert fake_tracking.experiment == "github-stars-tuning"
    assert fake_tracking.uri == "sqlite:///test.db"
    assert manifest["final_test_used_for_tuning"] is False
    assert manifest["registered_model"] is False
    assert len(manifest["validation_row_indices"]) == first.validation_rows
    assert saved_predictions.shape[0] == first.validation_rows
    assert {"stars", "prediction"}.issubset(saved_predictions.columns)
    assert "cv_results.csv" in fake_tracking.artifacts
    assert fake_tracking.tags["model_stage"] == "research_only_not_promoted"
    assert "registered_model_name" not in fake_tracking.models[0]
    assert "numpy.dtype" in fake_tracking.models[0]["skops_trusted_types"]

    second = tuning.tune_experiment(train_path=train_file, random_state=8, param_grid=FAST_GRID)
    assert first.best_params == second.best_params
    assert first.validation_metrics == second.validation_metrics
    assert first.cv_best_mae == second.cv_best_mae


def test_rejects_invalid_input_before_tracking(
    fake_tracking: FakeTracking, train_file: Path, sample_dataset: pd.DataFrame, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        tuning.tune_experiment(train_path=train_file, validation_fraction=0.0)
    with pytest.raises(ValueError, match="experiment_name"):
        tuning.tune_experiment(train_path=train_file, experiment_name=" ")
    with pytest.raises(ValueError, match="param_grid"):
        tuning.tune_experiment(train_path=train_file, param_grid={})
    with pytest.raises(ValueError, match="param_grid"):
        tuning.tune_experiment(train_path=train_file, param_grid={"regressor__n_estimators": []})
    small = tmp_path / "small.csv"
    validate_dataset(sample_dataset).iloc[:10].to_csv(small, index=False)
    with pytest.raises(ValueError, match="30 training"):
        tuning.tune_experiment(train_path=small)
    assert not fake_tracking.models


def test_metrics_reject_invalid_shapes_and_nonfinite() -> None:
    observed = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="invalid validation predictions"):
        tuning._metrics(observed, np.asarray([1.0]))
    with pytest.raises(ValueError, match="invalid validation predictions"):
        tuning._metrics(observed, np.asarray([1.0, np.inf]))
    assert tuning._metrics(observed, np.asarray([1.0, 3.0]))["mae"] == 0.5


def test_rejects_non_numeric_hyperparameters() -> None:
    with pytest.raises(TypeError, match="Unexpected"):
        tuning._normalize_params({"a": "b"})
    with pytest.raises(TypeError, match="Unexpected"):
        tuning._normalize_params({"a": True})


def test_cli(fake_tracking: FakeTracking, train_file: Path, monkeypatch: pytest.MonkeyPatch,
             capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(tuning, "DEFAULT_GRID", FAST_GRID)
    monkeypatch.setattr(sys, "argv", [
        "python -m src.tuning", "--train", str(train_file),
        "--tracking-uri", "sqlite:///cli.db", "--seed", "8",
        "--validation-fraction", "0.25",
    ])
    tuning.main()
    output = json.loads(capsys.readouterr().out)
    assert output["run_id"] == "tuning-run"
    assert output["training_rows"] == 24
    assert output["validation_rows"] == 8
    assert fake_tracking.uri == "sqlite:///cli.db"
