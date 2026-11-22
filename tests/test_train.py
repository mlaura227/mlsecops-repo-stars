"""Exercise MLflow behavior without using the network or depending on a server."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data import validate_dataset
from src.train import _sha256, main, train_experiment


class FakeMLflow:
    """Minimal run context and artifact recorder for unit tests."""

    def __init__(self) -> None:
        self.artifacts: list[tuple[str, str, bytes]] = []
        self.metrics: list[dict[str, float]] = []
        self.params: list[dict[str, object]] = []
        self.models: list[dict[str, object]] = []
        self.experiment = ""
        self.uri = ""
        self.runs: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import mlflow

        monkeypatch.setattr(mlflow, "set_tracking_uri", self.set_tracking_uri)
        monkeypatch.setattr(mlflow, "set_experiment", self.set_experiment)
        monkeypatch.setattr(mlflow, "start_run", self.start_run)
        monkeypatch.setattr(mlflow, "log_params", self.log_params)
        monkeypatch.setattr(mlflow, "log_metrics", self.log_metrics)
        monkeypatch.setattr(mlflow, "log_artifact", self.log_artifact)
        monkeypatch.setattr(mlflow.sklearn, "log_model", self.log_model)

    def set_tracking_uri(self, uri: str) -> None:
        self.uri = uri

    def set_experiment(self, experiment: str) -> None:
        self.experiment = experiment

    def start_run(self, run_name: str) -> object:
        self.runs.append(run_name)
        return nullcontext(SimpleNamespace(info=SimpleNamespace(run_id=f"run-{len(self.runs)}")))

    def log_params(self, params: dict[str, object]) -> None:
        self.params.append(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.append(metrics)

    def log_artifact(self, path: str, artifact_path: str) -> None:
        self.artifacts.append((Path(path).name, artifact_path, Path(path).read_bytes()))

    def log_model(self, **kwargs: object) -> None:
        self.models.append(kwargs)


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> FakeMLflow:
    fake = FakeMLflow()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def csv_splits(sample_dataset: pd.DataFrame, tmp_path: Path) -> tuple[Path, Path]:
    clean = validate_dataset(sample_dataset)
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    clean.iloc[:24].to_csv(train_file, index=False)
    clean.iloc[24:].to_csv(test_file, index=False)
    return train_file, test_file


def test_training_tracks_both_models(
    fake_mlflow: FakeMLflow, csv_splits: tuple[Path, Path]
) -> None:
    train_path, test_path = csv_splits
    results = train_experiment(
        train_path=train_path,
        test_path=test_path,
        tracking_uri="sqlite:///some-test.db",
        experiment_name="test-experiment",
        registered_model_name="test-regressor",
        random_state=9,
    )
    assert fake_mlflow.uri == "sqlite:///some-test.db"
    assert fake_mlflow.experiment == "test-experiment"
    assert fake_mlflow.runs == ["baseline-seed-9", "random_forest-seed-9"]
    assert len(results) == len(fake_mlflow.metrics) == len(fake_mlflow.params) == len(fake_mlflow.models) == 2
    assert [result.run_id for result in results] == ["run-1", "run-2"]
    assert fake_mlflow.models[0]["registered_model_name"] is None
    assert fake_mlflow.models[1]["registered_model_name"] == "test-regressor"
    assert all(row["train_rows"] == 24 and row["test_rows"] == 8 for row in fake_mlflow.params)
    assert all(len(row) == 64 for row in [fake_mlflow.params[0]["train_sha256"]])
    assert len(fake_mlflow.artifacts) == 4
    manifest = json.loads(fake_mlflow.artifacts[1][2])
    assert manifest["train_sha256"] == _sha256(train_path)
    assert manifest["test_sha256"] == _sha256(test_path)
    assert b"prediction" in fake_mlflow.artifacts[0][2]


def test_same_path_guard(fake_mlflow: FakeMLflow, csv_splits: tuple[Path, Path]) -> None:
    train_path, _ = csv_splits
    with pytest.raises(ValueError, match="must differ"):
        train_experiment(train_path=train_path, test_path=train_path)
    assert not fake_mlflow.runs


def test_cli(fake_mlflow: FakeMLflow, csv_splits: tuple[Path, Path],
             monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    train_path, test_path = csv_splits
    monkeypatch.setattr(
        sys, "argv", ["train-model", "--train", str(train_path), "--test", str(test_path),
                      "--tracking-uri", "sqlite:///cli.db", "--experiment", "cli-test",
                      "--registered-model", "cli-regressor", "--seed", "7"]
    )
    main()
    assert "run_id=run-1" in capsys.readouterr().out
    assert fake_mlflow.uri == "sqlite:///cli.db"
    assert fake_mlflow.models[1]["registered_model_name"] == "cli-regressor"
