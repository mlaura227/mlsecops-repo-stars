"""Exercise MLflow behavior without using the network or depending on a server."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data import validate_dataset
from src.promotion import PromotionDecision
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
        self.logged_dicts: list[tuple[dict[str, object], str]] = []
        self.logged_tags: dict[str, str] = {}
        self.logged_single_metrics: dict[str, float] = {}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import mlflow

        monkeypatch.setattr(mlflow, "set_tracking_uri", self.set_tracking_uri)
        monkeypatch.setattr(mlflow, "set_experiment", self.set_experiment)
        monkeypatch.setattr(mlflow, "start_run", self.start_run)
        monkeypatch.setattr(mlflow, "log_params", self.log_params)
        monkeypatch.setattr(mlflow, "log_metrics", self.log_metrics)
        monkeypatch.setattr(mlflow, "log_artifact", self.log_artifact)
        monkeypatch.setattr(mlflow, "log_dict", self.log_dict, raising=False)
        monkeypatch.setattr(mlflow, "set_tag", self.set_tag, raising=False)
        monkeypatch.setattr(mlflow, "log_metric", self.log_metric, raising=False)
        monkeypatch.setattr(mlflow.sklearn, "log_model", self.log_model)

    def set_tracking_uri(self, uri: str) -> None:
        self.uri = uri

    def set_experiment(self, experiment: str) -> None:
        self.experiment = experiment

    def start_run(self, run_name: str | None = None, run_id: str | None = None) -> object:
        if run_id is not None:
            return nullcontext(SimpleNamespace(info=SimpleNamespace(run_id=run_id)))
        assert run_name is not None
        self.runs.append(run_name)
        return nullcontext(SimpleNamespace(info=SimpleNamespace(run_id=f"run-{len(self.runs)}")))

    def log_params(self, params: dict[str, object]) -> None:
        self.params.append(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.append(metrics)

    def log_artifact(self, path: str, artifact_path: str) -> None:
        self.artifacts.append((Path(path).name, artifact_path, Path(path).read_bytes()))

    def log_model(self, **kwargs: object) -> SimpleNamespace:
        self.models.append(kwargs)
        version = 5 if kwargs.get("registered_model_name") is not None else None
        return SimpleNamespace(registered_model_version=version)

    def log_dict(self, data: dict[str, object], artifact_file: str) -> None:
        self.logged_dicts.append((data, artifact_file))

    def set_tag(self, key: str, value: str) -> None:
        self.logged_tags[key] = value

    def log_metric(self, key: str, value: float) -> None:
        self.logged_single_metrics[key] = value


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
    assert [result.registered_model_version for result in results] == [None, 5]
    assert all("numpy.dtype" in row["skops_trusted_types"] for row in fake_mlflow.models)
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


def test_cli_auto_promote_logs_decision(
    fake_mlflow: FakeMLflow, csv_splits: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.train as training_cli

    train_path, test_path = csv_splits
    calls: list[dict[str, object]] = []

    def fake_promote(**kwargs: object) -> PromotionDecision:
        calls.append(kwargs)
        return PromotionDecision(
            accepted=True, reason="gate passed", candidate_version=5,
            previous_champion_version=None,
            candidate_metrics={"mae": 1.0, "rmse": 2.0},
            baseline_metrics={"mae": 3.0, "rmse": 4.0},
            champion_metrics=None,
        )

    monkeypatch.setattr(training_cli, "promote_candidate", fake_promote)
    monkeypatch.setattr(sys, "argv", [
        "train-model", "--train", str(train_path), "--test", str(test_path),
        "--auto-promote", "--min-mae-improvement", "0.01",
    ])
    main()
    assert calls[0]["candidate_version"] == 5
    assert calls[0]["min_relative_improvement"] == 0.01
    assert fake_mlflow.logged_tags["promotion.status"] == "accepted"
    assert fake_mlflow.logged_dicts[0][1] == "promotion/decision.json"
    assert fake_mlflow.logged_single_metrics["promotion.candidate_mae"] == 1.0
    assert "Promotion: ACCEPTED" in capsys.readouterr().out


def test_cli_rejects_missing_registered_version(
    fake_mlflow: FakeMLflow, csv_splits: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path, test_path = csv_splits

    def unregistered_model(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(registered_model_version=None)

    monkeypatch.setattr("mlflow.sklearn.log_model", unregistered_model)
    monkeypatch.setattr(sys, "argv", [
        "train-model", "--train", str(train_path), "--test", str(test_path), "--auto-promote",
    ])
    with pytest.raises(RuntimeError, match="not registered"):
        main()
    assert not fake_mlflow.logged_tags


def test_promotion_logging_includes_champion(
    fake_mlflow: FakeMLflow,
) -> None:
    from src.train import _log_promotion_decision

    decision = PromotionDecision(
        accepted=False, reason="no improvement", candidate_version=6,
        previous_champion_version=5,
        candidate_metrics={"mae": 2.5, "rmse": 4.0},
        baseline_metrics={"mae": 3.0, "rmse": 5.0},
        champion_metrics={"mae": 1.5, "rmse": 3.0},
    )
    _log_promotion_decision("run-4", decision)
    assert fake_mlflow.logged_tags["promotion.status"] == "rejected"
    assert fake_mlflow.logged_single_metrics["promotion.champion_mae"] == 1.5
