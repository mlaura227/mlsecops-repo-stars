"""Promotion safety tests: bootstrap, regression guard, and registry errors."""

from pathlib import Path
from types import SimpleNamespace

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from src.data import validate_dataset
from src.promotion import (
    _beats_reference,
    _validate_metrics,
    promote_candidate,
    score_registered_version,
)


class FakeClient:
    """In-memory registry with optional champion and auditable writes."""

    def __init__(self, champion: int | None = None, error_code: str | None = None) -> None:
        self.champion = champion
        self.error_code = error_code
        self.alias_updates: list[tuple[str, str, str]] = []
        self.tags: list[tuple[str, str, str, str]] = []
        self.lookups: list[tuple[str, str]] = []

    def get_model_version(self, name: str, version: str) -> object:
        self.lookups.append((name, version))
        return SimpleNamespace(version=version)

    def get_registered_model(self, name: str) -> SimpleNamespace:
        if self.error_code is not None:
            raise MlflowException("Registry unavailable", error_code=self.error_code)
        aliases = {"champion": str(self.champion)} if self.champion is not None else {}
        return SimpleNamespace(aliases=aliases)

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.tags.append((name, version, key, value))

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.alias_updates.append((name, alias, version))
        self.champion = int(version)


@pytest.fixture
def evaluation_csv(sample_dataset: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "test.csv"
    validate_dataset(sample_dataset).to_csv(path, index=False)
    return path


def setup_gate(
    monkeypatch: pytest.MonkeyPatch, client: FakeClient, predictions: dict[int, list[float]]
) -> None:
    import src.promotion as promotion

    monkeypatch.setattr(promotion, "MlflowClient", lambda: client)
    monkeypatch.setattr(promotion.mlflow, "set_tracking_uri", lambda uri: None)

    def load_model(model_uri: str) -> object:
        version = int(model_uri.rsplit("/", 1)[-1])
        return SimpleNamespace(predict=lambda features: np.array(predictions[version]))

    monkeypatch.setattr(mlflow.sklearn, "load_model", load_model)


def baseline() -> dict[str, float]:
    return {"mae": 8.0, "rmse": 8.0}


def test_accepts_first_challenger(
    evaluation_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.read_csv(evaluation_csv)
    actual = frame["stars"].to_numpy()
    client = FakeClient()
    setup_gate(monkeypatch, client, {2: list(actual + 1)})
    decision = promote_candidate(candidate_version=2, baseline_metrics=baseline(), test_path=evaluation_csv)
    assert decision.accepted
    assert decision.previous_champion_version is None
    assert client.alias_updates == [("github-stars-regressor", "champion", "2")]
    assert client.tags[0][-1] == "accepted"
    assert decision.candidate_metrics["mae"] == 1.0


def test_rejects_weak_bootstrap(
    evaluation_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = pd.read_csv(evaluation_csv)["stars"].to_numpy()
    client = FakeClient()
    setup_gate(monkeypatch, client, {3: list(actual + 10)})
    decision = promote_candidate(candidate_version=3, baseline_metrics=baseline(), test_path=evaluation_csv)
    assert not decision.accepted
    assert "baseline" in decision.reason
    assert not client.alias_updates
    assert client.tags[0][-1] == "rejected"


def test_accepts_improved_challenger_over_champion(
    evaluation_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = pd.read_csv(evaluation_csv)["stars"].to_numpy()
    client = FakeClient(champion=1)
    setup_gate(monkeypatch, client, {1: list(actual + 3), 2: list(actual + 1)})
    decision = promote_candidate(candidate_version=2, baseline_metrics=baseline(), test_path=evaluation_csv)
    assert decision.accepted
    assert decision.previous_champion_version == 1
    assert decision.champion_metrics == {"mae": 3.0, "rmse": 3.0}
    assert client.champion == 2


def test_rejects_regression_without_alias_update(
    evaluation_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = pd.read_csv(evaluation_csv)["stars"].to_numpy()
    client = FakeClient(champion=1)
    setup_gate(monkeypatch, client, {1: list(actual + 1), 2: list(actual + 3)})
    decision = promote_candidate(candidate_version=2, baseline_metrics=baseline(), test_path=evaluation_csv)
    assert not decision.accepted
    assert "champion" in decision.reason
    assert client.champion == 1
    assert not client.alias_updates


def test_same_version_is_unchanged(
    evaluation_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = pd.read_csv(evaluation_csv)["stars"].to_numpy()
    client = FakeClient(champion=2)
    setup_gate(monkeypatch, client, {2: list(actual + 1)})
    decision = promote_candidate(candidate_version=2, baseline_metrics=baseline(), test_path=evaluation_csv)
    assert not decision.accepted
    assert "already champion" in decision.reason
    assert client.alias_updates == []


@pytest.mark.parametrize(
    "error_code", ["INTERNAL_ERROR", "RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"]
)
def test_registry_error_fails_closed(
    evaluation_csv: Path, monkeypatch: pytest.MonkeyPatch, error_code: str
) -> None:
    actual = pd.read_csv(evaluation_csv)["stars"].to_numpy()
    client = FakeClient(error_code=error_code)
    setup_gate(monkeypatch, client, {2: list(actual + 1)})
    with pytest.raises(MlflowException):
        promote_candidate(candidate_version=2, baseline_metrics=baseline(), test_path=evaluation_csv)
    assert not client.alias_updates


def test_bad_predictions_fail_closed(
    evaluation_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient()
    setup_gate(monkeypatch, client, {2: [float("nan")] * 32})
    with pytest.raises(ValueError, match="invalid predictions"):
        promote_candidate(candidate_version=2, baseline_metrics=baseline(), test_path=evaluation_csv)
    assert not client.alias_updates


@pytest.mark.parametrize("prediction", [[], [1.0]])
def test_prediction_length_checked(
    evaluation_csv: Path, monkeypatch: pytest.MonkeyPatch, prediction: list[float]
) -> None:
    setup_gate(monkeypatch, FakeClient(), {2: prediction})
    with pytest.raises(ValueError, match="invalid predictions"):
        score_registered_version("github-stars-regressor", 2, pd.read_csv(evaluation_csv))


def test_small_holdout_fails(monkeypatch: pytest.MonkeyPatch, sample_dataset: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least two"):
        score_registered_version("github-stars-regressor", 2, sample_dataset.iloc[:1])


@pytest.mark.parametrize("metric", ["mae", "rmse"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -2.0])
def test_invalid_metrics(metric: str, invalid: float) -> None:
    value = baseline()
    value[metric] = invalid
    with pytest.raises(ValueError, match="Invalid"):
        _validate_metrics(value)


def test_missing_metric() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        _validate_metrics({"mae": 1.0})


@pytest.mark.parametrize(
    ("reference", "candidate", "improvement", "expected"), [
        ({"mae": 0.0, "rmse": 0.0}, {"mae": 0.0, "rmse": 0.0}, 0.0, False),
        ({"mae": 4.0, "rmse": 8.0}, {"mae": 4.0, "rmse": 6.0}, 0.0, False),
        ({"mae": 4.0, "rmse": 8.0}, {"mae": 3.0, "rmse": 9.0}, 0.0, False),
        ({"mae": 4.0, "rmse": 8.0}, {"mae": 3.96, "rmse": 6.0}, 0.02, False),
        ({"mae": 4.0, "rmse": 8.0}, {"mae": 3.0, "rmse": 8.0}, 0.20, True),
    ]
)
def test_beat_reference_cases(
    reference: dict[str, float], candidate: dict[str, float], improvement: float, expected: bool
) -> None:
    assert _beats_reference(candidate, reference, improvement) is expected


@pytest.mark.parametrize(
    ("version", "name", "improvement", "metrics"), [
        (0, "good-model", 0.0, baseline()),
        (1, "   ", 0.0, baseline()),
        (1, "good-model", 1.0, baseline()),
        (1, "good-model", -0.1, baseline()),
        (1, "good-model", 0.0, {"mae": -1.0, "rmse": 1.0}),
    ]
)
def test_invalid_gate_config(
    version: int, name: str, improvement: float, metrics: dict[str, float],
    evaluation_csv: Path,
) -> None:
    with pytest.raises(ValueError):
        promote_candidate(candidate_version=version, baseline_metrics=metrics, test_path=evaluation_csv,
                          model_name=name, min_relative_improvement=improvement)
