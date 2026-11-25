"""Fail-closed validation promotion and registry-compatibility regression tests."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.validation_promotion import (
    DECISION_TAG, MODEL_NAME, PROTOCOL_TAG, _beats,
    promote_validation_candidate, validate_registry_name,
)

PROTO = "c" * 64
OTHER = "d" * 64


class FakeRegistry:
    """Small in-memory model registry with immutable version tags."""

    def __init__(self, champion: int | None = None, *, champion_protocol: str = PROTO) -> None:
        self.aliases = {} if champion is None else {"champion": str(champion)}
        self.tags: dict[int, dict[str, str]] = {
            7: {PROTOCOL_TAG: PROTO},
            6: {PROTOCOL_TAG: champion_protocol},
        }
        self.modified: list[tuple[str, ...]] = []

    def get_model_version(self, model: str, version: str) -> SimpleNamespace:
        assert model == MODEL_NAME
        return SimpleNamespace(tags=self.tags[int(version)])

    def get_registered_model(self, model: str) -> SimpleNamespace:
        assert model == MODEL_NAME
        return SimpleNamespace(aliases=self.aliases)

    def set_model_version_tag(self, model: str, version: str, key: str, value: str) -> None:
        self.modified.append(("tag", version, key, value))
        self.tags[int(version)][key] = value

    def set_registered_model_alias(self, model: str, alias: str, version: str) -> None:
        self.modified.append(("alias", alias, version))
        self.aliases[alias] = version


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"dummy": [1, 2, 3]})


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> FakeRegistry:
    import src.validation_promotion as promotion

    state = FakeRegistry()
    monkeypatch.setattr(promotion, "MlflowClient", lambda: state)
    monkeypatch.setattr(promotion, "score_registered_version", lambda name, version, frame: {
        "mae": 3.0 if version == 7 else 4.0,
        "rmse": 5.0 if version == 7 else 6.0,
    })
    return state


def promote(frame: pd.DataFrame, **kwargs: object) -> object:
    return promote_validation_candidate(
        candidate_version=7,
        protocol_id=PROTO,
        baseline_metrics={"mae": 5.0, "rmse": 7.0},
        validation_frame=frame,
        **kwargs,
    )


def test_first_promotion_bootstraps_only_valid_candidate(
    registry: FakeRegistry, frame: pd.DataFrame,
) -> None:
    result = promote(frame)
    assert result.accepted is True
    assert result.previous_champion_version is None
    assert result.candidate_metrics == {"mae": 3.0, "rmse": 5.0}
    assert registry.aliases == {"champion": "7"}
    assert ("tag", "7", DECISION_TAG, "accepted") in registry.modified
    assert registry.modified[-1] == ("alias", "champion", "7")


def test_same_candidate_is_not_promoted_again(registry: FakeRegistry, frame: pd.DataFrame) -> None:
    registry.aliases["champion"] = "7"
    result = promote(frame)
    assert not result.accepted
    assert "already" in result.reason
    assert not registry.modified


def test_better_candidate_replaces_comparable_champion(
    registry: FakeRegistry, frame: pd.DataFrame,
) -> None:
    registry.aliases["champion"] = "6"
    result = promote(frame)
    assert result.accepted
    assert result.champion_metrics == {"mae": 4.0, "rmse": 6.0}
    assert registry.aliases == {"champion": "7"}


def test_worse_candidate_does_not_replace_champion(
    registry: FakeRegistry, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.validation_promotion as promotion

    registry.aliases["champion"] = "6"
    monkeypatch.setattr(promotion, "score_registered_version", lambda name, version, frame: {
        "mae": 4.5 if version == 7 else 4.0,
        "rmse": 5.5 if version == 7 else 5.0,
    })
    result = promote(frame)
    assert not result.accepted
    assert "champion" in result.reason
    assert registry.aliases == {"champion": "6"}
    assert not any(event[0] == "alias" for event in registry.modified)


def test_worse_than_baseline_rejected(
    registry: FakeRegistry, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.validation_promotion as promotion

    monkeypatch.setattr(promotion, "score_registered_version", lambda name, version, frame: {
        "mae": 6.0, "rmse": 8.0,
    })
    result = promote(frame)
    assert not result.accepted
    assert "baseline" in result.reason
    assert "champion" not in registry.aliases


def test_incompatible_champion_fails_closed(
    registry: FakeRegistry, frame: pd.DataFrame,
) -> None:
    registry.aliases["champion"] = "6"
    registry.tags[6][PROTOCOL_TAG] = OTHER
    result = promote(frame)
    assert not result.accepted
    assert "incompatible" in result.reason
    assert result.champion_metrics is None
    assert registry.aliases["champion"] == "6"
    assert not any(event[0] == "alias" for event in registry.modified)


def test_candidate_missing_protocol_tag_fails_closed(
    registry: FakeRegistry, frame: pd.DataFrame,
) -> None:
    registry.tags[7].clear()
    with pytest.raises(ValueError, match="Candidate validation protocol"):
        promote(frame)
    assert not registry.modified


def test_registry_or_artifact_error_does_not_change_alias(
    registry: FakeRegistry, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.validation_promotion as promotion

    def broken(*args: object) -> object:
        raise ConnectionError("registry offline")

    monkeypatch.setattr(promotion, "score_registered_version", broken)
    with pytest.raises(ConnectionError, match="registry offline"):
        promote(frame)
    assert registry.modified == []


@pytest.mark.parametrize("name", ["", " ", "github-stars-regressor"])
def test_legacy_name_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="non-legacy"):
        validate_registry_name(name)


@pytest.mark.parametrize("bad", [0, -1])
def test_version_validation(registry: FakeRegistry, frame: pd.DataFrame, bad: int) -> None:
    with pytest.raises(ValueError, match="candidate_version"):
        promote_validation_candidate(candidate_version=bad, protocol_id=PROTO,
                                     baseline_metrics={"mae": 5.0, "rmse": 7.0}, validation_frame=frame)


@pytest.mark.parametrize("bad", ["", "X" * 64, "0" * 63])
def test_protocol_id_validation(registry: FakeRegistry, frame: pd.DataFrame, bad: str) -> None:
    with pytest.raises(ValueError, match="protocol_id"):
        promote_validation_candidate(candidate_version=7, protocol_id=bad,
                                     baseline_metrics={"mae": 5.0, "rmse": 7.0}, validation_frame=frame)


def test_config_and_frame_validation(registry: FakeRegistry, frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="min_relative_improvement"):
        promote(frame, min_relative_improvement=-0.1)
    with pytest.raises(ValueError, match="at least two"):
        promote(frame.iloc[:1])


@pytest.mark.parametrize("metrics", [
    {"mae": -1.0, "rmse": 4.0}, {"mae": 1.0},
    {"mae": float("nan"), "rmse": 5.0}, {"mae": 1.0, "rmse": float("inf")},
])
def test_invalid_metrics(metrics: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="Invalid validation metric"):
        _beats(metrics, {"mae": 5.0, "rmse": 5.0}, 0.01)


def test_mae_rmse_and_minimum_gain_gates() -> None:
    candidate = {"mae": 4.9, "rmse": 5.0}
    assert _beats(candidate, {"mae": 5.0, "rmse": 5.0}, 0.01)
    assert not _beats(candidate, {"mae": 5.0, "rmse": 4.0}, 0.01)
    assert not _beats(candidate, {"mae": 5.0, "rmse": 5.0}, 0.03)
    assert not _beats(candidate, {"mae": 0.0, "rmse": 5.0}, 0.01)
