"""Fast unit tests for real telemetry ingestion and Evidently report artifacts."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from src.data import FEATURE_COLUMNS, validate_dataset
from src.monitoring import (
    MIN_CURRENT_ROWS,
    create_drift_report,
    dataset_features,
    load_prediction_batch,
    normalize_features,
    settings_from_env,
)


@pytest.fixture
def features(sample_dataset: pd.DataFrame) -> pd.DataFrame:
    """Mimic API/CSV feature columns after training-time validation."""
    return validate_dataset(sample_dataset).loc[:, list(FEATURE_COLUMNS)].copy()


def test_normalization_and_target_exclusion(features: pd.DataFrame) -> None:
    raw = features.copy()
    raw["stars"] = 123
    raw["has_coc"] = raw["has_coc"].astype(str)
    actual = normalize_features(raw)
    assert list(actual.columns) == list(FEATURE_COLUMNS)
    assert "stars" not in actual
    assert actual["has_coc"].dtype == "int8"
    assert len(actual) == 32


@pytest.mark.parametrize(("column", "value", "message"), [
    ("language", "", "language"),
    ("language", None, "language"),
    ("forks", -1, "forks"),
    ("forks", "oops", "forks"),
    ("age_days", 1.1, "age_days"),
    ("open_issues", float("inf"), "open_issues"),
    ("has_coc", "maybe", "has_coc"),
])
def test_rejects_invalid_features(
    features: pd.DataFrame, column: str, value: object, message: str,
) -> None:
    bad = features.copy()
    bad[column] = bad[column].astype(object)
    bad.loc[bad.index[0], column] = value
    with pytest.raises(ValueError, match=message):
        normalize_features(bad)


def test_rejects_empty_and_missing_columns(features: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="No feature rows"):
        normalize_features(features.iloc[:0])
    with pytest.raises(ValueError, match="Missing monitoring features"):
        normalize_features(features.drop(columns=["forks"]))


def test_dataset_features_reads_original_schema(sample_dataset: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    sample_dataset.to_csv(path, index=False)
    result = dataset_features(path)
    assert len(result) == 32
    assert list(result.columns) == list(FEATURE_COLUMNS)


class FakeClient:
    """Simulate MLflow's public experiment and artifact methods."""

    def __init__(self, directory: Path, *, experiment: bool = True) -> None:
        self.directory = directory
        self.experiment = experiment
        self.run_tags = ["predict", "train", "predict", "predict"]
        self.missing = {"run-2"}
        self.corrupt = {"run-3"}
        self.params: dict[str, object] = {}

    def get_experiment_by_name(self, name: str) -> object | None:
        self.params["experiment_name"] = name
        return SimpleNamespace(experiment_id="44") if self.experiment else None

    def search_runs(self, **kwargs: object) -> list[SimpleNamespace]:
        self.params.update(kwargs)
        return [SimpleNamespace(
            data=SimpleNamespace(tags={"api.endpoint": tag}),
            info=SimpleNamespace(run_id=f"run-{index}"),
        ) for index, tag in enumerate(self.run_tags)]

    def list_artifacts(self, run_id: str, path: str) -> list[SimpleNamespace]:
        if run_id in self.missing:
            return []
        return [SimpleNamespace(path="monitoring/predictions.csv", is_dir=False)]

    def download_artifacts(self, run_id: str, artifact: str, dst_path: str) -> str:
        dest = Path(dst_path) / f"{run_id}.csv"
        data = {
            "language": ["Go"], "forks": [10], "open_issues": [4],
            "age_days": [900], "has_coc": [1], "has_contributing": [0],
            "has_workflows": [1], "has_readme": [1], "prediction": [4200],
        }
        pd.DataFrame(data).to_csv(dest, index=False)
        if run_id in self.corrupt:
            dest.write_text("bad-file\n", encoding="utf-8")
        return str(dest)


def test_load_api_predictions_skips_legacy_and_corrupt_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import src.monitoring as module

    client = FakeClient(tmp_path)
    client.run_tags.append("predict")
    monkeypatch.setattr(module, "MlflowClient", lambda tracking_uri: client)
    batch = load_prediction_batch(tracking_uri="http://mlflow:5000", experiment_name="project", limit=10)
    assert batch.scanned_runs == 4
    assert batch.skipped_runs == 2
    assert len(batch.features) == 2
    assert list(batch.features.columns) == list(FEATURE_COLUMNS)
    assert client.params["experiment_name"] == "project-api"
    assert client.params["order_by"] == ["attributes.start_time DESC"]


def test_empty_experiment_and_missing_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.monitoring as module

    client = FakeClient(tmp_path, experiment=False)
    monkeypatch.setattr(module, "MlflowClient", lambda tracking_uri: client)
    assert load_prediction_batch(tracking_uri="uri", experiment_name="x").features.empty
    client.experiment = True
    client.run_tags = ["train"]
    batch = load_prediction_batch(tracking_uri="uri", experiment_name="x")
    assert batch.scanned_runs == 0
    assert batch.features.empty


@pytest.mark.parametrize("invalid", [0, -1, 501])
def test_prediction_limit_guard(invalid: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        load_prediction_batch(tracking_uri="uri", experiment_name="x", limit=invalid)


def test_loading_stops_at_requested_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.monitoring as module

    client = FakeClient(tmp_path)
    client.missing.clear()
    client.corrupt.clear()
    client.run_tags = ["predict"] * 30
    monkeypatch.setattr(module, "MlflowClient", lambda tracking_uri: client)
    batch = load_prediction_batch(tracking_uri="uri", experiment_name="x", limit=10)
    assert len(batch.features) == 10
    assert batch.scanned_runs == 10


def test_report_writes_two_mlflow_artifacts(
    features: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.monitoring as module

    created: list[dict[str, object]] = []

    class FakeSnapshot:
        def save_html(self, path: str) -> None:
            Path(path).write_text("<html><body>Fake Evidently</body></html>", encoding="utf-8")

        def json(self) -> str:
            return json.dumps({"report": "data drift"})

    class FakeReport:
        def __init__(self, presets: object) -> None:
            created.append({"presets": presets})

        def run(self, *, current_data: pd.DataFrame, reference_data: pd.DataFrame) -> FakeSnapshot:
            created.append({"current": current_data, "reference": reference_data})
            return FakeSnapshot()

    fake_evidently = ModuleType("evidently")
    setattr(fake_evidently, "Report", FakeReport)
    fake_presets = ModuleType("evidently.presets")
    setattr(fake_presets, "DataDriftPreset", lambda: "preset")
    monkeypatch.setitem(sys.modules, "evidently", fake_evidently)
    monkeypatch.setitem(sys.modules, "evidently.presets", fake_presets)

    reports: list[tuple[str, str, str]] = []
    monkeypatch.setattr(module.mlflow, "set_tracking_uri", lambda uri: reports.append(("uri", uri, "")))
    monkeypatch.setattr(module.mlflow, "set_experiment", lambda name: reports.append(("exp", name, "")))
    monkeypatch.setattr(module.mlflow, "start_run", lambda run_name: nullcontext(
        SimpleNamespace(info=SimpleNamespace(run_id="drift-44"))))
    monkeypatch.setattr(module.mlflow, "set_tag", lambda k, v: reports.append(("tag", k, v)))
    monkeypatch.setattr(module.mlflow, "log_params", lambda data: reports.append(
        ("params", str(data["reference_rows"]), str(data["current_rows"]))))
    monkeypatch.setattr(module.mlflow, "log_artifact", lambda path, artifact_path: reports.append(
        ("artifact", Path(path).name, Path(path).read_text(encoding="utf-8"))))

    result = create_drift_report(
        reference=features, current=features.iloc[:MIN_CURRENT_ROWS],
        tracking_uri="http://mlflow:5000", experiment_name="project", source="api",
    )
    assert result.report_run_id == "drift-44"
    assert "Fake Evidently" in result.html
    assert [x[1] for x in reports if x[0] == "artifact"] == ["data_drift.html", "data_drift.json"]
    assert ("tag", "monitoring.source", "api") in reports
    assert ("exp", "project-monitoring", "") in reports
    assert list(created[1]["current"].columns) == list(FEATURE_COLUMNS)
    assert len(created[1]["reference"]) == len(features)


def test_report_validation_precedes_external_runtime(features: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="source"):
        create_drift_report(reference=features, current=features, tracking_uri="x", experiment_name="x", source="x")
    with pytest.raises(ValueError, match="At least"):
        create_drift_report(reference=features, current=features.iloc[:2],
                            tracking_uri="x", experiment_name="x", source="holdout-demo")


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLSECOPS_EXPERIMENT", "custom")
    monkeypatch.setenv("MLSECOPS_TRAIN_CSV", "data/train.csv")
    monkeypatch.setenv("MLSECOPS_TEST_CSV", "data/test.csv")
    assert settings_from_env() == (
        "http://localhost:5000", "custom", Path("data/train.csv"), Path("data/test.csv"),
    )
