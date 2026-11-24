"""Load logged inference traffic and produce auditable Evidently drift reports.

Only feature columns are compared; predictions and the target are excluded.
The reference is the training split, not the frequently reused holdout set.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.data import BOOLEAN_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, load_dataset

MIN_CURRENT_ROWS = 10
MAX_LOGGED_PREDICTIONS = 500


@dataclass(frozen=True)
class PredictionBatch:
    """Collected API events and an explicit count of skipped legacy/bad runs."""

    features: pd.DataFrame
    scanned_runs: int
    skipped_runs: int


@dataclass(frozen=True)
class DriftResult:
    """Rendered report and the MLflow run storing the same report."""

    html: str
    report_run_id: str


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate inference features, preserving the exact training schema/order."""
    if frame.empty:
        raise ValueError("No feature rows available for monitoring")
    missing = set(FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing monitoring features: {sorted(missing)}")

    result = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    language = pd.Series(result["language"], index=result.index).astype("string").str.strip()
    if language.isna().any() or language.eq("").any():
        raise ValueError("language must contain non-empty values")
    result["language"] = language.astype(str)

    for column in NUMERIC_FEATURES:
        values = pd.Series(pd.to_numeric(result[column], errors="coerce"), index=result.index)
        array = values.to_numpy(dtype="float64")
        if not np.isfinite(array).all() or (array < 0).any() or (array != np.floor(array)).any():
            raise ValueError(f"Invalid numeric feature: {column}")
        result[column] = values.astype("int64")

    for column in BOOLEAN_FEATURES:
        normalized = pd.Series(result[column], index=result.index).astype("string").str.strip().str.lower()
        binary = pd.Series(normalized.map({"1": 1, "0": 0, "true": 1, "false": 0}), index=result.index)
        if binary.isna().any():
            raise ValueError(f"Invalid boolean feature: {column}")
        result[column] = binary.astype("int8")
    return result


def dataset_features(path: Path) -> pd.DataFrame:
    """Load DVC-prepared dataset and select only model input features."""
    return normalize_features(load_dataset(path))


def _empty_features() -> pd.DataFrame:
    """Construct an empty schema without pandas ColumnIndex typing ambiguity."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in FEATURE_COLUMNS})


def load_prediction_batch(
    *, tracking_uri: str, experiment_name: str, limit: int = 200,
) -> PredictionBatch:
    """Load the most recent prediction CSV artifacts from MLflow.

    Per-request logging is adequate for coursework; production systems should
    instead batch telemetry rather than download one CSV per HTTP request.
    """
    if not 1 <= limit <= MAX_LOGGED_PREDICTIONS:
        raise ValueError("limit must be between 1 and 500")
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(f"{experiment_name}-api")
    if experiment is None:
        return PredictionBatch(_empty_features(), 0, 0)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=min(MAX_LOGGED_PREDICTIONS * 3, limit * 3),
        order_by=["attributes.start_time DESC"],
    )
    batches: list[pd.DataFrame] = []
    skipped = 0
    scanned = 0
    with tempfile.TemporaryDirectory() as directory:
        for run in runs:
            if run.data.tags.get("api.endpoint") != "predict":
                continue
            if len(batches) >= limit:
                break
            scanned += 1
            artifact = "monitoring/predictions.csv"
            available = client.list_artifacts(run.info.run_id, path="monitoring")
            if not any(item.path == artifact and not item.is_dir for item in available):
                skipped += 1
                continue
            local = client.download_artifacts(run.info.run_id, artifact, dst_path=directory)
            try:
                rows = pd.read_csv(local)
                if len(rows) != 1:
                    raise ValueError("One event must contain one prediction row")
                batches.append(normalize_features(rows))
            except (ValueError, pd.errors.ParserError, UnicodeError):
                # Historical corrupt requests should not suppress valid later data.
                skipped += 1

    if not batches:
        return PredictionBatch(_empty_features(), scanned, skipped)
    return PredictionBatch(pd.concat(batches, ignore_index=True), scanned, skipped)


def create_drift_report(
    *, reference: pd.DataFrame, current: pd.DataFrame, tracking_uri: str,
    experiment_name: str, source: str,
) -> DriftResult:
    """Run Evidently v0.7 drift evaluation, log HTML+JSON and return HTML."""
    if source not in {"api", "holdout-demo"}:
        raise ValueError("source must be api or holdout-demo")
    reference_features = normalize_features(reference)
    current_features = normalize_features(current)
    if len(current_features) < MIN_CURRENT_ROWS:
        raise ValueError(f"At least {MIN_CURRENT_ROWS} current observations are required")

    # Keep the report import close to its use to allow unit testing pure data
    # validation independently of the relatively heavy Evidently runtime.
    from evidently import Report
    from evidently.presets import DataDriftPreset

    snapshot = Report([DataDriftPreset()]).run(
        current_data=current_features, reference_data=reference_features,
    )
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        html_path = output / "data_drift.html"
        json_path = output / "data_drift.json"
        snapshot.save_html(str(html_path))
        json_path.write_text(snapshot.json(), encoding="utf-8")
        html = html_path.read_text(encoding="utf-8")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(f"{experiment_name}-monitoring")
        with mlflow.start_run(run_name=f"drift-{source}") as run:
            mlflow.set_tag("monitoring.source", source)
            mlflow.log_params({
                "reference_rows": len(reference_features),
                "current_rows": len(current_features),
                "source": source,
            })
            mlflow.log_artifact(str(html_path), artifact_path="monitoring")
            mlflow.log_artifact(str(json_path), artifact_path="monitoring")
            return DriftResult(html=html, report_run_id=run.info.run_id)


def settings_from_env() -> tuple[str, str, Path, Path]:
    """Use the same tracking URI and prepared splits as the API service."""
    return (
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        os.getenv("MLSECOPS_EXPERIMENT", "github-stars-regression"),
        Path(os.getenv("MLSECOPS_TRAIN_CSV", "data/processed/train.csv")),
        Path(os.getenv("MLSECOPS_TEST_CSV", "data/processed/test.csv")),
    )
