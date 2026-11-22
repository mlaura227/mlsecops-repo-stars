"""MLflow tracking, candidate registration and the training CLI."""

import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from src.data import FEATURE_COLUMNS, load_dataset
from src.preprocessing import split_features_target
from src.training import MODEL_KINDS, ModelKind, fit_and_evaluate


@dataclass(frozen=True)
class TrainingRun:
    """Run identifier and metrics for one trained candidate."""

    kind: ModelKind
    run_id: str
    metrics: dict[str, float]


def _sha256(path: Path) -> str:
    """Fingerprint a DVC-tracked CSV without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log_evaluation_artifacts(
    *, predictions: pd.DataFrame, train_hash: str, test_hash: str
) -> None:
    """Store the exact holdout predictions and schema beside the MLflow model."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        predictions_file = root / "test_predictions.csv"
        metadata_file = root / "dataset_manifest.json"
        predictions.to_csv(predictions_file, index=False)
        metadata_file.write_text(
            json.dumps(
                {
                    "train_sha256": train_hash,
                    "test_sha256": test_hash,
                    "feature_columns": list(FEATURE_COLUMNS),
                    "target_column": "stars",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(predictions_file), artifact_path="evaluation")
        mlflow.log_artifact(str(metadata_file), artifact_path="evaluation")


def train_experiment(
    *,
    train_path: Path = Path("data/processed/train.csv"),
    test_path: Path = Path("data/processed/test.csv"),
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "github-stars-regression",
    registered_model_name: str = "github-stars-regressor",
    random_state: int = 42,
) -> list[TrainingRun]:
    """Track a trivial baseline and a challenger, registering the challenger only.

    Registering a version does NOT promote it to a production/Champion alias.
    Automatic performance-gated promotion will be implemented separately.
    """
    if train_path.resolve() == test_path.resolve():
        raise ValueError("Training and holdout paths must differ")
    train_frame = load_dataset(train_path)
    test_frame = load_dataset(test_path)
    train_hash, test_hash = _sha256(train_path), _sha256(test_path)
    x_train, _ = split_features_target(train_frame)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    results: list[TrainingRun] = []
    for kind in MODEL_KINDS:
        with mlflow.start_run(run_name=f"{kind}-seed-{random_state}") as run:
            evaluation = fit_and_evaluate(train_frame, test_frame, kind=kind, random_state=random_state)
            mlflow.log_params(
                {
                    "model_kind": kind,
                    "random_state": random_state,
                    "train_rows": len(train_frame),
                    "test_rows": len(test_frame),
                    "target": "stars",
                    "train_sha256": train_hash,
                    "test_sha256": test_hash,
                }
            )
            mlflow.log_metrics(evaluation.metrics)
            _log_evaluation_artifacts(
                predictions=evaluation.predictions, train_hash=train_hash, test_hash=test_hash
            )
            mlflow.sklearn.log_model(   # pyright: ignore[reportPrivateImportUsage]
                sk_model=evaluation.pipeline,
                name="model",
                input_example=x_train.head(2),
                registered_model_name=(
                    registered_model_name if kind == "random_forest" else None
                ),
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                    "sklearn.tree._classes.Tree",
                    "sklearn.tree._classes.DecisionTreeClassifier",
                    "sklearn.tree._classes.DecisionTreeRegressor",
                    "sklearn.tree._classes.ExtraTreeClassifier",
                    "sklearn.tree._classes.ExtraTreeRegressor",
                    "sklearn.ensemble._forest.RandomForestClassifier",
                    "sklearn.ensemble._forest.RandomForestRegressor",
                    "sklearn.ensemble._forest.ExtraTreesClassifier",
                    "sklearn.ensemble._forest.ExtraTreesRegressor"
                ],
            )
            results.append(TrainingRun(kind=kind, run_id=run.info.run_id, metrics=evaluation.metrics))
    return results


def main() -> None:
    """Train and track both models using the configured MLflow backend."""
    parser = argparse.ArgumentParser(description="Train and track GitHub star regression models")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="github-stars-regression")
    parser.add_argument("--registered-model", default="github-stars-regressor")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    runs = train_experiment(
        train_path=args.train,
        test_path=args.test,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        registered_model_name=args.registered_model,
        random_state=args.seed,
    )
    for run in runs:
        print(f"{run.kind}: run_id={run.run_id}, metrics={run.metrics}")
