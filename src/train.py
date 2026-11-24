"""MLflow tracking, candidate registration and the training CLI."""

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.sklearn import log_model as log_sklearn_model

from src.data import FEATURE_COLUMNS, load_dataset
from src.preprocessing import split_features_target
from src.promotion import PromotionDecision, promote_candidate
from src.training import MODEL_KINDS, ModelKind, fit_and_evaluate


@dataclass(frozen=True)
class TrainingRun:
    """Run identifier and metrics for one trained candidate."""

    kind: ModelKind
    run_id: str
    metrics: dict[str, float]
    registered_model_version: int | None = None


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
    Pass --auto-promote to run the performance gate after training.
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
            model_info = log_sklearn_model(
                sk_model=evaluation.pipeline,
                name="model",
                input_example=x_train.head(2),
                registered_model_name=registered_model_name if kind == "random_forest" else None,
                # Trust only the known numpy/scikit-learn internals needed by skops.
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            )
            model_version = model_info.registered_model_version if kind == "random_forest" else None
            results.append(
                TrainingRun(
                    kind=kind,
                    run_id=run.info.run_id,
                    metrics=evaluation.metrics,
                    registered_model_version=model_version,
                )
            )
    return results


def log_promotion_decision(run_id: str, decision: PromotionDecision) -> None:
    """Persist the gate decision on the candidate MLflow run."""
    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(asdict(decision), "promotion/decision.json")
        mlflow.set_tag("promotion.status", "accepted" if decision.accepted else "rejected")
        mlflow.log_metric("promotion.candidate_mae", decision.candidate_metrics["mae"])
        mlflow.log_metric("promotion.baseline_mae", decision.baseline_metrics["mae"])
        if decision.champion_metrics is not None:
            mlflow.log_metric("promotion.champion_mae", decision.champion_metrics["mae"])


def main() -> None:
    """Train and track both models using the configured MLflow backend."""
    parser = argparse.ArgumentParser(description="Train and track GitHub star regression models")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="github-stars-regression")
    parser.add_argument("--registered-model", default="github-stars-regressor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-promote", action="store_true")
    parser.add_argument("--min-mae-improvement", type=float, default=0.0)
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
    if args.auto_promote:
        baseline = next(run for run in runs if run.kind == "baseline")
        challenger = next(run for run in runs if run.kind == "random_forest")
        if challenger.registered_model_version is None:
            raise RuntimeError("The challenger was not registered; promotion stopped")
        decision = promote_candidate(
            candidate_version=challenger.registered_model_version,
            baseline_metrics=baseline.metrics,
            test_path=args.test,
            model_name=args.registered_model,
            tracking_uri=args.tracking_uri,
            min_relative_improvement=args.min_mae_improvement,
        )
        log_promotion_decision(challenger.run_id, decision)
        print(f"Promotion: {'ACCEPTED' if decision.accepted else 'REJECTED'} — {decision.reason}")
