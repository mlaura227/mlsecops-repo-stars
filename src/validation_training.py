"""Train, tune and optionally promote using ONLY train.csv's inner holdout.

The original 200-row test.csv and its legacy champion are never accessed.
Use a separate model registry until a controlled serving migration is done.
"""

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.sklearn import log_model as log_sklearn_model
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.data import TARGET_COLUMN, load_dataset
from src.preprocessing import split_features_target
from src.training import build_pipeline
from src.tuning import DEFAULT_GRID
from src.validation_promotion import (
    MODEL_NAME, PROTOCOL_TAG, ValidationDecision, promote_validation_candidate,
    validate_registry_name,
)
from src.validation_protocol import fingerprint_file, split_validation_cohort


@dataclass(frozen=True)
class ValidationTrainingResult:
    """Recorded experiment and promotion decision, if requested."""

    run_id: str
    model_version: int
    protocol_id: str
    fit_rows: int
    validation_rows: int
    candidate_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    best_params: dict[str, int | float]
    decision: ValidationDecision | None


def _metrics(target: pd.Series, predicted: NDArray[np.float64]) -> dict[str, float]:
    expected = np.asarray(target, dtype=np.float64)
    values = np.asarray(predicted, dtype=np.float64)
    if expected.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("Invalid regression predictions for validation")
    return {
        "mae": float(mean_absolute_error(expected, values)),
        "rmse": float(np.sqrt(mean_squared_error(expected, values))),
    }


def _best_params(params: dict[str, object]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for key, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Unexpected hyperparameter type: {key}")
        result[key] = value
    return result


def train_validation_candidate(
    *,
    train_path: Path = Path("data/processed/train.csv"),
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "github-stars-validation-training",
    model_name: str = MODEL_NAME,
    random_state: int = 42,
    validation_fraction: float = 0.2,
    min_relative_improvement: float = 0.01,
    auto_promote: bool = False,
    param_grid: dict[str, list[int | float]] | None = None,
) -> ValidationTrainingResult:
    """Tune on 640 fit rows, compare on 160 validation rows, log in MLflow.

    The source is the 800-row train.csv (never the original 200-row test.csv).
    For auditability, the source bytes and exact validation indices are logged.
    """
    validate_registry_name(model_name)
    if not experiment_name.strip():
        raise ValueError("experiment_name cannot be blank")
    if not 0 <= min_relative_improvement < 1:
        raise ValueError("min_relative_improvement must be in [0, 1)")
    grid = DEFAULT_GRID if param_grid is None else param_grid
    if not grid or any(not values for values in grid.values()):
        raise ValueError("param_grid must contain nonempty search values")

    frame = load_dataset(train_path)
    fit, validation, protocol = split_validation_cohort(
        frame, source_sha256=fingerprint_file(train_path),
        random_state=random_state, validation_fraction=validation_fraction,
    )
    fit_x, fit_y = split_features_target(fit)
    val_x, val_y = split_features_target(validation)

    baseline = build_pipeline("baseline", random_state=random_state)
    baseline.fit(fit_x, fit_y)
    baseline_metrics = _metrics(val_y, np.asarray(baseline.predict(val_x), dtype=np.float64))

    search = GridSearchCV(
        estimator=build_pipeline("random_forest", random_state=random_state),
        param_grid=grid,
        scoring="neg_mean_absolute_error", cv=3, n_jobs=1,
        refit=True, error_score=float("-inf"),
    )
    search.fit(fit_x, fit_y)
    model = search.best_estimator_
    if not isinstance(model, Pipeline):
        raise TypeError("Expected Pipeline from hyperparameter search")
    prediction = np.asarray(model.predict(val_x), dtype=np.float64)
    candidate_metrics = _metrics(val_y, prediction)
    best_params = _best_params(search.best_params_)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"validation-tuned-seed-{random_state}") as run:
        mlflow.log_params({
            "random_state": random_state, "validation_fraction": validation_fraction,
            "protocol_id": protocol.protocol_id,
            "train_sha256": protocol.source_sha256,
            "fit_rows": len(fit), "validation_rows": len(validation),
            "cv_folds": 3, "model_name": model_name, **best_params,
        })
        mlflow.log_metrics({
            "validation_mae": candidate_metrics["mae"],
            "validation_rmse": candidate_metrics["rmse"],
            "baseline_validation_mae": baseline_metrics["mae"],
            "baseline_validation_rmse": baseline_metrics["rmse"],
            "cv_best_mae": -float(search.best_score_),
        })
        mlflow.set_tag("validation.protocol_id", protocol.protocol_id)
        mlflow.set_tag("data.final_test_used_for_selection", "false")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = val_x.reset_index(drop=True).copy()
            output[TARGET_COLUMN] = np.asarray(val_y, dtype=np.float64)
            output["prediction"] = prediction
            output.to_csv(root / "validation_predictions.csv", index=False)
            pd.DataFrame(search.cv_results_).to_csv(root / "cv_results.csv", index=False)
            (root / "validation_protocol.json").write_text(
                json.dumps(protocol.manifest(), indent=2), encoding="utf-8"
            )
            for artifact in root.iterdir():
                mlflow.log_artifact(str(artifact), artifact_path="validation")
        info = log_sklearn_model(
            sk_model=model, name="model", input_example=fit_x.head(2),
            registered_model_name=model_name,
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        raw_version = info.registered_model_version
        if raw_version is None:
            raise RuntimeError("MLflow did not register a model version")
        model_version = int(raw_version)
        run_id = run.info.run_id

    # Registration must be complete before adding the protocol tag. A newly
    # registered version cannot be promoted until this tag is stored.
    MlflowClient().set_model_version_tag(
        model_name, str(model_version), PROTOCOL_TAG, protocol.protocol_id
    )

    decision: ValidationDecision | None = None
    if auto_promote:
        decision = promote_validation_candidate(
            candidate_version=model_version, protocol_id=protocol.protocol_id,
            baseline_metrics=baseline_metrics, validation_frame=validation,
            tracking_uri=tracking_uri, model_name=model_name,
            min_relative_improvement=min_relative_improvement,
        )
        with mlflow.start_run(run_id=run_id):
            mlflow.log_dict(asdict(decision), "validation/promotion_decision.json")
            mlflow.set_tag("validation.promotion_status", "accepted" if decision.accepted else "rejected")

    return ValidationTrainingResult(
        run_id=run_id, model_version=model_version, protocol_id=protocol.protocol_id,
        fit_rows=len(fit), validation_rows=len(validation),
        candidate_metrics=candidate_metrics, baseline_metrics=baseline_metrics,
        best_params=best_params, decision=decision,
    )


def main() -> None:
    """Run without changing pyproject.toml entry points."""
    parser = argparse.ArgumentParser(description="Train with a versioned validation-only promotion gate")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-promote", action="store_true")
    parser.add_argument("--min-mae-improvement", type=float, default=0.01)
    args = parser.parse_args()
    result = train_validation_candidate(
        train_path=args.train, tracking_uri=args.tracking_uri,
        model_name=args.model_name, random_state=args.seed,
        auto_promote=args.auto_promote, min_relative_improvement=args.min_mae_improvement,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
