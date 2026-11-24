"""Validation-safe hyperparameter tuning, isolated from production promotion.

Only the existing training dataset is read. The held-out test dataset and the
MLflow champion alias are deliberately excluded from hyperparameter selection.
"""

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from mlflow.sklearn import log_model as log_sklearn_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset
from src.preprocessing import split_features_target
from src.training import build_pipeline

# A small, deterministic search space to keep local CPU training practical.
DEFAULT_GRID: dict[str, list[int | float]] = {
    "regressor__n_estimators": [80, 140],
    "regressor__min_samples_leaf": [1, 2],
    "regressor__max_features": [0.7, 1.0],
}


@dataclass(frozen=True)
class TuningResult:
    """Summary of a non-promoted tuned model and its validation evidence."""

    run_id: str
    validation_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    cv_best_mae: float
    best_params: dict[str, int | float]
    training_rows: int
    validation_rows: int


def _metrics(actual: pd.Series, predictions: NDArray[np.float64]) -> dict[str, float]:
    """Return finite, non-negative validation metrics for one fitted pipeline."""
    observed = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if predicted.shape != observed.shape or not np.isfinite(predicted).all():
        raise ValueError("Model returned invalid validation predictions")
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
    }


def _normalize_params(params: dict[str, object]) -> dict[str, int | float]:
    """Convert the numeric search parameters to JSON-serializable values."""
    normalized: dict[str, int | float] = {}
    for key, value in params.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            normalized[key] = value
        else:
            raise TypeError(f"Unexpected search parameter type for {key}")
    return normalized


def tune_experiment(
    *,
    train_path: Path = Path("data/processed/train.csv"),
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "github-stars-tuning",
    random_state: int = 42,
    validation_fraction: float = 0.2,
    param_grid: dict[str, list[int | float]] | None = None,
) -> TuningResult:
    """Tune using an inner CV, then validate on rows untouched by any fit.

    This workflow does not read test.csv, register models, or change champion.
    Production promotion continues to use the separate existing workflow.
    """
    if not 0.1 <= validation_fraction <= 0.4:
        raise ValueError("validation_fraction must be between 0.1 and 0.4")
    if not experiment_name.strip():
        raise ValueError("experiment_name cannot be blank")
    grid = DEFAULT_GRID if param_grid is None else param_grid
    if not grid or any(not values for values in grid.values()):
        raise ValueError("param_grid must contain non-empty choices")

    frame = load_dataset(train_path)
    if len(frame) < 30:
        raise ValueError("At least 30 training rows are required for tuning")
    inner_train, validation = train_test_split(
        frame, test_size=validation_fraction, random_state=random_state, shuffle=True
    )
    if not isinstance(inner_train, pd.DataFrame) or not isinstance(validation, pd.DataFrame):
        raise TypeError("Expected DataFrame outputs from train_test_split")

    x_train, y_train = split_features_target(inner_train)
    x_validation, y_validation = split_features_target(validation)
    baseline = build_pipeline("baseline", random_state=random_state)
    baseline.fit(x_train, y_train)
    baseline_metrics = _metrics(y_validation, np.asarray(baseline.predict(x_validation), dtype=np.float64))

    search = GridSearchCV(
        estimator=build_pipeline("random_forest", random_state=random_state),
        param_grid=grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=1,
        refit=True,
        error_score="raise",    # pyright: ignore[reportArgumentType]
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_
    if not isinstance(best, Pipeline):
        raise TypeError("Expected an sklearn Pipeline as the best estimator")
    predictions = np.asarray(best.predict(x_validation), dtype=np.float64)
    validation_metrics = _metrics(y_validation, predictions)
    best_params = _normalize_params(search.best_params_)

    source_sha256 = hashlib.sha256(train_path.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "source_train_csv": str(train_path),
        "source_sha256": source_sha256,
        "seed": random_state,
        "validation_fraction": validation_fraction,
        "fit_rows": len(inner_train),
        "validation_rows": len(validation),
        "validation_row_indices": [int(i) for i in validation.index.tolist()],
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "best_params": best_params,
        "cv_folds": 3,
        "final_test_used_for_tuning": False,
        "registered_model": False,
    }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"grid-search-seed-{random_state}") as run:
        mlflow.log_params(
            {
                "random_state": random_state,
                "validation_fraction": validation_fraction,
                "train_sha256": source_sha256,
                "search_candidates": len(search.cv_results_["params"]),
                "cv_folds": 3,
                **best_params,
            }
        )
        mlflow.log_metrics(
            {
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "baseline_validation_mae": baseline_metrics["mae"],
                "baseline_validation_rmse": baseline_metrics["rmse"],
                "cv_best_mae": -float(search.best_score_),
            }
        )
        mlflow.set_tag("model_stage", "research_only_not_promoted")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = x_validation.reset_index(drop=True).copy()
            output[TARGET_COLUMN] = np.asarray(y_validation, dtype=np.float64)
            output["prediction"] = predictions
            output.to_csv(root / "validation_predictions.csv", index=False)
            pd.DataFrame(search.cv_results_).to_csv(root / "cv_results.csv", index=False)
            (root / "tuning_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            for file in sorted(root.iterdir()):
                mlflow.log_artifact(str(file), artifact_path="tuning")
        log_sklearn_model(
            sk_model=best,
            name="tuned_candidate",
            input_example=x_train.head(2),
            # Only known NumPy/sklearn internals are approved for skops.
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        return TuningResult(
            run_id=run.info.run_id,
            validation_metrics=validation_metrics,
            baseline_metrics=baseline_metrics,
            cv_best_mae=-float(search.best_score_),
            best_params=best_params,
            training_rows=len(inner_train),
            validation_rows=len(validation),
        )


def main() -> None:
    """CLI to create an auditable, non-production MLflow tuning run."""
    parser = argparse.ArgumentParser(description="Tune Random Forest on training-only rows")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="github-stars-tuning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    result = tune_experiment(
        train_path=args.train,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        random_state=args.seed,
        validation_fraction=args.validation_fraction,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
