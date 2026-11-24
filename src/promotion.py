"""Fail-closed MLflow Model Registry promotion based on current holdout data.

A registered challenger is NOT a serving model until it wins the gate. This
module deliberately compares the same test rows for challenger and champion.
"""

from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data import load_dataset
from src.preprocessing import split_features_target

CHAMPION_ALIAS = "champion"


@dataclass(frozen=True)
class PromotionDecision:
    """Auditable result of evaluating one registered candidate."""

    accepted: bool
    reason: str
    candidate_version: int
    previous_champion_version: int | None
    candidate_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    champion_metrics: dict[str, float] | None


def _validate_metrics(metrics: dict[str, float]) -> None:
    """Do not accept models when comparison measurements are invalid."""
    for key in ("mae", "rmse"):
        if key not in metrics or not np.isfinite(metrics[key]) or metrics[key] < 0:
            raise ValueError(f"Invalid {key} metric for promotion")


def _beats_reference(
    candidate: dict[str, float], reference: dict[str, float], min_relative_improvement: float
) -> bool:
    """Require strictly lower MAE, adequate gain, and no RMSE regression."""
    _validate_metrics(candidate)
    _validate_metrics(reference)
    if reference["mae"] <= 0 or candidate["mae"] >= reference["mae"]:
        return False
    improvement = (reference["mae"] - candidate["mae"]) / reference["mae"]
    return improvement >= min_relative_improvement and candidate["rmse"] <= reference["rmse"]


def score_registered_version(model_name: str, version: int, frame: pd.DataFrame) -> dict[str, float]:
    """Reload a registered artifact and evaluate it on exactly these rows."""
    if len(frame) < 2:
        raise ValueError("Promotion evaluation requires at least two test rows")
    features, target = split_features_target(frame)
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{version}")  # type: ignore
    observed = np.asarray(target, dtype=np.float64)
    assert model is not None
    predicted = np.asarray(model.predict(features), dtype=np.float64)
    if predicted.shape != observed.shape or not np.isfinite(predicted).all():
        raise ValueError("Registered model returned invalid predictions")
    metrics = {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
    }
    _validate_metrics(metrics)
    return metrics


def promote_candidate(
    *,
    candidate_version: int,
    baseline_metrics: dict[str, float],
    test_path: Path,
    model_name: str = "github-stars-regressor",
    tracking_uri: str = "sqlite:///mlflow.db",
    min_relative_improvement: float = 0.0,
) -> PromotionDecision:
    """Promote only if the challenger beats baseline and current champion.

    Any unexpected registry, storage, or model loading failure propagates before
    the alias can be changed. Only a genuinely missing alias means bootstrap.
    """
    if candidate_version <= 0:
        raise ValueError("candidate_version must be a positive integer")
    if not model_name.strip():
        raise ValueError("model_name cannot be blank")
    if not 0 <= min_relative_improvement < 1:
        raise ValueError("min_relative_improvement must be in [0, 1)")
    _validate_metrics(baseline_metrics)

    mlflow.set_tracking_uri(tracking_uri)
    frame = load_dataset(test_path)
    client = MlflowClient()
    # Verify the candidate exists before attempting to change the serving alias.
    client.get_model_version(model_name, str(candidate_version))
    candidate_metrics = score_registered_version(model_name, candidate_version, frame)

    # Inspect the model's aliases directly. SQLite MLflow registries can raise
    # INVALID_PARAMETER_VALUE (not RESOURCE_DOES_NOT_EXIST) for a missing alias.
    # A missing alias is a valid bootstrap state; a registry failure is not.
    registered_model = client.get_registered_model(model_name)
    alias_version = registered_model.aliases.get(CHAMPION_ALIAS)
    previous_version = int(alias_version) if alias_version is not None else None
    champion_metrics = None
    if previous_version is not None:
        if previous_version == candidate_version:
            return PromotionDecision(
                accepted=False,
                reason="Candidate is already champion",
                candidate_version=candidate_version,
                previous_champion_version=previous_version,
                candidate_metrics=candidate_metrics,
                baseline_metrics=baseline_metrics,
                champion_metrics=candidate_metrics,
            )
        champion_metrics = score_registered_version(model_name, previous_version, frame)

    if not _beats_reference(candidate_metrics, baseline_metrics, min_relative_improvement):
        accepted, reason = False, "Candidate did not beat the baseline (MAE/RMSE gate)"
    elif champion_metrics is not None and not _beats_reference(
        candidate_metrics, champion_metrics, min_relative_improvement
    ):
        accepted, reason = False, "Candidate did not beat the current champion (MAE/RMSE gate)"
    else:
        accepted, reason = True, "Candidate passed all performance gates"

    # Tags make both accepted and rejected candidates auditable in the registry.
    client.set_model_version_tag(
        model_name, str(candidate_version), "gate_decision", "accepted" if accepted else "rejected"
    )
    client.set_model_version_tag(model_name, str(candidate_version), "gate_reason", reason)
    if accepted:
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, str(candidate_version))

    return PromotionDecision(
        accepted=accepted,
        reason=reason,
        candidate_version=candidate_version,
        previous_champion_version=previous_version,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        champion_metrics=champion_metrics,
    )
