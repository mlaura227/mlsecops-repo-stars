"""Fail-closed model promotion on a versioned, training-only validation cohort.

Never use the legacy registry: its champion was fitted on the validation rows.
"""

from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.promotion import CHAMPION_ALIAS, score_registered_version

MODEL_NAME = "github-stars-regressor-validation-v1"
LEGACY_MODEL_NAME = "github-stars-regressor"
PROTOCOL_TAG = "validation.protocol_id"
DECISION_TAG = "validation.gate_decision"


@dataclass(frozen=True)
class ValidationDecision:
    """Persistable details of an immutable evaluation decision."""

    accepted: bool
    reason: str
    candidate_version: int
    previous_champion_version: int | None
    protocol_id: str
    candidate_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    champion_metrics: dict[str, float] | None


def validate_registry_name(model_name: str) -> None:
    """Never silently overwrite a legacy model trained on validation rows."""
    if not model_name.strip() or model_name == LEGACY_MODEL_NAME:
        raise ValueError("Use a non-legacy validation registry name")


def _validate_metrics(metrics: dict[str, float]) -> None:
    for key in ("mae", "rmse"):
        if key not in metrics or not np.isfinite(metrics[key]) or metrics[key] < 0:
            raise ValueError(f"Invalid validation metric: {key}")


def _beats(candidate: dict[str, float], reference: dict[str, float], threshold: float) -> bool:
    """Strictly improve MAE by threshold without worsening RMSE."""
    _validate_metrics(candidate)
    _validate_metrics(reference)
    return (
        reference["mae"] > 0
        and candidate["mae"] < reference["mae"]
        and (reference["mae"] - candidate["mae"]) / reference["mae"] >= threshold
        and candidate["rmse"] <= reference["rmse"]
    )


def promote_validation_candidate(
    *,
    candidate_version: int,
    protocol_id: str,
    baseline_metrics: dict[str, float],
    validation_frame: pd.DataFrame,
    tracking_uri: str = "sqlite:///mlflow.db",
    model_name: str = MODEL_NAME,
    min_relative_improvement: float = 0.01,
) -> ValidationDecision:
    """Promote only on comparable model versions evaluated on the same cohort.

    Registry or scoring failures propagate and never modify the alias. A
    champion from a different protocol is rejected rather than evaluated on
    data it may have trained on.
    """
    validate_registry_name(model_name)
    if candidate_version <= 0:
        raise ValueError("candidate_version must be positive")
    if len(protocol_id) != 64 or any(char not in "0123456789abcdef" for char in protocol_id):
        raise ValueError("protocol_id must be a SHA-256 digest")
    if not 0 <= min_relative_improvement < 1:
        raise ValueError("min_relative_improvement must be in [0, 1)")
    _validate_metrics(baseline_metrics)
    if len(validation_frame) < 2:
        raise ValueError("Validation cohort must contain at least two rows")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    candidate = client.get_model_version(model_name, str(candidate_version))
    if candidate.tags.get(PROTOCOL_TAG) != protocol_id:
        raise ValueError("Candidate validation protocol does not match the requested cohort")

    # The registered artifact is scored rather than a potentially different
    # in-memory sklearn model. This also verifies that the saved model loads.
    candidate_metrics = score_registered_version(model_name, candidate_version, validation_frame)
    registered = client.get_registered_model(model_name)
    champion_raw = registered.aliases.get(CHAMPION_ALIAS)
    previous = int(champion_raw) if champion_raw is not None else None
    champion_metrics: dict[str, float] | None = None

    if previous is not None:
        if previous == candidate_version:
            return ValidationDecision(
                False, "Candidate already has champion alias", candidate_version,
                previous, protocol_id, candidate_metrics, baseline_metrics, candidate_metrics,
            )
        champion = client.get_model_version(model_name, str(previous))
        if champion.tags.get(PROTOCOL_TAG) != protocol_id:
            reason = "Existing champion uses an incompatible validation protocol; no promotion"
            client.set_model_version_tag(model_name, str(candidate_version), DECISION_TAG, "rejected")
            client.set_model_version_tag(model_name, str(candidate_version), "validation.gate_reason", reason)
            return ValidationDecision(
                False, reason, candidate_version, previous, protocol_id,
                candidate_metrics, baseline_metrics, None,
            )
        champion_metrics = score_registered_version(model_name, previous, validation_frame)

    if not _beats(candidate_metrics, baseline_metrics, min_relative_improvement):
        accepted, reason = False, "Candidate failed validation baseline MAE/RMSE gate"
    elif champion_metrics is not None and not _beats(
        candidate_metrics, champion_metrics, min_relative_improvement
    ):
        accepted, reason = False, "Candidate failed validation champion MAE/RMSE gate"
    else:
        accepted, reason = True, "Candidate passed comparable validation gates"

    client.set_model_version_tag(
        model_name, str(candidate_version), DECISION_TAG, "accepted" if accepted else "rejected"
    )
    client.set_model_version_tag(model_name, str(candidate_version), "validation.gate_reason", reason)
    if accepted:
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, str(candidate_version))

    return ValidationDecision(
        accepted, reason, candidate_version, previous, protocol_id,
        candidate_metrics, baseline_metrics, champion_metrics,
    )
