"""Inference, guarded retraining, and MLflow audit logging business logic.

Each prediction creates an MLflow run containing request/response JSON and one
monitoring CSV row. This is auditable for coursework, but should be replaced
with buffered event storage at high prediction volume.
"""

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.sklearn import load_model as load_sklearn_model
from mlflow.tracking import MlflowClient

from src.api_schemas import (
    ModelStatusResponse, PredictRequest, PredictResponse, TrainRequest, TrainResponse,
)
from src.data import BOOLEAN_FEATURES, FEATURE_COLUMNS
from src.promotion import CHAMPION_ALIAS, PromotionDecision, promote_candidate
from src.train import TrainingRun, log_promotion_decision, train_experiment
from src.validation_promotion import DECISION_TAG, PROTOCOL_TAG


class ChampionUnavailable(Exception):
    """No promoted model is available for inference."""


class InvalidPrediction(Exception):
    """A registered model returned an invalid prediction."""


@dataclass(frozen=True)
class ApiSettings:
    """Trusted configuration, never accepted as user-supplied API paths."""

    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "github-stars-regression"
    model_name: str = "github-stars-regressor"
    # Optional independent serving registry. /train always uses model_name.
    serving_model_name: str | None = None
    train_path: Path = Path("data/processed/train.csv")
    test_path: Path = Path("data/processed/test.csv")
    train_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "ApiSettings":
        """Read deployment settings once per request (environment overrides)."""
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
            experiment_name=os.getenv("MLSECOPS_EXPERIMENT", "github-stars-regression"),
            model_name=os.getenv("MLSECOPS_MODEL_NAME", "github-stars-regressor"),
            serving_model_name=os.getenv("MLSECOPS_SERVING_MODEL_NAME", "").strip() or None,
            train_path=Path(os.getenv("MLSECOPS_TRAIN_CSV", "data/processed/train.csv")),
            test_path=Path(os.getenv("MLSECOPS_TEST_CSV", "data/processed/test.csv")),
            train_api_key=os.getenv("MLSECOPS_TRAIN_API_KEY") or None,
        )


def _champion_version(model_name: str) -> int:
    """Resolve an existing approved alias, never default to newest version."""
    try:
        registry = MlflowClient().get_registered_model(model_name)
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            raise ChampionUnavailable("No registered model exists; train and promote a model first") from error
        raise
    alias_version = registry.aliases.get(CHAMPION_ALIAS)
    if alias_version is None:
        raise ChampionUnavailable("No champion alias exists; run training with auto-promote first")
    try:
        version = int(alias_version)
    except (TypeError, ValueError) as error:
        raise ChampionUnavailable("The champion alias points to an invalid model version") from error
    if version <= 0:
        raise ChampionUnavailable("The champion alias points to an invalid model version")
    return version


def _serving_champion(settings: ApiSettings) -> tuple[str, int, str | None]:
    """Resolve a pinned champion, verifying approval for a separate registry.

    /train still targets settings.model_name. An explicitly selected different
    registry is accepted only if its champion passed the versioned gate.
    """
    model_name = settings.serving_model_name or settings.model_name
    version = _champion_version(model_name)
    if model_name == settings.model_name:
        return model_name, version, None

    registered_version = MlflowClient().get_model_version(model_name, str(version))
    tags = registered_version.tags
    protocol_id = tags.get(PROTOCOL_TAG)
    if (
        tags.get(DECISION_TAG) != "accepted"
        or not isinstance(protocol_id, str)
        or len(protocol_id) != 64
        or any(character not in "0123456789abcdef" for character in protocol_id)
    ):
        raise ChampionUnavailable(
            "Selected serving champion lacks an accepted validation gate or protocol tag"
        )
    return model_name, version, protocol_id


def model_status(settings: ApiSettings) -> ModelStatusResponse:
    """Show the *actual* serving registry without retraining or scoring data."""
    mlflow.set_tracking_uri(settings.tracking_uri)
    model_name, version, protocol_id = _serving_champion(settings)
    return ModelStatusResponse(
        serving_model_name=model_name,
        serving_model_version=version,
        training_model_name=settings.model_name,
        validation_protocol_id=protocol_id,
    )


def _model_features(payload: PredictRequest) -> dict[str, object]:
    """Match training-time 0/1 encoding for boolean fields exactly."""
    features: dict[str, object] = payload.model_dump()
    for column in BOOLEAN_FEATURES:
        features[column] = int(bool(features[column]))
    return features


def _log_prediction_event(
    *, settings: ApiSettings, request_id: str, timestamp: str,
    payload: PredictRequest, prediction: float, model_version: int,
    serving_model_name: str, protocol_id: str | None,
) -> str:
    """Store one full JSON audit event and drift-ready CSV as MLflow artifacts."""
    mlflow.set_experiment(f"{settings.experiment_name}-api")
    with mlflow.start_run(run_name=f"predict-{request_id}") as run:
        mlflow.set_tag("api.endpoint", "predict")
        mlflow.log_params({"model_name": serving_model_name, "model_version": model_version})
        if protocol_id is not None:
            mlflow.set_tag("validation.protocol_id", protocol_id)
        mlflow.log_dict(
            {
                "request_id": request_id,
                "timestamp_utc": timestamp,
                "request": payload.model_dump(),
                "prediction": prediction,
                "model_name": serving_model_name,
                "model_version": model_version,
                "validation_protocol_id": protocol_id,
            },
            "api/predict.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            row = {
                **_model_features(payload),
                "prediction": prediction,
                "model_version": model_version,
                "request_id": request_id,
                "timestamp_utc": timestamp,
            }
            pd.DataFrame([row]).to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="monitoring")
        return run.info.run_id


def predict_stars(payload: PredictRequest, settings: ApiSettings) -> PredictResponse:
    """Load the currently approved version and log its input and output."""
    mlflow.set_tracking_uri(settings.tracking_uri)
    model_name, version, protocol_id = _serving_champion(settings)
    model = load_sklearn_model(model_uri=f"models:/{model_name}/{version}")
    if model is None:
        raise InvalidPrediction("Registry returned no model")
    features = _model_features(payload)
    frame = pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])
    values = np.asarray(model.predict(frame), dtype=np.float64)
    if values.shape != (1,) or not np.isfinite(values).all() or values[0] < 0:
        raise InvalidPrediction("Registered model produced invalid predictions")
    prediction = float(values[0])
    request_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    audit_run_id = _log_prediction_event(
        settings=settings, request_id=request_id, timestamp=timestamp,
        payload=payload, prediction=prediction, model_version=version,
        serving_model_name=model_name, protocol_id=protocol_id,
    )
    return PredictResponse(
        request_id=request_id,
        prediction=prediction,
        model_name=model_name,
        model_version=version,
        audit_run_id=audit_run_id,
    )


def _select_training_runs(runs: list[TrainingRun]) -> tuple[TrainingRun, TrainingRun]:
    """Reject incomplete training output instead of promoting arbitrary runs."""
    baseline = next((run for run in runs if run.kind == "baseline"), None)
    challenger = next((run for run in runs if run.kind == "random_forest"), None)
    if baseline is None or challenger is None or challenger.registered_model_version is None:
        raise RuntimeError("Training did not produce a baseline and registered challenger")
    return baseline, challenger


def _log_training_event(
    settings: ApiSettings, payload: TrainRequest, baseline: TrainingRun,
    challenger: TrainingRun, decision: PromotionDecision | None,
) -> str:
    """Capture /train input, output, and promotion outcome as an MLflow artifact."""
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(f"{settings.experiment_name}-api")
    with mlflow.start_run(run_name="api-train") as run:
        mlflow.set_tag("api.endpoint", "train")
        mlflow.log_dict(
            {
                "request": payload.model_dump(),
                "baseline": {"run_id": baseline.run_id, "metrics": baseline.metrics},
                "challenger": {
                    "run_id": challenger.run_id,
                    "metrics": challenger.metrics,
                    "registered_model_version": challenger.registered_model_version,
                },
                "promotion": None if decision is None else {
                    "accepted": decision.accepted,
                    "reason": decision.reason,
                    "previous_champion_version": decision.previous_champion_version,
                },
            },
            "api/train.json",
        )
        return run.info.run_id


def train_models(payload: TrainRequest, settings: ApiSettings) -> TrainResponse:
    """Train via the existing pipeline and use the Step 3 promotion gate."""
    runs = train_experiment(
        train_path=settings.train_path,
        test_path=settings.test_path,
        tracking_uri=settings.tracking_uri,
        experiment_name=settings.experiment_name,
        registered_model_name=settings.model_name,
        random_state=payload.random_state,
    )
    baseline, challenger = _select_training_runs(runs)
    raw_version = challenger.registered_model_version
    if raw_version is None:
        raise RuntimeError("Challenger must have a registry version")
    # MLflow model-version identifiers can be strings even when annotated as int.
    version = int(raw_version)
    decision: PromotionDecision | None = None
    if payload.auto_promote:
        decision = promote_candidate(
            candidate_version=version,
            baseline_metrics=baseline.metrics,
            test_path=settings.test_path,
            model_name=settings.model_name,
            tracking_uri=settings.tracking_uri,
            min_relative_improvement=payload.min_mae_improvement,
        )
        log_promotion_decision(challenger.run_id, decision)
    audit_run_id = _log_training_event(settings, payload, baseline, challenger, decision)
    return TrainResponse(
        baseline_run_id=baseline.run_id,
        challenger_run_id=challenger.run_id,
        baseline_metrics=baseline.metrics,
        challenger_metrics=challenger.metrics,
        registered_model_version=version,
        promotion_accepted=None if decision is None else decision.accepted,
        promotion_reason=None if decision is None else decision.reason,
        audit_run_id=audit_run_id,
    )
