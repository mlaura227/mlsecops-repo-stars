"""Thin HTTP adapter exposing guarded training and champion-only prediction."""

import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException

from src.api_schemas import PredictRequest, PredictResponse, TrainRequest, TrainResponse
from src.api_service import ApiSettings, ChampionUnavailable, InvalidPrediction, predict_stars, train_models

app = FastAPI(title="GitHub Stars MLOps API", version="0.4.0")


def get_settings() -> ApiSettings:
    """Resolve application configuration for dependency override in tests."""
    return ApiSettings.from_env()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check (does not assert model availability)."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    settings: Annotated[ApiSettings, Depends(get_settings)],
) -> PredictResponse:
    """Return a prediction only for the current promoted champion."""
    try:
        return predict_stars(payload, settings)
    except ChampionUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except InvalidPrediction as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/train", response_model=TrainResponse)
def train(
    payload: TrainRequest,
    settings: Annotated[ApiSettings, Depends(get_settings)],
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> TrainResponse:
    """Train and optionally promote; if configured, require a training API key."""
    expected = settings.train_api_key
    if expected is not None and not secrets.compare_digest(api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing training API key")
    return train_models(payload, settings)
