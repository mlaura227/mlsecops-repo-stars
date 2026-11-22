"""Reproducible regression training and framework-independent evaluation."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.data import TARGET_COLUMN
from src.preprocessing import build_preprocessor, split_features_target

ModelKind = Literal["baseline", "random_forest"]
MODEL_KINDS: tuple[ModelKind, ...] = ("baseline", "random_forest")


@dataclass(frozen=True)
class Evaluation:
    """Fitted end-to-end model and its held-out evaluation outputs."""

    pipeline: Pipeline
    metrics: dict[str, float]
    predictions: pd.DataFrame


def build_pipeline(kind: ModelKind, *, random_state: int = 42) -> Pipeline:
    """Build an unfitted model with preprocessing inside the CV-safe pipeline."""
    if kind == "baseline":
        regressor = DummyRegressor(strategy="median")
    elif kind == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, random_state=random_state, n_jobs=1
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("regressor", regressor)])


def fit_and_evaluate(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    kind: ModelKind,
    random_state: int = 42,
) -> Evaluation:
    """Fit on training rows only; calculate held-out metrics and drift-ready rows."""
    if train_frame.empty or len(test_frame) < 2:
        raise ValueError("Training data cannot be empty and test data requires at least two rows")

    x_train, y_train = split_features_target(train_frame)
    x_test, y_test = split_features_target(test_frame)
    pipeline = build_pipeline(kind, random_state=random_state)
    pipeline.fit(x_train, y_train)

    actual = np.asarray(y_test, dtype=np.float64)
    predicted = np.asarray(pipeline.predict(x_test), dtype=np.float64)
    metrics = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }
    rows = x_test.reset_index(drop=True).copy()
    rows[TARGET_COLUMN] = actual
    rows["prediction"] = predicted
    return Evaluation(pipeline=pipeline, metrics=metrics, predictions=rows)
