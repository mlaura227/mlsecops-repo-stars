"""Train-only fitting, model comparison and evaluation schema."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

from src.data import FEATURE_COLUMNS, validate_dataset
from src.training import MODEL_KINDS, build_pipeline, fit_and_evaluate


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_fit_and_evaluate_models(sample_dataset: pd.DataFrame, kind: str) -> None:
    clean = validate_dataset(sample_dataset)
    evaluated = fit_and_evaluate(clean.iloc[:24], clean.iloc[24:], kind=kind)
    assert set(evaluated.metrics) == {"mae", "rmse", "r2"}
    assert np.isfinite(list(evaluated.metrics.values())).all()
    assert list(evaluated.predictions.columns) == [*FEATURE_COLUMNS, "stars", "prediction"]
    assert len(evaluated.predictions) == 8
    assert evaluated.predictions["stars"].tolist() == list(range(124, 132))
    assert "stars" not in evaluated.pipeline.named_steps["preprocess"].feature_names_in_
    assert evaluated.metrics["rmse"] >= 0


def test_regressors_and_randomness() -> None:
    baseline = build_pipeline("baseline")
    forest = build_pipeline("random_forest", random_state=13)
    assert isinstance(baseline.named_steps["regressor"], DummyRegressor)
    assert isinstance(forest.named_steps["regressor"], RandomForestRegressor)
    assert forest.named_steps["regressor"].random_state == 13
    with pytest.raises(ValueError, match="Unknown model kind"):
        build_pipeline("incorrect")  # type: ignore[arg-type]  # Explicit runtime validation test.


@pytest.mark.parametrize("train_count,test_count", [(0, 8), (24, 1)])
def test_rejects_insufficient_rows(sample_dataset: pd.DataFrame, train_count: int, test_count: int) -> None:
    clean = validate_dataset(sample_dataset)
    with pytest.raises(ValueError, match="at least two rows|cannot be empty"):
        fit_and_evaluate(clean.iloc[:train_count], clean.iloc[-test_count:], kind="baseline")
