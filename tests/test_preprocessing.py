"""Tests for train-only fitting, feature shapes and unknown categories."""

import numpy as np
import pandas as pd

from src.data import BOOLEAN_FEATURES, FEATURE_COLUMNS, validate_dataset
from src.preprocessing import build_preprocessor, split_features_target


def test_feature_target_separation(sample_dataset: pd.DataFrame) -> None:
    x, y = split_features_target(validate_dataset(sample_dataset))
    assert list(x.columns) == list(FEATURE_COLUMNS)
    assert "stars" not in x.columns
    assert y.tolist() == list(range(100, 132))


def test_preprocessor_handles_unseen_languages(sample_dataset: pd.DataFrame) -> None:
    train = validate_dataset(sample_dataset)
    x, _ = split_features_target(train)
    transformer = build_preprocessor()
    transformed_train = transformer.fit_transform(x)
    unseen = x.iloc[[0]].copy()
    unseen.loc[unseen.index[0], "language"] = "NeverSeenBefore"
    transformed_test = transformer.transform(unseen)
    assert transformed_train.shape[0] == len(train)
    assert transformed_test.shape == (1, transformed_train.shape[1])
    assert np.isfinite(transformed_test).all()
    assert not any("stars" in name for name in transformer.get_feature_names_out())
    assert len(BOOLEAN_FEATURES) == 4


def test_preprocessor_imputes_missing_numeric_values(sample_dataset: pd.DataFrame) -> None:
    x, _ = split_features_target(validate_dataset(sample_dataset))
    preprocessor = build_preprocessor().fit(x)
    sample = x.iloc[[0]].copy()
    sample.loc[sample.index[0], "forks"] = np.nan
    prediction_features = preprocessor.transform(sample)
    assert np.isfinite(prediction_features).all()
