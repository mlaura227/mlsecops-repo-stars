"""Unfitted, leakage-safe preprocessing components for model training."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors from the `stars` target without leaking labels."""
    target = frame[TARGET_COLUMN]
    if not isinstance(target, pd.Series):   # type: ignore
        raise TypeError("Expected a single target column")
    features = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    return features, target.copy()


def build_preprocessor() -> ColumnTransformer:
    """Return an UNFITTED transformer; the caller fits it on X_train only."""
    numeric = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    binary = SimpleImputer(strategy="most_frequent")
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("binary", binary, list(BOOLEAN_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
