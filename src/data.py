"""Data contract, validation, and reproducible dataset preparation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "stars"
CATEGORICAL_FEATURES = ("language",)
NUMERIC_FEATURES = ("forks", "open_issues", "age_days")
BOOLEAN_FEATURES = ("has_coc", "has_contributing", "has_workflows", "has_readme")
FEATURE_COLUMNS = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES, *BOOLEAN_FEATURES)
DATASET_COLUMNS = ("language", TARGET_COLUMN, *NUMERIC_FEATURES, *BOOLEAN_FEATURES)


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of a raw dataset or raise on invalid input.

    Counts must be finite, non-negative integers. Boolean values are encoded as
    0/1, and language labels are stripped without merging distinct categories.
    """
    if frame.empty:
        raise ValueError("Dataset must contain at least one row")

    missing = set(DATASET_COLUMNS) - set(frame.columns)
    unexpected = set(frame.columns) - set(DATASET_COLUMNS)
    if missing or unexpected:
        raise ValueError(f"Invalid columns: missing={sorted(missing)}, unexpected={sorted(unexpected)}")

    clean = frame.loc[:, list(DATASET_COLUMNS)].copy()
    language = clean["language"].astype("string").str.strip()
    if language.isna().any() or language.eq("").any():
        raise ValueError("language must be a non-empty string")
    clean["language"] = language.astype(str)

    for column in (TARGET_COLUMN, *NUMERIC_FEATURES):
        # Explicit Series construction removes pandas' ambiguous scalar/array typing.
        numeric = pd.Series(
            pd.to_numeric(clean[column], errors="coerce"),
            index=clean.index,
        )
        values = numeric.to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite numbers")
        if (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"{column} must contain non-negative integers")
        clean[column] = numeric.astype("int64")

    for column in BOOLEAN_FEATURES:
        normalized = clean[column].astype("string").str.strip().str.lower()
        encoded = pd.Series(
            normalized.map({"true": 1, "false": 0, "1": 1, "0": 0}),
            index=clean.index,
        )

        if encoded.isna().any():
            raise ValueError(f"{column} must contain boolean values")
        clean[column] = encoded.astype("int8")

    return clean


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV and apply the shared data contract."""
    return validate_dataset(pd.read_csv(path))


def split_dataset(
    clean: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split already-validated rows, without fitting any transformation."""
    train, test = train_test_split(clean, test_size=test_size, random_state=random_state, shuffle=True)
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        raise TypeError("Expected DataFrame outputs from train_test_split")
    return train.copy(), test.copy()


def prepare_datasets(
    input_path: Path, output_dir: Path, *, test_size: float = 0.2, random_state: int = 42
) -> dict[str, Path]:
    """Write cleaned, train, and test CSVs ready for DVC tracking."""
    clean = load_dataset(input_path)
    train, test = split_dataset(clean, test_size=test_size, random_state=random_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "clean": output_dir / "repositories_clean.csv",
        "train": output_dir / "train.csv",
        "test": output_dir / "test.csv",
    }
    clean.to_csv(paths["clean"], index=False)
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    return paths
