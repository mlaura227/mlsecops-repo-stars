"""Contract, determinism and export tests."""

import pandas as pd
import pytest

from src.data import DATASET_COLUMNS, load_dataset, prepare_datasets, split_dataset, validate_dataset


def test_normalizes_columns_and_types(sample_dataset: pd.DataFrame) -> None:
    clean = validate_dataset(sample_dataset)
    assert tuple(clean.columns) == DATASET_COLUMNS
    assert clean["language"].iloc[0] == "Python"
    assert clean["has_coc"].tolist()[:4] == [1, 0, 1, 0]
    assert clean["has_contributing"].tolist()[:4] == [0, 1, 0, 1]
    assert clean["has_workflows"].tolist()[:4] == [1, 0, 1, 0]
    assert clean["has_readme"].tolist()[:4] == [1, 0, 1, 0]
    assert sample_dataset["language"].iloc[0] == " Python "  # no in-place mutation


@pytest.mark.parametrize("column", ["language", "stars", "forks"])
def test_missing_column_raises(sample_dataset: pd.DataFrame, column: str) -> None:
    with pytest.raises(ValueError, match="Invalid columns"):
        validate_dataset(sample_dataset.drop(columns=column))


def test_unexpected_column_raises(sample_dataset: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unexpected"):
        validate_dataset(sample_dataset.assign(owner="private"))


def test_empty_dataset_raises(sample_dataset: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least one row"):
        validate_dataset(sample_dataset.iloc[0:0])


@pytest.mark.parametrize("value", [None, "  "])
def test_invalid_language_raises(sample_dataset: pd.DataFrame, value: object) -> None:
    sample_dataset.loc[0, "language"] = value
    with pytest.raises(ValueError, match="language"):
        validate_dataset(sample_dataset)


@pytest.mark.parametrize(
    ("value", "message"),
    [(None, "finite"), (float("inf"), "finite"), ("wrong", "finite"), (-1, "non-negative"), (2.5, "non-negative")],
)
def test_invalid_numeric_raises(sample_dataset: pd.DataFrame, value: object, message: str) -> None:
    sample_dataset["stars"] = sample_dataset["stars"].astype(object)
    sample_dataset.loc[0, "stars"] = value
    with pytest.raises(ValueError, match=message):
        validate_dataset(sample_dataset)


@pytest.mark.parametrize("value", [None, "occasionally", 2])
def test_invalid_boolean_raises(sample_dataset: pd.DataFrame, value: object) -> None:
    sample_dataset.loc[0, "has_readme"] = value
    with pytest.raises(ValueError, match="has_readme"):
        validate_dataset(sample_dataset)


def test_split_is_deterministic_and_disjoint(sample_dataset: pd.DataFrame) -> None:
    clean = validate_dataset(sample_dataset)
    first_train, first_test = split_dataset(clean)
    again_train, again_test = split_dataset(clean)
    assert len(first_train) == 25
    assert len(first_test) == 7
    assert first_train.index.intersection(first_test.index).empty
    pd.testing.assert_frame_equal(first_train, again_train)
    pd.testing.assert_frame_equal(first_test, again_test)


def test_csv_roundtrip_and_exports(sample_dataset: pd.DataFrame, tmp_path: object) -> None:
    from pathlib import Path

    directory = Path(str(tmp_path))
    raw = directory / "raw.csv"
    sample_dataset.to_csv(raw, index=False)
    loaded = load_dataset(raw)
    outputs = prepare_datasets(raw, directory / "processed")
    assert set(outputs) == {"clean", "train", "test"}
    assert len(pd.read_csv(outputs["clean"])) == len(loaded)
    train = pd.read_csv(outputs["train"])
    test = pd.read_csv(outputs["test"])
    assert len(train) + len(test) == len(loaded)
    assert set(train["stars"]).isdisjoint(set(test["stars"]))
