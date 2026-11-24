"""Tests for deterministic, isolated, protocol-versioned validation cohorts."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.validation_protocol import (
    PROTOCOL_VERSION,
    assert_comparable_protocols,
    fingerprint_file,
    split_validation_cohort,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"stars": list(range(100)), "forks": list(range(100))})


def test_fingerprint_is_based_on_bytes(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    source.write_bytes(b"stars,forks\n1,2\n")
    assert fingerprint_file(source) == hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_bytes(b"stars,forks\n1,3\n")
    assert fingerprint_file(source) != hashlib.sha256(b"stars,forks\n1,2\n").hexdigest()


def test_validation_split_is_reproducible_and_disjoint(frame: pd.DataFrame) -> None:
    digest = hashlib.sha256(b"dataset-v1").hexdigest()
    train, validation, protocol = split_validation_cohort(
        frame, source_sha256=digest
    )
    next_train, next_validation, repeated = split_validation_cohort(
        frame, source_sha256=digest
    )
    assert len(train) == 80
    assert len(validation) == 20
    assert train.index.is_unique and validation.index.is_unique
    assert not set(train.index) & set(validation.index)
    assert set(train.index) | set(validation.index) == set(frame.index)
    pd.testing.assert_frame_equal(train, next_train)
    pd.testing.assert_frame_equal(validation, next_validation)
    assert repeated == protocol
    assert protocol.protocol_version == PROTOCOL_VERSION
    assert protocol.fit_rows == 80 and protocol.validation_rows == 20
    assert tuple(validation.index) == protocol.validation_indices
    assert protocol.manifest()["final_test_used_for_selection"] is False


def test_protocol_mismatch_is_rejected(frame: pd.DataFrame) -> None:
    digest = hashlib.sha256(b"dataset-v1").hexdigest()
    _, _, first = split_validation_cohort(frame, source_sha256=digest)
    _, _, alternative_seed = split_validation_cohort(frame, source_sha256=digest, random_state=1)
    _, _, alternative_data = split_validation_cohort(
        frame, source_sha256=hashlib.sha256(b"dataset-v2").hexdigest()
    )
    assert_comparable_protocols(first, first)
    for incompatible in (alternative_seed, alternative_data):
        with pytest.raises(ValueError, match="different validation protocols"):
            assert_comparable_protocols(first, incompatible)


@pytest.mark.parametrize("fraction", [0.0, 0.09, 0.41, 1.0])
def test_invalid_fraction_rejected(frame: pd.DataFrame, fraction: float) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        split_validation_cohort(
            frame, source_sha256="a" * 64, validation_fraction=fraction
        )


@pytest.mark.parametrize("source_sha256", ["", "A" * 64, "0" * 63, "x" * 64])
def test_invalid_hash_rejected(frame: pd.DataFrame, source_sha256: str) -> None:
    with pytest.raises(ValueError, match="source_sha256"):
        split_validation_cohort(frame, source_sha256=source_sha256)


def test_rejects_small_frame(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="At least 30"):
        split_validation_cohort(frame.iloc[:10], source_sha256="a" * 64)


def test_rejects_duplicate_index(frame: pd.DataFrame) -> None:
    duplicate = frame.copy()
    duplicate.index = [0] * len(duplicate)
    with pytest.raises(ValueError, match="indices must be unique"):
        split_validation_cohort(duplicate, source_sha256="a" * 64)


def test_rejects_negative_seed(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="random_state"):
        split_validation_cohort(frame, source_sha256="a" * 64, random_state=-1)


def test_splits_have_independent_dataframes(frame: pd.DataFrame) -> None:
    train, validation, _ = split_validation_cohort(frame, source_sha256="a" * 64)
    original = frame.copy()
    train.iloc[0, 0] = -1
    validation.iloc[0, 0] = -2
    pd.testing.assert_frame_equal(frame, original)


def test_prepare_files_persists_manifest_and_rejects_protocol_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame
) -> None:
    import src.data as data
    from src.validation_protocol import prepare_validation_files

    monkeypatch.setattr(data, "load_dataset", lambda path: frame.copy())
    source = tmp_path / "train.csv"
    source.write_text("dummy source", encoding="utf-8")
    output_dir = tmp_path / "validation"
    protocol = prepare_validation_files(source, output_dir)
    assert (output_dir / "fit.csv").exists()
    assert (output_dir / "validation.csv").exists()
    assert (output_dir / "validation_protocol.json").exists()
    assert len(pd.read_csv(output_dir / "fit.csv")) == 80
    assert len(pd.read_csv(output_dir / "validation.csv")) == 20
    assert prepare_validation_files(source, output_dir).protocol_id == protocol.protocol_id
    with pytest.raises(ValueError, match="Existing validation protocol differs"):
        prepare_validation_files(source, output_dir, random_state=100)


def test_cli_prepares_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame,
                            capsys: pytest.CaptureFixture[str]) -> None:
    import json
    import sys
    import src.data as data
    import src.validation_protocol as protocol_module

    monkeypatch.setattr(data, "load_dataset", lambda path: frame.copy())
    source = tmp_path / "train.csv"
    source.write_text("dummy source", encoding="utf-8")
    output_dir = tmp_path / "validation"
    monkeypatch.setattr(sys, "argv", ["validation-protocol", "--train", str(source),
                                      "--output-dir", str(output_dir), "--seed", "7",
                                      "--fraction", "0.25"])
    protocol_module.main()
    output = json.loads(capsys.readouterr().out)
    assert output["random_state"] == 7
    assert output["validation_rows"] == 25
    assert output["protocol_id"]
