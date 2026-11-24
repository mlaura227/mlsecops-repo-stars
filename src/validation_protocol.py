"""Deterministic validation cohort and evaluation-protocol fingerprint.

This module does not touch the final test.csv or mutate MLflow aliases.
A model trained on all 800 train.csv rows must NOT be compared as an
independent champion on this module's 160-row inner-validation cohort.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROTOCOL_VERSION = "train-only-validation-v1"


@dataclass(frozen=True)
class ValidationProtocol:
    """Identify an immutable training/validation evaluation cohort."""

    protocol_version: str
    protocol_id: str
    source_sha256: str
    random_state: int
    validation_fraction: float
    fit_rows: int
    validation_rows: int
    validation_indices: tuple[int, ...]

    def manifest(self) -> dict[str, object]:
        """Produce an auditable manifest for MLflow artifact logging."""
        return dict(asdict(self), final_test_used_for_selection=False)


def fingerprint_file(path: Path) -> str:
    """Fingerprint the same source bytes that DVC versions."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_validation_cohort(
    frame: pd.DataFrame,
    *,
    source_sha256: str,
    random_state: int = 42,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, ValidationProtocol]:
    """Split train.csv only; return disjoint cohorts and a stable ID.

    The caller must retain the exact CSV hash with candidate model versions.
    Repeated decisions on the same cohort can overfit validation; the final
    test set must remain untouched until final, non-selection assessment.
    """
    if len(frame) < 30:
        raise ValueError("At least 30 rows are required for the validation protocol")
    if not frame.index.is_unique:
        raise ValueError("Input row indices must be unique")
    if not 0.1 <= validation_fraction <= 0.4:
        raise ValueError("validation_fraction must be between 0.1 and 0.4")
    if random_state < 0:
        raise ValueError("random_state must be non-negative")
    if len(source_sha256) != 64 or any(letter not in "0123456789abcdef" for letter in source_sha256):
        raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest")

    fit, validation = train_test_split(
        frame, test_size=validation_fraction, random_state=random_state, shuffle=True
    )
    if not isinstance(fit, pd.DataFrame) or not isinstance(validation, pd.DataFrame):
        raise TypeError("Expected DataFrame outputs from train_test_split")

    index_values = tuple(int(index) for index in validation.index)
    fingerprint_data = {
        "protocol_version": PROTOCOL_VERSION,
        "source_sha256": source_sha256,
        "random_state": random_state,
        "validation_fraction": validation_fraction,
        "validation_indices": index_values,
    }
    protocol_id = hashlib.sha256(
        json.dumps(fingerprint_data, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return fit.copy(), validation.copy(), ValidationProtocol(
        protocol_version=PROTOCOL_VERSION,
        protocol_id=protocol_id,
        source_sha256=source_sha256,
        random_state=random_state,
        validation_fraction=validation_fraction,
        fit_rows=len(fit),
        validation_rows=len(validation),
        validation_indices=index_values,
    )


def assert_comparable_protocols(
    candidate: ValidationProtocol, champion: ValidationProtocol
) -> None:
    """Fail closed before comparing metrics from incompatible evaluation cohorts."""
    if candidate.protocol_id != champion.protocol_id:
        raise ValueError(
            "Candidate and champion use different validation protocols; "
            "do not compare metrics or replace the champion"
        )


def prepare_validation_files(
    train_path: Path, output_dir: Path, *, random_state: int = 42,
    validation_fraction: float = 0.2,
) -> ValidationProtocol:
    """Persist inner split only; never load or change the final test data."""
    from src.data import load_dataset

    source = load_dataset(train_path)
    fit, validation, protocol = split_validation_cohort(
        source,
        source_sha256=fingerprint_file(train_path),
        random_state=random_state,
        validation_fraction=validation_fraction,
    )
    manifest_path = output_dir / "validation_protocol.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("protocol_id") != protocol.protocol_id:
            raise ValueError("Existing validation protocol differs; inspect before overwriting")
    output_dir.mkdir(parents=True, exist_ok=True)
    fit.to_csv(output_dir / "fit.csv", index=False)
    validation.to_csv(output_dir / "validation.csv", index=False)
    manifest_path.write_text(
        json.dumps(protocol.manifest(), indent=2), encoding="utf-8"
    )
    return protocol


def main() -> None:
    """Generate split artifacts for later MLflow candidate/champion comparisons."""
    parser = argparse.ArgumentParser(description="Prepare reproducible inner validation files")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=0.2)
    args = parser.parse_args()
    protocol = prepare_validation_files(
        args.train, args.output_dir,
        random_state=args.seed, validation_fraction=args.fraction,
    )
    print(json.dumps(protocol.manifest(), indent=2))


if __name__ == "__main__":
    main()
