"""CLI entry point for reproducible data preparation."""

import argparse
from pathlib import Path

from src.data import prepare_datasets


def main() -> None:
    """Validate source CSV and save a deterministic train/test split."""
    parser = argparse.ArgumentParser(description="Prepare the GitHub stars dataset")
    parser.add_argument("--input", type=Path, default=Path("data/raw/mlsecops_prep_set.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    outputs = prepare_datasets(args.input, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
