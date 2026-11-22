"""CLI entry-point behavior."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from src.prepare_data import main


def test_cli_outputs_paths(
    sample_dataset: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "raw.csv"
    sample_dataset.to_csv(raw, index=False)
    monkeypatch.setattr(sys, "argv", ["prepare-data", "--input", str(raw), "--output-dir", str(tmp_path / "out")])
    main()
    assert "clean:" in capsys.readouterr().out
    assert (tmp_path / "out" / "train.csv").exists()
