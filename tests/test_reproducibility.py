"""Unit tests for CI-safe DVC and environment preflight checks."""

from pathlib import Path

import pytest

from src.reproducibility import main, shared_remote_configured, verify_repository


def make_repository(root: Path) -> Path:
    """Build a minimal tracked repository without contacting a remote."""
    for name in ("pyproject.toml", "uv.lock", "compose.yaml"):
        (root / name).write_text("stub\n", encoding="utf-8")
    (root / ".python-version").write_text("3.13\n", encoding="utf-8")
    (root / ".dvc").mkdir()
    (root / ".dvc/config").write_text("[core]\n", encoding="utf-8")
    (root / "data/raw").mkdir(parents=True)
    (root / "data/raw/mlsecops_prep_set.csv.dvc").write_text("outs: []\n", encoding="utf-8")
    return root


def test_metadata_check_is_sufficient_without_actual_data(tmp_path: Path) -> None:
    result = verify_repository(make_repository(tmp_path))
    assert result.ok
    assert result.errors == ()
    assert "not checked" in result.notes[0]


def test_missing_repository_metadata_is_reported(tmp_path: Path) -> None:
    result = verify_repository(tmp_path)
    assert not result.ok
    assert any("uv.lock" in error for error in result.errors)
    assert any("Python 3.13" in error for error in result.errors)
    assert any("DVC pointer" in error for error in result.errors)


def test_wrong_python_version_is_rejected(tmp_path: Path) -> None:
    make_repository(tmp_path)
    (tmp_path / ".python-version").write_text("3.12", encoding="utf-8")
    assert any("3.13" in error for error in verify_repository(tmp_path).errors)


def test_raw_dataset_pointer_is_required(tmp_path: Path) -> None:
    make_repository(tmp_path)
    pointer = tmp_path / "data/raw/mlsecops_prep_set.csv.dvc"
    pointer.unlink()
    (tmp_path / "data/processed.dvc").write_text("outs: []", encoding="utf-8")
    errors = verify_repository(tmp_path).errors
    assert any("Raw dataset" in error for error in errors)


@pytest.mark.parametrize("pointer", ["data/raw.dvc", "data.dvc"])
def test_directory_pointer_is_accepted(tmp_path: Path, pointer: str) -> None:
    make_repository(tmp_path)
    (tmp_path / "data/raw/mlsecops_prep_set.csv.dvc").unlink()
    path = tmp_path / pointer
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("outs: []", encoding="utf-8")
    assert verify_repository(tmp_path).ok


def test_data_mode_reports_missing_files(tmp_path: Path) -> None:
    result = verify_repository(make_repository(tmp_path), mode="data")
    assert not result.ok
    assert any("data/processed/train.csv" in error for error in result.errors)
    assert any("dvc pull" in error for error in result.errors)


def test_data_mode_passes_when_all_files_exist(tmp_path: Path) -> None:
    from src.reproducibility import REQUIRED_DATA_FILES

    make_repository(tmp_path)
    for filename in REQUIRED_DATA_FILES:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
    assert verify_repository(tmp_path, mode="data").ok


def test_local_only_remote_is_not_considered_shared(tmp_path: Path) -> None:
    make_repository(tmp_path)
    (tmp_path / ".dvc/config.local").write_text(
        '[core]\n remote = demo\n[remote "demo"]\n url = C:/Data/DVC\n', encoding="utf-8"
    )
    assert not shared_remote_configured(tmp_path / ".dvc/config")
    assert not verify_repository(tmp_path, mode="shared-remote").ok


def test_shared_remote_in_committed_config_passes(tmp_path: Path) -> None:
    make_repository(tmp_path)
    (tmp_path / ".dvc/config").write_text(
        '[core]\n remote = shared\n[remote "shared"]\n url = s3://my-bucket/data\n', encoding="utf-8"
    )
    result = verify_repository(tmp_path, mode="shared-remote")
    assert result.ok
    assert "not tested" in result.notes[0]


@pytest.mark.parametrize("uri", ["file:///tmp/dvc", "C:/Users/Name/DVC", "/tmp/dvc", ""])
def test_local_or_empty_remote_url_is_rejected(tmp_path: Path, uri: str) -> None:
    config = tmp_path / "config"
    config.write_text(f'[core]\n remote = demo\n[remote "demo"]\n url = {uri}\n', encoding="utf-8")
    assert not shared_remote_configured(config)


def test_invalid_remote_configuration_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.write_text("[core\n", encoding="utf-8")
    assert not shared_remote_configured(config)
    assert not shared_remote_configured(tmp_path / "absent")


def test_cli_reports_errors_and_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(tmp_path), "--mode", "metadata"]) == 1
    assert "FAIL: reproducibility preflight" in capsys.readouterr().out


def test_cli_reports_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_repository(tmp_path)
    assert main(["--root", str(tmp_path), "--mode", "metadata"]) == 0
    assert "PASS: reproducibility preflight" in capsys.readouterr().out
