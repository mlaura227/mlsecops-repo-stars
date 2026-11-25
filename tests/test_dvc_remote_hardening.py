"""Tests for shared DVC configuration safety and fresh-clone evidence."""

from pathlib import Path

import pytest

from src.reproducibility import (
    committed_remote_has_credentials,
    local_default_override,
    verify_repository,
)


def setup_repo(root: Path) -> Path:
    for name in ("pyproject.toml", "uv.lock", "compose.yaml"):
        (root / name).write_text("stub\n", encoding="utf-8")
    (root / ".python-version").write_text("3.13\n", encoding="utf-8")
    (root / ".dvc").mkdir()
    (root / ".dvc/config").write_text(
        '[core]\n remote = shared\n[remote "shared"]\n'
        ' url = s3://dvc\n endpointurl = https://example.org/bucket.s3\n',
        encoding="utf-8",
    )
    (root / "data/raw").mkdir(parents=True)
    (root / "data/raw/input.csv.dvc").write_text("outs: []\n", encoding="utf-8")
    (root / "data/raw/mlsecops_prep_set.csv.dvc").write_text("outs: []\n", encoding="utf-8")
    return root


def test_shared_remote_without_local_override(tmp_path: Path) -> None:
    root = setup_repo(tmp_path)
    assert local_default_override(root) is None
    assert verify_repository(root, "shared-remote").ok


def test_local_demo_remote_detected(tmp_path: Path) -> None:
    root = setup_repo(tmp_path)
    (root / ".dvc/config.local").write_text(
        '[core]\n remote = demo\n[remote "demo"]\n url = C:/data\n', encoding="utf-8"
    )
    assert local_default_override(root) == "demo"
    errors = verify_repository(root, "shared-remote").errors
    assert any("overrides" in error for error in errors)
    assert any("--unset --local" in error for error in errors)


def test_matching_local_remote_is_valid(tmp_path: Path) -> None:
    root = setup_repo(tmp_path)
    (root / ".dvc/config.local").write_text(
        '[core]\n remote = shared\n', encoding="utf-8"
    )
    assert verify_repository(root, "shared-remote").ok


def test_local_credentials_do_not_fail_shared_preflight(tmp_path: Path) -> None:
    root = setup_repo(tmp_path)
    (root / ".dvc/config.local").write_text(
        '[remote "shared"]\n access_key_id = secret\n', encoding="utf-8"
    )
    assert verify_repository(root, "shared-remote").ok


@pytest.mark.parametrize("key", ["access_key_id", "secret_access_key", "password", "token"])
def test_committed_credentials_are_rejected(tmp_path: Path, key: str) -> None:
    root = setup_repo(tmp_path)
    with (root / ".dvc/config").open("a", encoding="utf-8") as config_file:
        config_file.write(f" {key} = must-not-commit\n")
    assert committed_remote_has_credentials(root / ".dvc/config")
    assert any("credentials" in item for item in verify_repository(root, "shared-remote").errors)


def test_no_committed_credentials(tmp_path: Path) -> None:
    root = setup_repo(tmp_path)
    assert not committed_remote_has_credentials(root / ".dvc/config")
    assert not committed_remote_has_credentials(root / "missing")


def test_malformed_local_config_fails_closed(tmp_path: Path) -> None:
    root = setup_repo(tmp_path)
    (root / ".dvc/config.local").write_text("[invalid", encoding="utf-8")
    assert local_default_override(root) == "<invalid local config>"
    assert not verify_repository(root, "shared-remote").ok


def test_malformed_committed_config_is_not_treated_as_credentials(tmp_path: Path) -> None:
    root = setup_repo(tmp_path)
    (root / ".dvc/config").write_text("[broken", encoding="utf-8")
    assert not committed_remote_has_credentials(root / ".dvc/config")
    assert not verify_repository(root, "shared-remote").ok


def test_fresh_clone_script_checks_actual_pull() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/verify_fresh_clone.ps1").read_text(encoding="utf-8")
    for token in ("git clone", "uv sync --locked", "dvc pull", "--mode data", "--mode shared-remote"):
        assert token in script
    assert "Remove-Item" in script
    assert "AWS_SECRET_ACCESS_KEY" in script


def test_manual_workflow_is_opt_in_only() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / ".github/workflows/dvc-remote.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in script
    assert "pull_request:" not in script
    assert "DVC_ACCESS_KEY_ID" in script
    assert "DVC_SECRET_ACCESS_KEY" in script
    assert "dvc pull" in script
