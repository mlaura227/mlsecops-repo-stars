"""Keep DVC reproducibility checks CI-safe on fresh pull requests."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_checks_dvc_metadata_without_remote_credentials() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "uv run --locked python -m src.reproducibility --mode metadata" in workflow
    assert "uv sync --locked" in workflow
    assert "dvc pull" not in workflow


def test_ci_builds_all_services_from_compose() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "docker compose config --quiet" in workflow
    assert "docker compose build" in workflow
