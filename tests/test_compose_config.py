"""Static deployment contracts and optional Docker Compose schema checks."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_docker_build_uses_python_313_and_locked_uv_dependencies() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.13-slim" in dockerfile
    assert "uv sync --locked --no-dev --no-install-project" in dockerfile
    assert "COPY src/ ./src/" in dockerfile
    assert "USER mlsecops" in dockerfile


def test_compose_includes_tracking_api_and_preparation() -> None:
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "mlflow:" in compose
    assert "prepare-data:" in compose
    assert "api:" in compose
    assert "dashboard:" in compose
    assert "127.0.0.1:8501:8501" in compose
    assert "COPY dashboard/ ./dashboard/" in (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "service_completed_successfully" in compose
    assert "service_healthy" in compose
    assert "http://mlflow:5000" in compose
    assert "sqlite:////mlflow/mlflow.db" in compose
    assert "mlflow-state:/mlflow" in compose
    assert "./data:/app/data:ro" in compose
    assert '127.0.0.1:8000:8000' in compose
    assert '127.0.0.1:5000:5000' in compose


def test_compose_does_not_bake_local_data_or_secrets_into_image() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "data/" in dockerignore.splitlines()
    assert ".env" in dockerignore.splitlines()
    assert ".venv" in dockerignore.splitlines()
    assert ".git" in dockerignore.splitlines()


def test_docker_compose_resolves_to_valid_services_when_available() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not installed in this test environment")
    result = subprocess.run(
        ["docker", "compose", "--file", "compose.yaml", "config", "--format", "json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    config = json.loads(result.stdout)
    services = config["services"]
    assert set(services) == {"api", "mlflow", "prepare-data", "dashboard"}
    assert services["api"]["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert services["dashboard"]["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert services["api"]["depends_on"]["mlflow"]["condition"] == "service_healthy"
    assert services["api"]["depends_on"]["prepare-data"]["condition"] == "service_completed_successfully"
