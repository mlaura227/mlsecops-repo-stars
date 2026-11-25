"""Regression tests for safe opt-in serving configuration in Compose."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compose_separates_serving_from_legacy_training() -> None:
    text = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "MLSECOPS_MODEL_NAME: github-stars-regressor" in text
    assert "MLSECOPS_SERVING_MODEL_NAME: ${MLSECOPS_SERVING_MODEL_NAME:-}" in text


def test_example_requires_explicit_serving_opt_in() -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "# MLSECOPS_SERVING_MODEL_NAME=github-stars-regressor-validation-v1" in example
    assert "\nMLSECOPS_SERVING_MODEL_NAME=" not in example
