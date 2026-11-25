"""Config tests that protect explicit training migration and rollback."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_compose_has_opt_in_workflow() -> None:
    text = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "MLSECOPS_TRAIN_WORKFLOW: ${MLSECOPS_TRAIN_WORKFLOW:-legacy}" in text
    assert "MLSECOPS_MODEL_NAME: github-stars-regressor" in text
    assert "MLSECOPS_SERVING_MODEL_NAME: ${MLSECOPS_SERVING_MODEL_NAME:-}" in text


def test_sample_config_preserves_legacy_default() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "# MLSECOPS_TRAIN_WORKFLOW=validation" in text
    assert "\nMLSECOPS_TRAIN_WORKFLOW=validation" not in text
