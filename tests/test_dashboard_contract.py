"""Thin UI contract: all computations live in src.monitoring."""

from pathlib import Path


def test_dashboard_remains_thin_and_uses_report_engine() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = (project_root / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert "from src.monitoring import" in script
    assert "load_prediction_batch(" in script
    assert "create_drift_report(" in script
    assert "st.iframe(" in script
    assert "Holdout split (demo only)" in script
    assert "DataDriftPreset" not in script
    assert "MlflowClient" not in script
