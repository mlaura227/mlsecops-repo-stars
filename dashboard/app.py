"""Thin Streamlit view; monitoring logic and MLflow access live in src/."""

import streamlit as st

from src.monitoring import (
    MIN_CURRENT_ROWS,
    create_drift_report,
    dataset_features,
    load_prediction_batch,
    settings_from_env,
)


def main() -> None:
    """Interactive report for actual logged traffic or an explicit demo split."""
    st.set_page_config(page_title="GitHub Stars — Drift Monitoring", layout="wide")
    st.title("GitHub Stars · Data Drift Monitoring")
    st.caption("Evidently comparison of model inputs; labels and predictions are excluded.")
    tracking_uri, experiment_name, train_path, test_path = settings_from_env()
    mode = st.radio(
        "Current data source",
        ["Logged API predictions", "Holdout split (demo only)"],
        horizontal=True,
    )
    limit = st.slider("Maximum recent prediction requests", 10, 500, 200, step=10)

    try:
        reference = dataset_features(train_path)
        if mode == "Logged API predictions":
            batch = load_prediction_batch(
                tracking_uri=tracking_uri, experiment_name=experiment_name, limit=limit,
            )
            current = batch.features
            if batch.skipped_runs:
                st.warning(f"Ignored {batch.skipped_runs} incomplete or invalid prediction runs.")
            source = "api"
        else:
            current = dataset_features(test_path)
            source = "holdout-demo"
            st.info("Demo only: this is the held-out split, not live production traffic.")
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Monitoring data unavailable: {error}")
        return

    a, b = st.columns(2)
    a.metric("Reference rows (training split)", len(reference))
    b.metric("Current feature rows", len(current))
    st.dataframe(current.head(10), use_container_width=True)

    if len(current) < MIN_CURRENT_ROWS:
        st.warning(
            f"Only {len(current)} logged predictions found. Generate at least "
            f"{MIN_CURRENT_ROWS} /predict requests, or select the explicitly labeled demo mode."
        )
        return

    if st.button("Generate and log Evidently report", type="primary"):
        try:
            result = create_drift_report(
                reference=reference, current=current, tracking_uri=tracking_uri,
                experiment_name=experiment_name, source=source,
            )
        except Exception as error:
            # Streamlit boundary: show the operational error instead of a blank page.
            st.error(f"Evidently / MLflow report generation failed: {error}")
            return
        st.success(f"Report logged in MLflow run {result.report_run_id}")
        st.download_button("Download HTML report", result.html, file_name="data_drift.html", mime="text/html")
        st.iframe(result.html, height=900)


if __name__ == "__main__":
    main()
