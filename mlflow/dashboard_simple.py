import streamlit as st
import mlflow
import pandas as pd
import os

st.set_page_config(page_title="MLflow Dashboard", layout="wide")
st.title("📊 MLflow Experiment Dashboard")

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")

if not os.path.exists("mlflow/mlflow.db"):
    st.warning("No MLflow database found. Please run training first!")
    st.stop()

experiment = mlflow.get_experiment_by_name("House_Price_Prediction")

if experiment:
    st.success(f"Connected to experiment: {experiment.name}")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if not runs.empty:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{runs['metrics.r2_score'].iloc[-1]:.4f}")
        with col2:
            st.metric("RMSE", f"{runs['metrics.rmse'].iloc[-1]:.2f}")
        with col3:
            st.metric("MAE", f"{runs['metrics.mae'].iloc[-1]:.2f}")
        with col4:
            st.metric("Total Runs", len(runs))

        # All runs
        st.subheader("All Runs")
        st.dataframe(runs)

        # Model versions
        st.subheader("Model Registry")
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions("name='HousePricePredictor'")
        if versions:
            data = []
            for v in versions:
                data.append({
                    "Version": v.version,
                    "Stage": v.current_stage,
                    "Created": pd.to_datetime(v.creation_timestamp, unit='ms')
                })
            st.dataframe(pd.DataFrame(data))
        else:
            st.info("No model versions found.")
    else:
        st.warning("No runs found.")
else:
    st.error("Experiment not found.")
