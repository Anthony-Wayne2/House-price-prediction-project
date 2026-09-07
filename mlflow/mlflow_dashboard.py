#!/usr/bin/env python3
"""
Simple MLflow dashboard to view experiment results.
"""
import streamlit as st
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="MLflow Dashboard", layout="wide")

st.title("📊 MLflow Experiment Dashboard")

# Use SQLite backend
mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")

# Check if database exists
if not os.path.exists("mlflow/mlflow.db"):
    st.warning("No MLflow database found. Please run training first!")
    st.stop()

# Get experiment
experiment = mlflow.get_experiment_by_name("House_Price_Prediction")

if experiment:
    st.success(f"Connected to experiment: {experiment.name}")

    # Get runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if not runs.empty:
        st.subheader("📈 Training Metrics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{runs['metrics.r2_score'].iloc[-1]:.4f}")
        with col2:
            st.metric("RMSE", f"{runs['metrics.rmse'].iloc[-1]:.2f}")
        with col3:
            st.metric("MAE", f"{runs['metrics.mae'].iloc[-1]:.2f}")
        with col4:
            st.metric("Total Runs", len(runs))

        # Show runs table
        st.subheader("📋 All Runs")
        display_cols = ['run_id', 'metrics.r2_score', 'metrics.rmse', 'metrics.mae', 'status']
        if available_cols:
        available_cols = [c for c in display_cols if c in runs.columns]
            st.dataframe(runs[available_cols])
        else:
            st.info("No run data available")

        # Plot metrics
        st.subheader("📊 Metrics Comparison")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        if 'metrics.r2_score' in runs.columns:
            axes[0].bar(range(len(runs)), runs['metrics.r2_score'])
            axes[0].set_title('R² Score')
            axes[0].set_ylabel('R²')

        if 'metrics.rmse' in runs.columns:
            axes[1].bar(range(len(runs)), runs['metrics.rmse'])
            axes[1].set_title('RMSE')
            axes[1].set_ylabel('RMSE')

        st.pyplot(fig)

        # Model versions
        st.subheader("📦 Model Registry")
        client = mlflow.tracking.MlflowClient()
            if model_versions:
        try:
            model_versions = client.search_model_versions("name='HousePricePredictor'")
                for mv in model_versions:
                data = []
                    data.append({
                        "Version": mv.version,
                        "Stage": mv.current_stage,
                        "Created": pd.to_datetime(mv.creation_timestamp, unit='ms'),
                        "Description": mv.description or "-"
                    })
                st.dataframe(pd.DataFrame(data))
            else:
                st.info("No model versions found.")
        except Exception as e:
            st.info(f"Model registry not available: {e}")
    else:
        st.warning("No runs found. Train a model first!")
else:
    st.error("Experiment not found. Run training first!")
