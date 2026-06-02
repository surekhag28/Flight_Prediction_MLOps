#!/bin/bash


uv add --group core pyspark pandas numpy pyarrow boto3 s3fs pydantic pyyaml requests great-expectations

uv add --group ml lightgbm xgboost catboost scikit-learn mlflow evidently optuna joblib shap matplotlib

uv add --group feast "feast[redis]"

uv add --group airflow apache-airflow apache-airflow-providers-apache-spark

uv add --group serving fastapi uvicorn prometheus-client slowapi httpx

uv add --group dashboard streamlit folium streamlit-folium pydeck plotly

uv add --group streaming faust-streaming kafka-python

uv add --group dev pytest ruff httpx pytest-asyncio pytest-cov bandit

uv sync --all-groups