"""
Product C: Flight Anomaly Detector

Algorithm: AutoML Champion - Isolation Forest/LocalOutlierFactor (novelty=True)
            Selected by auto_hpo.py; defaults to IsolationForest when not specified.


Anomaly Signals:
    - Sudden heading changes (>90 degrees in one snapshot interval)
    - Abnormal speed at altitude combinations
    - Unusual vertical rate patterns
    - Very short distance  travelled between snapshots at cruise altitude

contaimination=0.05 means we expect ~5% anomalous flight patterns globally.

MLflow layout (inside the 'pipeline' experiment):
    pipeline_<timestamp>    (parent run, created by DAG)
        training_anomaly    (child run, created by DAG)
        Tags:  algorithm=<champion_algorithm>
"""

from __future__ import annotations
import json
import time
import tempfile
from datetime import datetime, UTC
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from src.config.config import get_settings
from src.core.logger import get_logger
from src.ml.algorithms import ANOMALY_REGISTRY
from src.ml.training_utils import (
    get_fs,
    fit_model,
    load_parquet_data,
    sample_parquet,
    resolve_algorithm,
    standalone_run,
)
from src.utils.exceptions import InsufficientDataError, MlflowError, ModelTrainingError
from src.utils.mlflow_utils import child_run, setup_mlflow

logger = get_logger(__name__)
settings = get_settings()

BUCKET = settings.miniosettings.bucket
GOLD_FLIGHTS_PATH = f"{BUCKET}/gold/flight_state_features"
ARTIFACT_DIR = Path("/tmp/artifacts/anomaly")
TRAIN_CFG = settings.training.anomaly
MODEL_NAME = settings.mlflow.model_names.anomaly
PIPELINE_EXP = settings.mlflow.experiments.pipeline
SAMPLE_ROWS = settings.training.anomaly.sample_rows


def run(
    pipeline_run_id: str,
    pipeline_parent_run_id: str,
    best_params: dict | None = None,
    algorithm: str = "isolation_forest",
) -> dict:
    """
    Train the anomaly detector as a 'training_anomaly' child run under pipeline_parent_run_id

    Args:
        pipeline_run_id:str     Shared timestamp ID across the DAG
        pipeline_parent_run_id:str  Mlflow run ID of the pipeline parent run.
        best_params:dict             Hyperparameters from HPO
        algorithm                   Algorithm name from ANOMALY_REGISTRY; default as 'isolation_forest'

    Returns:
        {"pipeline_run_id","run_id","metrics","features_columns","algorithm"}
    """

    if pipeline_run_id is None:
        pipeline_run_id = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    algorithm, spec = resolve_algorithm(algorithm, ANOMALY_REGISTRY, "isolation_forest")

    setup_mlflow()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fs = get_fs()

    try:
        df = sample_parquet(f"{BUCKET}/gold/flight_state_features", fs, SAMPLE_ROWS)
        logger.info(f"Total data points: {len(df)}")
    except (FileNotFoundError, OSError, InsufficientDataError) as e:
        raise ModelTrainingError(
            f"Failed to load data from MinIO with pipeline_run_id={pipeline_run_id}"
        ) from e

    feature_cols = TRAIN_CFG.feature_columns
    available = [c for c in feature_cols if c in df.columns]
    df = df[available].dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(df[available].values)
    logger.info(f"Anomaly features scaled: rows = {len(X)} and features = {available}")

    if pipeline_parent_run_id:
        ctx = child_run(
            pipeline_parent_run_id,
            "training_anomaly",
            PIPELINE_EXP,
            tags={
                "model_type": "anomaly_detector",
                "pipeline_run_id": pipeline_run_id,
                "algorithm": algorithm,
            },
        )
    else:
        ctx = standalone_run(
            PIPELINE_EXP,
            f"training_anomaly_{pipeline_run_id}",
            pipeline_run_id,
            algorithm,
        )

    try:
        with ctx as run:
            if algorithm == "isolation_forest":
                default_params = {
                    "n_estimators": TRAIN_CFG.n_estimators,
                    "contamination": TRAIN_CFG.contamination,
                    "max_features": 1.0,
                    "bootstrap": False,
                }  # for default isolation_forest algorithm
            else:
                default_params = {
                    "n_neighbors": 20,
                    "contamination": TRAIN_CFG.contamination,
                }

            params = {**default_params, **(best_params or {})}

            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("hpo_used", best_params is None)
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_param("feature_columns", feature_cols)
            mlflow.log_param("training_rows", len(X))

            t0 = time.time()
            model = spec.factory(params, TRAIN_CFG.random_state)
            model.fit(X)
            train_time = time.time() - t0

            raw_scores = model.decision_function(X)
            predictions = model.predict(X)
            anomaly_count = int((predictions == -1).sum())

            metrics = {
                "anomaly_count": anomaly_count,
                "anomaly_rate": float(anomaly_count / len(predictions)),
                "mean_score": float(raw_scores.mean()),
                "score_spread": float(raw_scores.std()),
                "train_time": train_time,
                "training_rows": len(X),
            }
            mlflow.log_metrics(metrics)
            logger.info(
                f"Anomaly detection training completed, algorithm={algorithm}, metrics= {metrics}"
            )

            bundle = {
                "model": model,
                "scaler": scaler,
                "feature_columns": available,
                "algorithm": algorithm,
            }
            bundle_path = str(ARTIFACT_DIR / "anomaly_bundle.joblib")
            joblib.dump(bundle, bundle_path)
            mlflow.log_artifact(bundle_path, "model")

            # Log as sklearn pipeline in MLflow registry
            step_name = algorithm.replace("_", "")

            sklearn_pipeline = SklearnPipeline([("scaler", scaler), (step_name, model)])

            with tempfile.TemporaryDirectory() as _tmpdir:
                mlflow.sklearn.save_model(
                    sklearn_pipeline, _tmpdir + "/sklearn-pipeline"
                )
                mlflow.log_artifact(
                    _tmpdir + "/sklearn-pipeline", artifact_path="sklearn-pipeline"
                )

        run_id = run.info.run_id

    except Exception as e:
        raise ModelTrainingError(f"Anomaly detection model training failed") from e

    return {
        "pipeline_run_id": pipeline_run_id,
        "run_id": run_id,
        "metrics": metrics,
        "feature_columns": feature_cols,
        "algorithm": algorithm,
    }
