"""
Product B: Airspace Congestion Regressor

Algorithm: Champion model selected through AutoHPO (auto_hpo.py); defaults to LightGBM when not specified.
Target: congestion_score (0-1 continuous; computed in silver_to_gold_congestion.py)
Metrics: RMSE, MAE, R2

MLflow layout (inside the 'pipeline' experiment):
    congestion_<timestamp> (parent run created by DAG)
        training_congestion (child run, created here)
        Tags: algorithm=<champion_algorithm>
"""

from __future__ import annotations
import mlflow
import numpy as np
import pandas as pd
import json
import time
import tempfile
import joblib
import mlflow.sklearn
from datetime import datetime, UTC
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config.config import get_settings
from src.core.logger import get_logger
from src.ml.algorithms import REGRESSOR_REGISTRY
from src.ml.training_utils import (
    get_fs,
    fit_model,
    load_parquet_data,
    sample_parquet,
    resolve_algorithm,
    standalone_run,
)

from src.utils.exceptions import ModelTrainingError, InsufficientDataError
from src.utils.mlflow_utils import child_run, setup_mlflow

logger = get_logger(__name__)
settings = get_settings()

BUCKET = settings.miniosettings.bucket
GOLD_CONGESTION_PATH = f"{BUCKET}/gold/airport_congestion_features"
ARTIFACT_DIR = Path("/tmp/artifacts/congestion")
TRAIN_CFG = settings.training.congestion
MODEL_NAME = settings.mlflow.model_names.congestion
PIPELINE_EXP = settings.mlflow.experiments.pipeline
SAMPLE_TRAIN_ROWS = settings.training.congestion.sample_rows


def run(
    pipeline_run_id: str,
    pipeline_parent_run_id: str,
    best_params: dict | None = None,
    algorithm: str = "lgbm",
) -> dict:
    """Train the congestion score Regressor as child run under pipeline_parent_run_id

    Args:
        pipeline_run_id: str                Shared timestamp ID across the DAG
        pipeline_parent_run_id: str         Mlflow run ID of the pipeline parent run
        best_params: dict                   Hyperparamters from HPO
        algorithm: str                      Algorithm name from REGRESSOR_REGISTRY (default: "lgbm")

    Returns:
        {"pipeline_run_id", "run_id", "metrics","feature_columns","algorithm"}
    """

    if pipeline_run_id is None:
        pipeline_run_id = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    algorithm, spec = resolve_algorithm(algorithm, REGRESSOR_REGISTRY, "lgbm")

    fs = get_fs()
    setup_mlflow()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = sample_parquet(GOLD_CONGESTION_PATH, fs, SAMPLE_TRAIN_ROWS)
    except Exception as e:
        raise ModelTrainingError(
            f"Failed to load data files from : {GOLD_CONGESTION_PATH}"
        ) from e

    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["hour_of_day"] = df["event_timestamp"].dt.hour
    df["day_of_week"] = df["event_timestamp"].dt.dayofweek

    feature_cols = TRAIN_CFG.feature_columns
    target_col = TRAIN_CFG.target_column

    available_cols = [c for c in feature_cols if c in df.columns]
    df = df[available_cols + [target_col]].dropna()

    if len(df) > TRAIN_CFG.max_rows:
        df = df.sample(TRAIN_CFG.max_rows, random_state=TRAIN_CFG.random_state)

    X = df[available_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_CFG.test_size, random_state=TRAIN_CFG.random_state
    )

    logger.info(
        f"Congestion split ready: X_train = {len(X_train)}, X_test = {len(X_test)}"
    )

    if pipeline_parent_run_id:
        ctx = child_run(
            pipeline_parent_run_id,
            "training_congestion",
            PIPELINE_EXP,
            tags={
                "model_type": "regressor",
                "pipeline_run_id": pipeline_run_id,
                "algorithm": algorithm,
            },
        )
    else:
        ctx = standalone_run(
            PIPELINE_EXP,
            f"training_congestion_{pipeline_run_id}",
            pipeline_run_id,
            algorithm,
        )

    try:
        with ctx as run:
            params = best_params if best_params else TRAIN_CFG.lgbm_params
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("hpo_used", best_params is not None)
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_param("feature_columns", json.dumps(available_cols))
            mlflow.log_param("train_rows", len(X_train))

            t0 = time.time()
            model = spec.factory(params, TRAIN_CFG.random_state)
            fit_model(model, algorithm, X_train, y_train)
            train_time = time.time() - t0

            y_pred = np.clip(model.predict(X_test), 0.0, 1.0)
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
                "train_time_sec": round(train_time, 2),
            }

            mlflow.log_metrics(metrics)
            logger.info(
                f"Congestion training complete, algorithm={algorithm}, metrics={metrics}"
            )

            feature_importances = spec.get_feature_importance(model, X_train)
            if feature_importances is not None:
                importance_dict = dict(
                    zip(available_cols, feature_importances.tolist())
                )
                mlflow.log_dict(importance_dict, "feature_importance.json")
            else:
                logger.info(
                    f"Feature importances not available for algorithm: {algorithm}"
                )

            from src.ml.explain import log_shap_summary

            log_shap_summary(model, X_train, available_cols, ARTIFACT_DIR)

            model_path = str(ARTIFACT_DIR / "congestion_model.joblib")
            joblib.dump(model, model_path)

            with tempfile.TemporaryDirectory() as _tmpdir:
                mlflow.sklearn.save_model(model, _tmpdir + "/model")
                mlflow.log_artifact(_tmpdir + "/model", artifact_path="/model")

            run_id = run.info.run_id

    except Exception as e:
        raise ModelTrainingError(
            f"Congestion model training failed, pipeline_run_id = {pipeline_run_id}, algorithm={algorithm}"
        )

    return {
        "pipeline_run_id": pipeline_run_id,
        "run_id": run_id,
        "metrics": metrics,
        "feature_columns": available_cols,
        "algorithm": algorithm,
    }
