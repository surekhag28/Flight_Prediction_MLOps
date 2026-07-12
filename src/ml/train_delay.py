"""
Model A: Flight delay risk classifier.

Algorithm: AutoML best model - LightGBM / XGBoost / CatBoost / RandomForest / HGB
                                selected by auto_hpo.py; default to lightgbm when not specified.

Target: delay_risk (proxy label, create_labels.py)
Metrics: AUC-ROC, F1, Precision, Recall

MLflow layout (inside the 'pipeline' experiment):
    pipeline_<timestamp> (parent run, created by DAG)
        training_delay   (child run, created here)
            tags: {algorithm: <best_algo_name>}
"""

from __future__ import annotations
import tempfile
import os
import json
import time
from datetime import datetime, UTC
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from src.config.config import get_settings
from src.ml.algorithms import CLASSIFIER_REGISTRY
from src.ml.training_utils import get_fs, load_parquet_data, resolve_algorithm
from src.utils.exceptions import InsufficientDataError, ModelTrainingError
from src.core.logger import get_logger
from src.utils.mlflow_utils import setup_mlflow, child_run
from src.ml.training_utils import standalone_run, fit_model

logger = get_logger(__name__)
settings = get_settings()

BUCKET = settings.miniosettings.bucket
GOLD_LABELS_PATH = f"{BUCKET}/gold/labels"
ARTIFACT_DIR = Path("/tmp/artifacts/delay")
TRAIN_CFG = settings.training.delay
MODEL_NAME = settings.mlflow.model_names.delay
PIPELINE_EXP_NAME = settings.mlflow.experiments.delay


def run(
    pipeline_run_id: str | None = None,
    pipeline_parent_run_id: str | None = None,
    best_params: dict | None = None,
    algorithm: str = "lgbm",
) -> dict:
    """
    Train the delay risk classifier as a 'training_delay' child run under pipeline_parent_run_id.

    Args:
        pipeline_run_id: str            Shared timestamp ID across the DAG.
        pipeline_parent_run_id: str     MLflow run ID of the pipeline parent run
                                        When None, runs without parent (standalone run)
        best_params:dict                Hyperparameters from HPO for best model. Falls back to
                                        default lgbm config
        algorithm:str                   Algorithm name from auto_hpo for best model.
                                        Falls back to default from config.

    Returns:
        {"pipeline_run_id":<run_id>, "run_id":<run_id>, "feature_columns":<feature_columns>, "algorithm":<algorithm>}
    """

    if pipeline_run_id is None:
        pipeline_run_id = datetime.now(tz=UTC).strftime("%d%m%y_%H%M%S")

    algorithm, spec = resolve_algorithm(algorithm, CLASSIFIER_REGISTRY, "lgbm")

    setup_mlflow()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fs = get_fs()

    try:
        df = load_parquet_data(GOLD_LABELS_PATH, fs, "delay label")
        logger.info(f"Delay risk data loaded from :{GOLD_LABELS_PATH}")
    except InsufficientDataError:
        logger.exception("Not enough data to train the model.")
        raise
    except FileNotFoundError:
        logger.exception(f"Unable to find file at location : {GOLD_LABELS_PATH}")
        raise
    except Exception as e:
        raise ModelTrainingError(
            f"Failed to load data files from : {GOLD_LABELS_PATH}"
        ) from e

    feature_cols = TRAIN_CFG.feature_columns
    target_col = TRAIN_CFG.target_column

    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < len(feature_cols):
        missing_cols = list(set(feature_cols) - set(available_cols))
        logger.warning(
            f"Missing delay features in data : missing columns = {missing_cols}"
        )

    feature_cols = available_cols
    df = df[feature_cols + [target_col]].dropna()

    X = df[feature_cols].values
    y = df[feature_cols].values

    logger.info(f"Total data points in delay label dataset: {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_CFG.max_rows, stratify=y
    )

    logger.info(
        f"Delay split data: train_size= {len(X_train)}, test_size= {len(X_test)}, delay_rate: {round(float(y_train.mean()),4)}, algorithm = {algorithm}"
    )

    if pipeline_parent_run_id:
        ctx = child_run(
            pipeline_parent_run_id,
            "training_delay",
            PIPELINE_EXP_NAME,
            tags={
                "model_type": "classifier",
                "pipeline_run_id": pipeline_run_id,
                "algorithm": algorithm,
            },
        )
    else:
        ctx = standalone_run(
            PIPELINE_EXP_NAME,
            f"training_delay_{pipeline_run_id}",
            pipeline_run_id,
            algorithm,
        )  # used when running the training script outside Airflow DAG

    try:
        with ctx as run:
            params = best_params if best_params else TRAIN_CFG.lgbm_params.copy()

            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("hpo_used", best_params is not None)
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_param("feature_columns", json.dumps(feature_cols))
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))

            t0 = time.time()
            model = spec.factory(params, TRAIN_CFG.random_state)
            fit_model(model, algorithm, X_train, y_train, X_test, y_test)
            train_time = time.time() - t0

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = {
                "auc_roc": float(roc_auc_score(y_test, y_prob)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "train_time_sec": round(train_time, 2),
            }

            mlflow.log_metrics(metrics)
            logger.info(
                f"Delay training completed, train_time: {train_time}, algorithm: {algorithm}, metrics: {metrics}"
            )

            feature_importances = spec.get_feature_importance(model, X_train)
            if feature_importances:
                importance_dict = dict(zip(feature_cols, feature_importances.tolist()))
                mlflow.log_dict(importance_dict, "feature_importance.json")
            else:
                logger.info(
                    f"Feature importances not available for algorithm : {algorithm}"
                )

            from src.ml.explain import log_shap_summary

            log_shap_summary(model, X_train, feature_cols, ARTIFACT_DIR)

            model_path = str(ARTIFACT_DIR / "delay_model.joblib")
            joblib.dump(model, model_path)

            with tempfile.TemporaryDirectory() as _tmpdir:
                mlflow.sklearn.save_model(model, _tmpdir + "/model")
                mlflow.log_artifact(_tmpdir + "/model", artifact_path="model")

            run_id = run.info.run_id
    except (InsufficientDataError, FileNotFoundError):
        raise
    except Exception as e:
        raise ModelTrainingError(
            f"Delay model training failed: pipeline_run_id: {pipeline_run_id}, algorithm: {algorithm}"
        )
    return {
        "pipeline_run_id": pipeline_run_id,
        "run_id": run_id,
        "algorithm": algorithm,
        "metrics": metrics,
        "feature_columns": feature_cols,
    }


""" if __name__ == "__main__":

    result = run(algorithm="lgbm")
    logger.info(result.keys())
    # print(len(result.keys()))
    # print(result.get("metrics"))
    # print(json.dumps(result, indent=2, default=str))
 """
