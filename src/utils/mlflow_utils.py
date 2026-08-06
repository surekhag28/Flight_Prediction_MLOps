"""
MLflow helper utilities for the Flight Prediction Platform.

Experiment Layout (folder tree in MLflow UI).
-----------------------------------------------
Aviation MLOps
    Delay Risk/                 one parent run per pipeline_run_id
        delay_auto              HPO child (Optuna trials nested inside)
        training_delay          training child
    Congestion/                 One parent run per pipeline_run_id
        congestion_auto         HPO child
        training_congestion     training child
    Anomaly Detection/          One parent run per pipeline_run_id
        anomaly_auto            Anomaly child
        training_anomaly        training child
    Pipeline/                   cross-model orchestration
        evaluation              evaluation child
        registry                registry child
"""

from __future__ import annotations
import os
from typing import Any
import mlflow
from mlflow.tracking import MlflowClient
from collections.abc import Generator
from contextlib import contextmanager
from src.core.logger import get_logger
from src.utils.exceptions import MlflowError

logger = get_logger(__name__)


def setup_mlflow() -> MlflowClient:
    """Initialise MLflow tracking URI and artifact store credentials"""

    from src.config.config import get_settings

    settings = get_settings()
    uri = os.getenv("MLFLOW_TRACKING_URI", settings.mlflow.tracking_uri)
    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking uri: {uri}")
    logger.info(os.getenv("MLFLOW_S3_ENDPOINT_URL"))

    # MinIO artifact store credentials
    os.environ.setdefault(
        "MLFLOW_S3_ENDPOINT_URL",
        os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9002"
        ),  # "http://minio:9000"
    )
    os.environ.setdefault(
        "AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    )
    os.environ.setdefault(
        "AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    )

    logger.info(f"MLflow s3 endpoint url: {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")
    return MlflowClient()


def get_or_create_experiment(experiment_name: str) -> str:
    """Return experiment ID, creating the experiment when it does not exist."""

    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            exp_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created MLflow {experiment_name} experiment: {exp_id} ")
        else:
            exp_id = exp.experiment_id
        return exp_id
    except Exception as e:
        raise MlflowError(f"Failed to get/create experiment: {experiment_name}") from e


def create_parent_run(
    experiment_name: str, run_name: str, tags: dict[str, str] | None = None
) -> str:
    """Create a top-level (parent) MLflow run and return its run_id"""

    exp_id = get_or_create_experiment(experiment_name)
    try:
        client = MlflowClient()
        all_tags = {"mlflow.runName": run_name}
        if tags:
            all_tags.update(tags)
        run = client.create_run(experiment_id=exp_id, run_name=run_name, tags=all_tags)
        logger.info(
            f"Created parent run {experiment_name}/{run_name} with run_id: {run.info.run_id}"
        )
        return run.info.run_id
    except Exception as e:
        raise MlflowError(f"Failed to create parent run {run_name}") from e


def finish_run(run_id: str, status: str = "FINISHED") -> None:
    try:
        MlflowClient().set_terminated(run_id, status=status)
        logger.info("Finished MLflow run for {run_id} with status {status}")
    except Exception as e:
        logger.warning(f"Could not finish run for {run_id}")


@contextmanager
def child_run(
    parent_run_id: str,
    run_name: str,
    experiment_name: str,
    tags: dict[str, str] | None = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """
    Context manager that creates a nested (child) Mlflow run under a parent.

    The parent run does not need to be active run in this thread because we set the
    `mlflow.parentRunId` system tag directly.

    Usage:
        with child_run(parent_run_id,"delay_auto", "Aviation MLOps/Delay Risk") as run:
            mlflow.log_metric("cv_auc",0.82)
    """

    exp_id = get_or_create_experiment(experiment_name)
    client = MlflowClient()
    all_tags = {"mlflow.parentRunId": parent_run_id, "mlflow.runName": run_name}

    if tags:
        all_tags.update(tags)

    try:
        child = client.create_run(
            experiment_id=exp_id, run_name=run_name, tags=all_tags
        )
        child_run_id = child.info.run_id
        logger.info(
            f"Child run {run_name} started: {child_run_id} for parent {parent_run_id}"
        )
    except Exception as e:
        raise MlflowError(
            f"Failed to create child run {run_name} for parent {parent_run_id}"
        ) from e

    try:
        mlflow.set_experiment(
            experiment_name
        )  # set the active experiment for trials to inherit from
        with mlflow.start_run(run_id=child_run_id):
            yield mlflow.active_run()
        client.set_terminated(child_run_id, status="FINISHED")
    except Exception as e:
        client.set_terminated(child_run_id, status="FAILED")
        raise MlflowError(f"Failed to start {run_name} child run: {child_run_id}")


# model registry helpers
def get_production_model(model_name: str) -> Any | None:
    """Load the production-stage model from Mlflow registry"""
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions is None:
            logger.warning(f"No production model found for {model_name}")
            return None
        logger.info(
            f"Loading production model for {model_name} with version: {versions[0].version}"
        )
        model_uri = f"models:/{model_name}/Production"
        return mlflow.sklearn.load_model(model_uri=model_uri)
    except Exception as e:
        logger.error(f"Failed to load production model {model_name}")
        return None


def get_production_model_version(model_name: str) -> str | None:
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        return versions[0].version if versions else None
    except Exception as e:
        logger.error(f"Unable to find latest version for production model {model_name}")
        return None


def promote_to_production(
    model_name: str, run_id: str, artifact_path: str = "model"
) -> str:
    """
    Register a model from a training run and promote to production.
    """

    setup_mlflow()
    client = MlflowClient()
    try:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri, name=model_name)
        version = result.version

        # Archive existing production versions first
        for v in client.get_latest_versions(model_name, stages=["Production"]):
            client.transition_model_version_stage(
                name=model_name, version=v.version, stage="Archived"
            )
            logger.info(f"Archived model:{model_name} version: {version}")

        client.transition_model_version_stage(
            name=model_name, version=version, stage="Production"
        )
        logger.info(
            f"Promoted model:{model_name} with version: {version} to Production"
        )

        return version
    except Exception as e:
        raise MlflowError(
            f"Failed to promoted model: {model_name} to Production"
        ) from e
