from __future__ import annotations

import s3fs
import pandas as pd
import mlflow
import pyarrow.parquet as pq
from typing import Any
from contextlib import contextmanager
from src.config.config import get_settings
from src.core.logger import get_logger

from src.core.logger import get_logger
from src.utils.exceptions import InsufficientDataError
from src.ml.algorithms import AlgorithmSpec
from src.utils.mlflow_utils import get_or_create_experiment

settings = get_settings()
logger = get_logger(__name__)


def get_fs() -> s3fs.S3FileSystem:
    """Create an S3FileSystem pointing at the configured MinIO instance"""
    return s3fs.S3FileSystem(
        endpoint_url=settings.miniosettings.endpoint,
        key=settings.miniosettings.access_key,
        secret=settings.miniosettings.secret_key,
    )


def load_parquet_data(path: str, fs: s3fs.S3FileSystem, name: str) -> pd.DataFrame:
    """
    Loads all parquet files under path from MinIO server.
    Raises InsufficientDataError if no files are present.
    """

    files = fs.glob(f"{path}/**/*.parquet")
    if not files:
        raise InsufficientDataError(f"No {name} files found", context={"path": path})
    dfs = [pq.read_table(fs.open(f)).to_pandas() for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {name} training data: total rows = {len(df)}")
    return df


def resolve_algorithm(
    algorithm: str, registry: dict[str, AlgorithmSpec], default: str
) -> tuple[str, AlgorithmSpec]:
    """Lookup algorithm in registry. Falling back to default if not found."""
    if algorithm in registry:
        return algorithm, registry[algorithm]
    return default, registry[default]


@contextmanager
def standalone_run(
    experiment_name: str,
    run_name: str,
    pipeline_run_id: str,
    algorithm: str,
    extra_tags: dict[str, Any] | None = None,
):
    """Context manager for standalone training (no parent).
    Used when running the training scripts outside the airflow DAG.
    """

    exp_id = get_or_create_experiment(experiment_name)
    tags = {"pipeline_run_id": pipeline_run_id, "algorithm": algorithm}
    if extra_tags:
        tags.update(extra_tags)
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags=tags) as run:
        yield run


def fit_model(
    model: Any, algorithm: str, X_train, y_train, X_test=None, y_test=None
) -> None:
    """Fit a model with best algorithm found durin HPO process"""

    if X_test is not None:
        model.fit(X_train, y_train, eval_set=[X_test, y_test], callbacks=[])
    else:
        model.fit(X_train, y_train)
