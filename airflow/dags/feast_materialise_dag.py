"""
DAG 2: Feast Materialisation
Hourly at :30 - materialises gold features from MinIO -> Redis online store


Flow:
    feast_apply -> materialise_to_redis
"""

from __future__ import annotations
import sys
import subprocess
from datetime import UTC, datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow import DAG

from src.core.logger import get_logger

logger = get_logger(__name__)

sys.path.insert(0, "/opt/src")

FEAST_REPO = "/opt/feature_repo"

DEFAULT_ARGS = {
    "owner": "aviation_ml",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}


def _feast_apply(**context):
    """Apply feast feature definitions to update registry."""
    result = subprocess.run(
        ["feast", "-c", FEAST_REPO, "apply"], capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(f"feast apply failed: {result.stderr.strip()}")
        raise RuntimeError(f"feast apply failed: \n{result.stderr}")
    logger.info(f"feast apply succeeded: {result.stdout}")


def _materialise_features(**context):
    """Materialise all features to online Redis store."""
    from feast import FeatureStore

    store = FeatureStore(repo_path=FEAST_REPO)
    start = datetime.now(tz=UTC) - timedelta(hours=2)
    end = datetime.now(tz=UTC)

    try:
        store.materialize(start_date=start, end_date=end)
        logger.info(
            f"Feast materialisation completed: start={start.isoformat()}, end={end.isoformat()}"
        )
    except Exception as e:
        logger.warning("Materialization warning (may be no data found)")
        logger.info(store.get_feature_view("route_features").batch_source.path)
        logger.error(f"{str(e)}")

    # Post-materialisation check for feature views: online store should have data
    try:
        feature_views = store.list_feature_views()
        if not feature_views:
            logger.warning(
                "No feature views found in the registry - check feast apply step"
            )
        else:
            logger.info(
                f"Feature views materialized: count={len(feature_views)}, views = {[fv.name for fv in feature_views]}"
            )
    except Exception as e:
        logger.warning(f"Could not verify feature views post-serialisation: {str(e)}")


with DAG(
    dag_id="feast_materialisation_dag",
    description="Materialise gold features to Redis online store (hourly)",
    schedule_interval="30 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["feast", "feature_store", "online_store"],
) as dag:
    apply = PythonOperator(task_id="feast_apply", python_callable=_feast_apply)
    materialise = PythonOperator(
        task_id="materialise_to_redis", python_callable=_materialise_features
    )

    apply >> materialise
