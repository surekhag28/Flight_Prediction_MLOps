"""
DAG 1: Data Ingestion Pipeline
Runs every 15 minutes
    fetch_opensky_api > validate > write to bronze > bronze to silver > gold flights
"""

from __future__ import annotations
import sys
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow import DAG
from src.data_quality.validate import publish_data_docs
from src.core.logger import get_logger

DEFAULT_ARGS = {
    "ownder": "aviation_ml",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(hours=1),
}


def _ingest(**context):
    from dag_utils import set_pipeline_run_id
    from src.ingestion.opensky_client import ingest

    run_id = set_pipeline_run_id(**context)
    log = get_logger(__name__)
    result = ingest()
    context["task_instance"].xcom_push(key="bronze_key", value=result["bronze_key"])
    context["task_instance"].xcom_push(
        key="aircraft_count", value=result["aircraft_count"]
    )
    log.info(
        f"Ingestion complete, aircraft_count= {result['aircraft_count']}, bronze_key = {result['bronze_key']}"
    )
    return result


def _validate(**context):
    log = get_logger(__name__)
    count = (
        context["task_instance"].xcom_pull(
            task_ids="fetch_opensky_api", key="aircraft_count"
        )
        or 0
    )

    if count < 100:
        log.warning(
            f"Aircraft count below threhsold, count={count}, threhsold=100",
        )
        raise ValueError(f"Only {count} aircraft states received - possible API issue")
    log.info("Schema validation passed")
    return count


def _validate_bronze(**context):
    """
    Great expectation validation of the bronze dataset.
    Reads the most recent bronze parquet file from MinIO and runs the bronze suite.
    Raises DataQualityError on critical failures and blocks the downstream tasks.
    """

    import pandas as pd
    import s3fs
    from dag_utils import get_pipeline_run_id
    from src.config.config import get_settings
    from src.data_quality.validate import validate_bronze
    from src.utils.exceptions import DataQualityError, MinioError
    import json

    run_id = get_pipeline_run_id(**context)
    log = get_logger(__name__)
    bronze_key = context["task_instance"].xcom_pull(
        task_ids="fetch_opensky_api", key="bronze_key"
    )

    settings = get_settings()

    fs = s3fs.S3FileSystem(
        key=settings.miniosettings.access_key,
        secret=settings.miniosettings.secret_key,
        endpoint_url=settings.miniosettings.endpoint,
    )

    path = f"{settings.miniosettings.bucket}/{bronze_key}"
    try:
        with fs.open(path) as f:
            payload = json.load(f)
        df = pd.DataFrame(payload["states"])
        if "timestamp" not in df.columns and "last_contact" in df.columns:
            df["timestamp"] = df["last_contact"]
    except Exception as e:
        raise MinioError(
            f"Cannot read bronze file for validation: {bronze_key}",
            context={"error": str(e)},
        ) from e

    result = validate_bronze(df)
    publish_data_docs([result], run_id)

    log.info(
        f"Bronze validation complete: passed: {result.passed}, evaluated: {result.evaluated_expectations}, "
        "failed: {result.failed_expectations}, success_rate: {result.success_rate}"
    )

    if not result.passed:
        failed_types = [f["expectation_type"] for f in result.failed_expectations]
        raise DataQualityError(
            f"Data quality check failed at bronze layer for {len(result.failed_expectations)} expectations",
            context={"failed_expectations": failed_types, "bronze_key": bronze_key},
        )

    return result.to_dict()


# def test():
#     print("in airflow")
#     _ingest()


# test()

with DAG(
    dag_id="opensky_ingest_dag",
    description="Fetching data from OpenSky API and ingesting to > bronze > silver > gold layer",
    schedule_interval="0 * * * *",
    start_date=datetime(2026, 6, 6),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ingestion", "opensky", "etl"],
) as dag:
    fetch = PythonOperator(task_id="fetch_opensky_api", python_callable=_ingest)
    validate = PythonOperator(task_id="validate_schema", python_callable=_validate)
    validate_bronze = PythonOperator(
        task_id="validate_bronze", python_callable=_validate_bronze
    )

    fetch >> validate >> validate_bronze
