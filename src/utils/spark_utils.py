"""
Spark session factory with MinIO/S3A support.

All spark jobs in this project uses get_spark_session() as the entrypoint for processing
"""

from __future__ import annotations
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from src.core.logger import get_logger

from pyspark.sql import SparkSession

logger = get_logger(__name__)

BUCKET = os.getenv("MINIO_BUCKET", "aviation-lake")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

logger.info(f"Bucket :  {BUCKET}")
logger.info(f"Minio Endpoint :  {MINIO_ENDPOINT}")

BRONZE_BASE = f"s3a://{BUCKET}/bronze/opensky"
SILVER_BASE = f"s3a://{BUCKET}/silver/flight_states"
GOLD_FLIGHTS_BASE = f"s3a://{BUCKET}/gold/flight_state_features"
GOLD_CONGESTION_BASE = f"s3a://{BUCKET}/gold/airport_congestion_features"
GOLD_ROUTES_BASE = f"s3a://{BUCKET}/gold/route_features"
GOLD_LABELS_BASE = f"s3a://{BUCKET}/gold/labels"


def get_spark_session(
    app_name: str, extra_configs: dict[str, Any] | None = None
) -> SparkSession:

    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.getenv("SPARK_MASTER_URL", "local[*]"))
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        # Performance
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "512m")
        .config("spark.executor.memory", "512m")
        .config("spark.driver.maxResultSize", "256m")
        # Parquet
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
    )

    # Hadoop-AWS JARs path (pre-installed in Dockerfile)
    spark_home = os.getenv("SPARK_HOME", "")
    if spark_home:
        jars_dir = Path(spark_home) / "jars"
        hadoop_aws = jars_dir / "hadoop-aws-3.3.4.jar"
        aws_sdk = jars_dir / "aws-java-sdk-bundle-1.12.262.jar"
        if hadoop_aws.exists() and aws_sdk.exists():
            builder = builder.config("spark.jars", f"{hadoop_aws},{aws_sdk}")
        else:
            # Fallback: download via Maven (requires internet)
            builder = builder.config(
                "spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
            )

    if extra_configs:
        for key, value in extra_configs.items():
            builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"Spark session {app_name} created master={spark.sparkContext.master}")
    return spark


class StageMetadata:
    def __init__(self, stage_name: str, pipeline_run_id: str):
        self.stage_name = stage_name
        self.pipeline_run_id = pipeline_run_id
        self.start_time = time.time()
        self.metrics: dict[str, Any] = {}
        self.artifacts: dict[str, Any] = {}
        self.status = "running"
        self.error: str | None = None

    def add_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def add_artifact(self, key: str, path: Any) -> None:
        self.artifacts[key] = path

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "pipeline_run_id": self.pipeline_run_id,
            "status": self.status,
            "start_time": self.start_time,
            "duration_secs": round(time.time()) - round(self.start_time),
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "error": self.error,
        }

    def save(self, output_dir="/tmp/metadata"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / f"{self.stage_name}_{self.pipeline_run_id}.json"

        with file_path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Stage metadata saved to {file_path}")
        return str(file_path)
