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

BUCKET = os.getenv("MINIO__BUCKET")
MINIO_ENDPOINT = os.getenv("MINIO__ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

BRONZE_BASE = f"s3a://{BUCKET}/bronze/opensky"
SILVER_BASE = f"s3a://{BUCKET}/silver/flight_states"


def get_spark_session(
    app_name: str, extra_configs: dict[str, Any] | None = None
) -> SparkSession:

    builder = (
        SparkSession.appName(app_name)
        .master(os.getenv("SPARK_MASTER_URL", "local[*]"))
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
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
    spark.sparkContext.SetLogLevel("WARN")
    logger.info(f"Spark session {app_name} created master={spark.sparkContext.master}")
    return spark
