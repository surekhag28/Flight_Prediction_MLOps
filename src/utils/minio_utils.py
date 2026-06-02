"""
MinIO/S3 compatible data storage helpers using boto3
Used for raw JSON writes (ingestion) to bronze layer and metadata upload.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from src.core.logger import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)


MINIO_ENDPOINT = os.getenv("MINIO__ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
DEFAULT_BUCKET = os.getenv("MINIO_BUCKET", "aviation-lake")


def get_s3_client() -> boto3.client:
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(bucket: str = DEFAULT_BUCKET) -> None:
    client = get_s3_client()

    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
            client.create_bucket(bucket)
            logger.info(f"Created {bucket} bucket")
        raise


def put_json(key: str, data: Any, bucket: str = DEFAULT_BUCKET) -> None:
    client = get_s3_client()
    body = json.dumps(data, default=str).encode("utf-8")
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.debug(f"Uploaded JSON to s3://{bucket}/{key} ({len(body)} bytes)")
