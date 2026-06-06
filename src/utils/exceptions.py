"""
Custom exception hierarchy

All the exceptions inherit from AviationMLError so that each caller can catch their respective error at the granularity level
"""

from __future__ import annotations


class AviationMLError(Exception):
    """Base exception for every runtime error in the application."""

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.context: dict = context or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base} [{ctx}]"
        return base


class MinioError(AviationMLError):
    """Error reading from or writing to Minio / S3."""

    pass


class DataQualityError(AviationMLError):
    """ETL quality gate failed (row count, schema complaince, null ratio)."""

    pass
