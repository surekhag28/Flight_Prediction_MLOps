import os
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class BaseConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class MinioSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="MINIO__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    endpoint: str = "http://minio:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "aviation-lake"


class Settings(BaseConfigSettings):
    api_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "flight-prediction-ml"

    miniosettings: MinioSettings = Field(default_factory=MinioSettings)


def get_settings() -> Settings:
    return Settings()
