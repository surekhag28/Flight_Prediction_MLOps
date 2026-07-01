import os
import yaml
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
)

ENV_FILE_PATH = Path(__file__).parent.parent / ".env"
YAML_FILE_PATH = Path(__file__).parent / "settings.yaml"


class MinioSettings(BaseModel):
    endpoint: str = "http://minio:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "aviation-lake"


class AirportConfig(BaseModel):
    code: str
    name: str
    lat: float
    lon: float


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        class YamlSettingsSource(PydanticBaseSettingsSource):
            def __call__(self):
                if not YAML_FILE_PATH.exists():
                    return {}
                with YAML_FILE_PATH.open() as f:
                    return yaml.safe_load(f) or {}

            def get_field_value(self, field, field_name):
                return None, field_name, False

        yaml_settings = YamlSettingsSource(settings_cls)

        return (
            init_settings,
            yaml_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    api_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "flight-prediction-ml"

    miniosettings: MinioSettings = Field(default_factory=MinioSettings)
    airports: list[AirportConfig] = Field(default_factory=list)


def get_settings() -> Settings:
    return Settings()
