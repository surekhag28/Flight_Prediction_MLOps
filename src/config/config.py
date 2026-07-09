import os
import yaml
from pathlib import Path
from typing import Literal, Any
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
    endpoint: str = "http://localhost:9002"  # "http://minio:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "aviation-lake"


class AirportConfig(BaseModel):
    code: str
    name: str
    lat: float
    lon: float


class ThresholdsConfig(BaseModel):
    delay_risk_altitude_m: float = 3000.0
    delay_risk_velocity_m: float = 100.0
    congestion_high: float = 0.70


class HPOConfig(BaseModel):
    n_trials: int = 30
    sample_rows: int = 50_000
    n_cv_folds: int = 3
    min_sample_rows_delay: int = 500
    min_sample_rows_congestion: int = 500
    min_sample_rows_anomaly: int = 500


class DelayTrainingConfig(BaseModel):
    target_column: str = "delay_risk"
    feature_columns: list[str] = Field(
        default_factory=lambda: [
            "speed_ms",
            "altitude_m",
            "vertical_rate_ms",
            "heading_change_5m",
            "avg_speed_15m",
            "aircraft_count_50km",
            "congestion_score",
            "hour_of_day",
            "day_of_week",
        ]
    )
    max_rows: int = 500_000
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    min_auc: float = 0.62
    min_f1: float = 0.50
    lgbm_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
        }
    )


class TrainingConfig(BaseModel):
    delay: DelayTrainingConfig = Field(default_factory=DelayTrainingConfig)


class MLflowExperimentsConfig(BaseModel):
    delay: str = "Aviation MLOps/Delay Risk"
    congestion: str = "Aviation MLOPs/Congestion"
    anomaly: str = "Aviation MLOps/Anomaly Detection"
    pipeline: str = "Aviation MLOps/Pipeline"


class MLflowModelNamesConfig(BaseModel):
    delay: str = "aviation-delay-risk-model"
    congestion: str = "aviation-congestion-model"
    anomaly: str = "aviation-anomaly-model"


class MLflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5001"  # "http://mlflow:5000"
    dagshub_username: str = ""
    dagshub_password: str = ""
    experiments: MLflowExperimentsConfig = Field(
        default_factory=MLflowExperimentsConfig
    )
    model_names: MLflowModelNamesConfig = Field(default_factory=MLflowModelNamesConfig)


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
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    hpo: HPOConfig = Field(default_factory=HPOConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def get_settings() -> Settings:
    return Settings()
