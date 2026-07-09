"""
Proxy label generation for all three ML models.

These are engineered proxy labels. OpenSky free API has no ground-truth
delay or anomaly data. The proxies are physically motivated signals.

Model A - delay_risk (binary classifier)
    1 if aircraft is at altitude > 3000m AND velocity < 100 m/s
    OR airport_congestion_score > 0.70
    Rationale: slow aircraft at altitude = likely holding patterns = likely getting delayed

Read gold parquet files, joins flight + congestion features, adds delay_risk label,
writes combined labelled data to gold/labels location.
"""

from __future__ import annotations
import typing
from datetime import UTC, datetime

import pandas as pd
import s3fs
import pyarrow.parquet as pq
from src.config.config import get_settings
from src.core.logger import get_logger
from src.utils.exceptions import InsufficientDataError

logger = get_logger(__name__)
settings = get_settings()

BUCKET = settings.miniosettings.bucket
THRESHOLDS = settings.thresholds
GOLD_FLIGHTS_PATH = f"s3://{BUCKET}/gold/flight_state_features"
GOLD_CONGESTION_PATH = f"s3://{BUCKET}/gold/airport_congestion_features"
GOLD_LABELS_PATH = f"s3://{BUCKET}/gold/labels"


def read_gold_parquet(path: str, fs: s3fs.S3FileSystem) -> pd.DataFrame:
    from src.ml.training_utils import load_parquet_data

    try:
        return load_parquet_data(path.replace("s3://", ""), fs, path.split("/")[-1])
    except InsufficientDataError as e:
        raise FileNotFoundError(f"No parquet files found at {path}") from e


def generate_delay_labels(
    flights_df: pd.DataFrame, congestion_df: pd.DataFrame
) -> pd.DataFrame:
    """Join flight features with nearby airport congestion, compute delay_risk proxy"""

    congestion_df = congestion_df.copy()
    congestion_df["snapshot_hour"] = pd.to_datetime(
        congestion_df["event_timestamp"]
    ).dt.floor("h")
    max_congestion = (
        congestion_df.groupby("snapshot_hour")["congestion_score"]
        .max()
        .reset_index()
        .rename(columns={"congestion_score": "max_congestion_score"})
    )

    flights_df = flights_df.copy()
    flights_df["snapshot_hour"] = pd.to_datetime(
        flights_df["event_timestamp"]
    ).dt.floor("h")

    merged = flights_df.merge(max_congestion, on="snapshot_hour", how="left")
    merged["max_congestion_score"] = merged["max_congestion_score"].fillna(0.0)

    altitude_slow = (merged["altitude_m"] > THRESHOLDS.delay_risk_altitude_m) & (
        merged["speed_ms"] < THRESHOLDS.delay_risk_velocity_m
    )

    high_congestion = merged["max_congestion_score"] > THRESHOLDS.congestion_high

    merged["delay_risk"] = (altitude_slow | high_congestion).astype(
        int
    )  # proxy delay labels

    delay_rate = merged["delay_risk"].mean()
    logger.info(
        "Delay stats: %d total, %.1f%% delay risk", len(merged), delay_rate * 100
    )

    return merged


def write_labels(df: pd.DataFrame, fs: s3fs.S3FileSystem) -> str:
    """Write labelled dataset to gold/labels/ as parquet file"""
    import pyarrow as pa

    table = pa.Table.from_pandas(df)
    file_path = f"{GOLD_LABELS_PATH.replace('s3://','')}/labels.parquet"
    with fs.open(file_path, "wb") as f:
        pq.write_table(table, f)
    logger.info(f"Labels written to s3://{file_path}")

    return f"s3://{file_path}"


def run(pipeline_run_id: str | None = None) -> dict:
    from src.ml.training_utils import get_fs

    if pipeline_run_id is None:
        pipeline_run_id = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    logger.info(f"Generating proxy labels for (pipeline_run_id= {pipeline_run_id})")
    fs = get_fs()
    flights_df = read_gold_parquet(GOLD_FLIGHTS_PATH, fs)
    logger.info(f"Flight features: {len(flights_df)} rows")

    congestion_df = read_gold_parquet(GOLD_CONGESTION_PATH, fs)
    logger.info(f"Airport congestion features: {len(congestion_df)} rows")

    labelled_df = generate_delay_labels(flights_df, congestion_df)
    output_path = write_labels(labelled_df, fs)
    summary = {
        "pipeline_run_id": pipeline_run_id,
        "total rows": len(labelled_df),
        "delay_risk_count": int(labelled_df["delay_risk"].sum()),
        "delay_risk_rate": float(labelled_df["delay_risk"].mean()),
        "output_path": output_path,
        "status": "success",
    }

    print(labelled_df.head(5))

    return summary


if __name__ == "__main__":
    result = run()
    import json

    print(json.dumps(result, indent=2))
