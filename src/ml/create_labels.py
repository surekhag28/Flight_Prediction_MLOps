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
import pyarrow as pa
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


def read_gold_parquet(
    path: str, fs: s3fs.S3FileSystem, mode: str = "full"
) -> pd.DataFrame:
    from src.ml.training_utils import load_parquet_data

    try:
        return load_parquet_data(
            path.replace("s3://", ""), fs, path.split("/")[-1], mode
        )
    except InsufficientDataError as e:
        print(e)
        raise
        # raise FileNotFoundError(f"No parquet files found at {path}") from e


def build_congestion_lookup(congestion_df: pd.DataFrame) -> dict:
    """Creates hourly maximum congestion score

    Example:
    {
        Timestamp("2026-07-29 10:00"): 0.72,
        Timestamp("2026-07-29 11:00"): 0.65
    }

    """

    congestion_df["snapshot_hour"] = pd.to_datetime(
        congestion_df["event_timestamp"]
    ).dt.floor("h")

    max_congestion = congestion_df.groupby("snapshot_hour")["congestion_score"].max()

    logger.info(f"Created congestion lookup with {len(max_congestion)} hours")

    return max_congestion.to_dict()


# Generate labels per batch of data


def generate_delay_labels(
    flights_df: pd.DataFrame, congestion_lookup: dict
) -> pd.DataFrame:
    """Generate delay_risk labels for one flight dataframe
    This processes one parquet file at a time.
    """

    flights_df["snapshot_hour"] = pd.to_datetime(
        flights_df["event_timestamp"]
    ).dt.floor("h")

    flights_df["max_congestion_score"] = (
        flights_df["snapshot_hour"].map(congestion_lookup).fillna(0.0)
    )

    altitude_slow = (flights_df["altitude_m"] > THRESHOLDS.delay_risk_altitude_m) & (
        flights_df["speed_ms"] < THRESHOLDS.delay_risk_velocity_m
    )

    high_congestion = flights_df["max_congestion_score"] > THRESHOLDS.congestion_high

    flights_df["delay_risk"] = (altitude_slow | high_congestion).astype("uint8")

    delay_rate = flights_df["delay_risk"].mean()

    logger.info(
        f"Generated labels: rows={len(flights_df)} and delay_rate={delay_rate*100}"
    )

    return flights_df


def write_labels(flight_iterator, congestion_lookup, fs: s3fs.S3FileSystem) -> dict:
    """Write one labels.parquet file."""

    import gc

    file_path = f"{GOLD_LABELS_PATH.replace('s3://','')}/labels.parquet"

    total_rows = 0
    delay_count = 0
    writer = None

    f = fs.open(file_path, "wb")
    writer = None
    try:

        for index, flights_df in enumerate(flight_iterator):
            logger.info(f"Processing flight parquet file: {index}")

            labelled_df = generate_delay_labels(flights_df, congestion_lookup)
            table = pa.Table.from_pandas(labelled_df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(f, table.schema)

            writer.write_table(table)

            total_rows += len(labelled_df)
            delay_count += int(labelled_df["delay_risk"].sum())

            del flights_df
            del labelled_df
            del table
            gc.collect()

    finally:
        if writer is not None:
            writer.close()
        f.close()

    logger.info(f"labels written successfully: rows={total_rows}")

    return {
        "total_rows": total_rows,
        "delay_risk_count": delay_count,
        "delay_risk_rate": (delay_count / total_rows if total_rows > 0 else 0),
        "output_path": (f"s3://{file_path}"),
    }


def run(pipeline_run_id: str | None = None) -> dict:
    from src.ml.training_utils import get_fs

    if pipeline_run_id is None:
        pipeline_run_id = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    logger.info(f"Generating proxy labels for (pipeline_run_id= {pipeline_run_id})")
    fs = get_fs()

    congestion_df = read_gold_parquet(GOLD_CONGESTION_PATH, fs)
    logger.info(f"Airport congestion features: {len(congestion_df)} rows")
    congestion_lookup = build_congestion_lookup(congestion_df)

    del congestion_df

    flights_iterator = read_gold_parquet(GOLD_FLIGHTS_PATH, fs, "iterator")
    summary = write_labels(flights_iterator, congestion_lookup, fs)
    summary.update({"pipeline_run_id": pipeline_run_id, "status": "success"})

    logger.info(f"Label generation completed: {summary}")

    return summary


if __name__ == "__main__":
    result = run()
    import json

    print(json.dumps(result, indent=2))
