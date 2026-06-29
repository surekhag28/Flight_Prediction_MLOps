"""
Silver -> Gold: Flight State features spark job.

Computes per-aircraft window features using spark window functions:
    - heading_change_5m: heading change since previous snapshot.
    - avg_speed_15m: rolling average speed (last 15 min time window)
    - distance_10m_10km: distance travelled since previous snapshot (haverstine)
    - hour_of_day, day_of_week: temporal features

These features are used by:
    - Model A: Flight Delay Classifier
    - Model B: Anomaly Detection

Output Parquet requires event_timestamp column for Feast offline store.
"""

from __future__ import annotations
import os
import math
import logging
from datetime import UTC, datetime

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from src.core.logger import get_logger

from src.utils.spark_utils import (
    GOLD_FLIGHTS_BASE,
    SILVER_BASE,
    StageMetadata,
    get_spark_session,
)

logger = get_logger(__name__)


def _haversine_udf(lat1, lon1, lat2, lon2):
    """Haversine distance in km as a spark UDF"""

    if any(v is None for v in (lat1, lon1, lat2, lon2)):
        return None
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


haversine_udf = F.udf(_haversine_udf, DoubleType())


def run(spark: SparkSession, pipeline_run_id: str) -> dict:
    meta = StageMetadata("silver_to_gold_flights", pipeline_run_id)

    try:
        logger.info(f"Reading silver data from {SILVER_BASE}")
        silver_df = spark.read.parquet(SILVER_BASE)

        # considering flights which are flying
        df = silver_df.filter(~F.col("on_ground")).filter(
            F.col("velocity").isNotNull()
            & F.col("heading").isNotNull()
            & F.col("baro_altitude").isNotNull()
        )

        w = Window.partitionBy(F.col("aircraft_id")).orderBy(F.col("event_timestamp"))

        # rolling window for 15mins
        w_15m = (
            Window.partitionBy(F.col("aircraft_id"))
            .orderBy(F.col("event_timestamp").cast("long"))
            .rangeBetween(-900, 0)  # 15 x 60 seconds
        )

        df = df.withColumn("prev_heading", F.lag("heading", 1).over(w))
        df = df.withColumn("prev_lat", F.lag("latitude", 1).over(w))
        df = df.withColumn("prev_lon", F.lag("longitude", 1).over(w))

        # features computation
        df = df.withColumn(
            "heading_change_5m",
            F.when(
                F.col("heading").isNotNull(),
                F.abs(F.col("prev_heading") - F.col("heading")),
            ).otherwise(0.0),
        )

        df = df.withColumn("avg_speed_15m", F.avg("velocity").over(w_15m))

        df = df.withColumn(
            "distance_10m_km",
            F.when(
                F.col("prev_lat").isNotNull() & F.col("prev_lon").isNotNull(),
                haversine_udf(
                    F.col("latitude"),
                    F.col("longitude"),
                    F.col("prev_lat"),
                    F.col("prev_lon"),
                ),
            ).otherwise(0.0),
        )

        df = df.withColumn("hour_of_day", F.hour("event_timestamp"))
        df = df.withColumn("day_of_week", F.dayofweek("event_timestamp"))

        gold_df = df.select(
            "aircraft_id",
            "callsign",
            "origin_country",
            "event_timestamp",
            "latitude",
            "longitude",
            F.col("velocity").alias("speed_ms"),
            F.col("baro_altitude").alias("altitude_ms"),
            F.col("vertical_rate").alias("vertical_rate_ms"),
            F.col("heading").alias("heading_deg"),
            "heading_change_5m",
            "avg_speed_15m",
            "distance_10m_km",
            "hour_of_day",
            "day_of_week",
            F.col("on_ground").cast("int").alias("is_on_ground"),
        ).dropna(subset=["aircraft_id", "event_timestamp"])

        gold_df = gold_df.withColumn("created_timestamp", F.current_timestamp())

        (gold_df.coalesce(4).write.mode("overwrite").parquet(GOLD_FLIGHTS_BASE))

        count = spark.read.parquet(GOLD_FLIGHTS_BASE).count()
        meta.add_metric("gold_rows_written", count)
        meta.add_artifact("gold_flights_path", GOLD_FLIGHTS_BASE)
        meta.status = "success"
        logger.info(f"Gold flights: wrote {count} rows to {GOLD_FLIGHTS_BASE}")

    except Exception as e:
        meta.status = "failed"
        meta.error = str(e)
        logger.error(f"silver_to_gold_flights failed: {e}", exc_info=True)
        raise
    finally:
        meta.save()

    return meta.to_dict()


def main() -> None:
    run_id = os.getenv(
        "PIPELINE_RUN_ID", datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    )
    spark = get_spark_session("aviation_silver_to_gold_flights")
    try:
        result = run(spark, run_id)
        logger.info(result)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
