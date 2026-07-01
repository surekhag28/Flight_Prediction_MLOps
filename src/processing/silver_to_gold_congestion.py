"""
Silver -> Gold: Airport Congestion Features Spark Job.

For each snapshot timestamp and each of major 50 airports, computes below features:
    1. aircraft_count_50km: aircraft within 50km radius.
    2. arrivals_last_30m: aircraft descending (vertical_rate < -1 m/s)
    3. departures_last_30m: aircraft climbing (vertical_rate > 1 m/s)
    4. congestion_score: aircraft_count_50km / max_observed (0-1)
    5. avg_altitude_50km: mean altitude of aircraft nearby
"""

from __future__ import annotations
import os
import math
from datetime import UTC, datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from src.config.config import get_settings

from src.utils.spark_utils import (
    get_spark_session,
    GOLD_CONGESTION_BASE,
    SILVER_BASE,
    StageMetadata,
)

from src.core.logger import get_logger

logger = get_logger(__name__)

AIRPORTS = get_settings().airports  # list of AirportConfig objects


def _haversine_udf(lat1: float, lon1: float, lat2: float, lon2: float) -> float | None:
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
    meta = StageMetadata("silver_to_gold_congestion", pipeline_run_id)

    try:
        logger.info(f"Reading silver from {SILVER_BASE}")
        silver_df = spark.read.parquet(SILVER_BASE).filter(
            (F.col("latitude").isNotNull()) & (F.col("longitude").isNotNull())
        )

        airport_data = [(a.code, a.name, a.lat, a.lon) for a in AIRPORTS]
        airports_schema = StructType(
            [
                StructField("airport_code", StringType(), False),
                StructField("airport_name", StringType(), False),
                StructField("airport_lat", DoubleType(), False),
                StructField("airport_lon", DoubleType(), False),
            ]
        )

        airport_df = spark.createDataFrame(airport_data, schema=airports_schema)

        # creating timed-bucket (per-hour snapshot) to aggregate
        bucketed = silver_df.withColumn(
            "snapshot_ts", F.date_trunc("hour", F.col("event_timestamp"))
        )

        # cross join aircraft with airport data (safe to join because of only 50 airport data points)
        crossed = bucketed.crossJoin(F.broadcast(airport_df))

        with_distance = crossed.withColumn(
            "dist_km",
            haversine_udf(
                F.col("latitude"),
                F.col("longitude"),
                F.col("airport_lat"),
                F.col("airport_lon"),
            ),
        )

        # filter to 50km radius
        nearby = with_distance.filter(F.col("dist_km") <= 50.0)

        agg = nearby.groupBy("airport_code", "airport_name", "snapshot_ts").agg(
            F.count("*").alias("aircraft_count_50km"),
            F.sum(F.when(F.col("vertical_rate") < -1.0, 1).otherwise(0.0)).alias(
                "arrivals_last_30m"
            ),
            F.sum(F.when(F.col("vertical_rate") > 1.0, 1).otherwise(0.0)).alias(
                "departure_last_30m"
            ),
            F.avg("baro_altitude").alias("avg_altitude_50km"),
        )

        # compute max observed count for normalisation
        max_count_df = agg.agg(F.max("aircraft_count_50km").alias("max_count"))
        max_count = max_count_df.collect()[0]["max_count"]

        gold_df = (
            agg.withColumn(
                "congestion_score",
                (
                    F.col("aircraft_count_50km")
                    / F.lit(float(max_count)).cast(DoubleType())
                ).cast(DoubleType()),
            )
            .withColumn("event_timestamp", F.col("snapshot_ts"))
            .withColumn("created_timestamp", F.current_timestamp())
        ).select(
            "airport_code",
            "airport_name",
            "event_timestamp",
            "created_timestamp",
            "aircraft_count_50km",
            "arrivals_last_30m",
            "departure_last_30m",
            "avg_altitude_50km",
            "congestion_score",
        )

        meta.add_metric("max_aircraft_count_50km", max_count)

        (gold_df.coalesce(4).write.mode("overwrite").parquet(GOLD_CONGESTION_BASE))

        count = spark.read.parquet(GOLD_CONGESTION_BASE).count()
        meta.add_metric("gold_rows_written", count)
        meta.add_artifact("gold_congestion_path", GOLD_CONGESTION_BASE)
        meta.status = "success"
        logger.info(f"Gold congestion wrote {count} rows to {GOLD_CONGESTION_BASE}")

    except Exception as e:
        meta.status = "failed"
        meta.error = str(e)
        logger.error(f"silver_to_gold_congestion failed: {e}", exc_info=True)
        raise
    finally:
        meta.save()

    return meta.to_dict()


def main():
    run_id = os.getenv(
        "PIPELINE_RUN_ID", datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    )
    spark = get_spark_session("aviation_silver_to_gold_congestion")
    try:
        result = run(spark, run_id)
        print(result)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
