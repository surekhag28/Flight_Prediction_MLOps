"""
Silver -> Gold: Route-Level features spark job.

Routes are inferred from callsign prefix (airline ICAO code + flight number)
For each route, aggregates:
    - route_density: count of all flights on this route pattern
    - avg_cruise_speed: mean velocity when altitude > 5000 m
    - historical_avg_delay_proxy: mean proxy delay_risk per route
"""

from __future__ import annotations
import os
import logging
from datetime import UTC, datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.core.logger import get_logger

from src.utils.spark_utils import (
    GOLD_ROUTES_BASE,
    SILVER_BASE,
    StageMetadata,
    get_spark_session,
)

logger = get_logger(__name__)
DELAY_RISK_ALTITUDE_M = 3000.0
DELAY_RISK_VELOCITY_MS = 100.0


def run(spark: SparkSession, pipeline_run_id: str) -> dict:
    meta = StageMetadata("silver_to_gold_routes", pipeline_run_id)

    try:
        logger.info(f"Reading silver data from {SILVER_BASE}")

        # filtering non null and flights data
        df = spark.read.parquet(SILVER_BASE).filter(
            (F.col("callsign").isNotNull()) & (~F.col("on_ground"))
        )

        # Derive route_id from callsign prefix (first 3 chars = airline, next 1-4 = route variant)
        df = df.withColumn(
            "route_id",
            F.when(
                F.col("callsign") >= 4, F.substring(F.col("callsign"), 1, 6)
            ).otherwise(F.col("callsign")),
        )

        # proxy delay label
        df = df.withColumn(
            "delay_risk_proxy",
            F.when(
                (F.col("baro_altitude") > DELAY_RISK_ALTITUDE_M)
                & (F.col("velocity") < DELAY_RISK_VELOCITY_MS),
                1.0,
            ).otherwise(0.0),
        )

        # cruise speed: velocity when at cruise altitude
        cruise_df = df.filter(F.col("baro_altitude") > 5000.0)

        # route-level aggregations

        route_agg = df.groupBy("route_id").agg(
            F.count("*").alias("route_density"),
            F.avg("delay_risk_proxy").alias("historical_avg_delay_proxy"),
            F.countDistinct("aircraft_id").alias("unique_aircraft_count"),
        )

        cruise_agg = cruise_df.groupBy("route_id").agg(
            F.avg("velocity").alias("avg_cruise_speed")
        )

        gold_df = route_agg.join(cruise_agg, on="route_id", how="left")

        gold_df = gold_df.withColumn(
            "event_timestamp", F.current_timestamp()
        ).withColumn("created_timestamp", F.current_timestamp())

        gold_df = gold_df.select(
            "route_id",
            "event_timestamp",
            "created_timestamp",
            "route_density",
            "avg_cruise_speed",
            "historical_avg_delay_proxy",
            "unique_aircraft_count",
        )

        (gold_df.coalesce(4).write.mode("overwrite").parquet(GOLD_ROUTES_BASE))

        count = spark.read.parquet(GOLD_ROUTES_BASE).count()
        meta.add_artifact("gold_rows_written", count)
        meta.add_artifact("gold_routes_path", GOLD_ROUTES_BASE)
        meta.status = "success"
        logger.info(f"Gold routes: wrote {count} rows to {GOLD_ROUTES_BASE}")
    except Exception as e:
        meta.status = "failed"
        meta.error = str(e)
        logger.error(f"silver_to_gold_routes failed: {str(e)}", exc_info=True)
        raise
    finally:
        meta.save()

    return meta.to_dict()


def main():
    run_id = os.getenv(
        "PIPELINE_RUN_ID", datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    )
    spark = get_spark_session("aviation_silver_to_gold_routes")
    try:
        result = run(spark, run_id)
        print(result)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
