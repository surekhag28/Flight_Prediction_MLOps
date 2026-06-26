"""
Bronze --> Silver Spark ETL Job.

Reads raw OpenSky Json filesfrom MinIO bronze layer, applies:
    1. Schema enforcement(all 17 OpenSky fields + metadata)
    2. Null filtering (drop rows with null lat/lon/altitude)
    3. Deduplication on (icao24, time_position)
    4. Timestamp normalisation(unix seconds --> UTC timestamp)
    5. Outlier filtering(altitude, velocity bounds)

Writes cleaned parquet to silver layer partitioned by year/month/day
"""

from __future__ import annotations
import os
import logging
from datetime import datetime, UTC


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.utils.exceptions import DataQualityError
from src.utils.spark_utils import get_spark_session
from src.core.logger import get_logger
from src.utils.spark_utils import (
    BRONZE_BASE,
    SILVER_BASE,
    StageMetadata,
    get_spark_session,
)

logger = get_logger(__name__)

# data quality thresholds
_MIN_SILVER_ROWS = 50
_MAX_NULL_RATIO = 0.30
_REQUIRED_COLUMNS = {
    "aircraft_id",
    "latitude",
    "longitude",
    "baro_altitude",
    "event_timestamp",
}

STATE_SCHEMA = StructType(
    [
        StructField("icao24", StringType(), True),
        StructField("callsign", StringType(), True),
        StructField("origin_country", StringType(), True),
        StructField("time_position", LongType(), True),
        StructField("last_contact", LongType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("baro_altitude", DoubleType(), True),
        StructField("on_ground", BooleanType(), True),
        StructField("velocity", DoubleType(), True),
        StructField("true_track", DoubleType(), True),
        StructField("vertical_rate", DoubleType(), True),
        StructField("sensors", StringType(), True),  # stored as string (list repr)
        StructField("geo_altitude", DoubleType(), True),
        StructField("squawk", StringType(), True),
        StructField("spi", BooleanType(), True),
        StructField("position_source", IntegerType(), True),
    ]
)

BRONZE_SCHEMA = StructType(
    [
        StructField("fetch_timestamp", LongType(), True),
        StructField("ingest_timestamp", LongType(), True),
        StructField("aircraft_count", IntegerType(), True),
        StructField("states", ArrayType(STATE_SCHEMA), True),
    ]
)


def run(spark: SparkSession, pipeline_run_id: str, lookback_hours: int = 2) -> dict:
    meta = StageMetadata("bronze_to_silver", pipeline_run_id)

    try:
        logger.info(f"Reading bronze JSON from {BRONZE_BASE}")
        raw_df = (
            spark.read.option("multiLine", "true")
            .option("recursiveFileLookup", "true")
            .schema(BRONZE_SCHEMA)
            .json(BRONZE_BASE)
        )

        raw_count = raw_df.count()
        meta.add_metric("bronze_files_read", raw_count)
        logger.info(f"Read {raw_count} bronze records")

        exploded = raw_df.select(
            F.col("fetch_timestamp"),
            F.col("ingest_timestamp"),
            F.explode(F.col("states")).alias("s"),
        ).select(
            "fetch_timestamp",
            "ingest_timestamp",
            F.col("s.icao24").alias("aircraft_id"),
            F.col("s.callsign"),
            F.col("s.origin_country"),
            F.col("s.time_position"),
            F.col("s.last_contact"),
            F.col("s.longitude"),
            F.col("s.latitude"),
            F.col("s.baro_altitude"),
            F.col("s.on_ground"),
            F.col("s.velocity"),
            F.col("s.true_track").alias("heading"),
            F.col("s.vertical_rate"),
            F.col("s.geo_altitude"),
            F.col("s.squawk"),
            F.col("s.position_source"),
        )

        # timestamp normalisation unix -> UTC
        with_ts = exploded.withColumn(
            "event_timestamp", F.to_timestamp(F.col("time_position"))
        ).withColumn("event_date", F.to_date(F.col("event_timestamp")))

        # drop rows with null spatial data

        not_null = with_ts.filter(
            F.col("latitude").isNotNull()
            & F.col("longitude").isNotNull()
            & F.col("baro_altitude").isNotNull()
            & F.col("aircraft_id").isNotNull()
        )

        # filter outliers
        filtered = not_null.filter(
            ((F.col("baro_altitude") >= -500) & (F.col("baro_altitude") <= 15000))
            & ((F.col("velocity") >= 0) & (F.col("velocity") <= 400))
            & ((F.col("latitude") >= -90) & (F.col("latitude") <= 90))
            & ((F.col("longitude") >= -180) & (F.col("longitude") <= 180))
        )

        # deduplicate of aircraft_id and time_position
        deduped = filtered.drop_duplicates(["aircraft_id", "time_position"])

        silver_df = deduped.select(
            "aircraft_id",
            "callsign",
            "origin_country",
            "event_timestamp",
            "event_date",
            "latitude",
            "longitude",
            "baro_altitude",
            "geo_altitude",
            "on_ground",
            "velocity",
            "heading",
            "vertical_rate",
            "squawk",
            "position_source",
            F.year("event_timestamp").alias("year"),
            F.month("event_timestamp").alias("month"),
            F.dayofmonth("event_timestamp").alias("day"),
        )

        silver_count = silver_df.count()
        meta.add_metric("silver_rows_count", silver_count)
        meta.add_metric("rows_dropped", (raw_count - silver_count))

        missing_cols = _REQUIRED_COLUMNS - set(silver_df.columns)
        if missing_cols:
            raise DataQualityError(
                "Silver schema missing required columns",
                context={
                    "missing": sorted(missing_cols),
                    "pipeline_run_id": pipeline_run_id,
                },
            )
        if silver_count < _MIN_SILVER_ROWS:
            raise DataQualityError(
                f"Silver rows count {silver_count} below minimum {_MIN_SILVER_ROWS}",
                context={
                    "silver_count": silver_count,
                    "pipeline_run_id": pipeline_run_id,
                },
            )

        total = max(silver_count, 1)
        null_checks = {
            col: silver_df.filter(F.col(col).isNull()).count() / total
            for col in ["aircraft_id", "latitude", "longitude", "baro_altitude"]
        }

        bad_cols = {c: r for c, r in null_checks.items() if r > _MAX_NULL_RATIO}
        if bad_cols:
            raise DataQualityError(
                "Null ratio exceeded in silver data columns",
                context={"columns": bad_cols, "pipeline_run_id": pipeline_run_id},
            )

        logger.info(
            f"Data Quality gate check completed for rows: {silver_count} and null_ratios: {null_checks}"
        )

        (
            silver_df.coalesce(4)
            .write.mode("append")
            .partitionBy("year", "month", "day")
            .parquet(SILVER_BASE)
        )

        meta.add_artifact("silver_path", SILVER_BASE)
        meta.status = "Success"
        logger.info(f"Silver: written {silver_count} rows to {SILVER_BASE} path")

    except Exception as e:
        meta.status = "failed"
        meta.error = str(e)
        logger.error(f"Bronze to silver processing failed: {e}", exc_info=True)
        raise
    finally:
        meta.save()

    return meta.to_dict()


def main() -> None:
    logger.info("Starting bronze to silver pipeline")
    run_id = os.getenv(
        "PIPELINE_RUN_ID", datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    )
    spark = get_spark_session("bronze_to_silver")
    try:
        run(spark, run_id)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
