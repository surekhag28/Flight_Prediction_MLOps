# Data Engineering Pipeline

# Overview
The data pipeline runs in three stages: ingestion(raw->bronze), ETL(bronze->silver->gold) and feature materialisation(gold->redis). Each stage is orchestrated by Airflow DAGs and stores data in MinIO's S3 compatible object store.

# Airflow DAGs

DAG | Schedule | Purpose |
opensky_ingest_dag | Every 15mins | Ingest OpenSky API -> Bronze -> Silver -> Gold |
feast_materialize_dag | Hourly (:30) | Materialize gold features to Redis |
monitor_dag | Daily 06:00 UTC | Drift + performance checks, trigger retrain |
retrain_dag | Monthly + on-demand | Full HPO + training pipeline


# Stage 1 - Ingestion (Bronze)
DAG: opensky_ingest_dag
Source: OpenSky Network REST API (https://opensky-network.org/api/states/all) 
Destination: aviation-lake/bronze/flights/<run_id>/data.json

# What it does?

src/ingestion/opensky_client.py calls the OpenSky /states/all endpoint and downloads the full live flight state vector. Each response contains up to ~10,000 aircraft with:

icao24 — ICAO 24-bit address (hex)
callsign — flight callsign
origin_country — country of registration
time_position — Unix timestamp of last position update
longitude, latitude — WGS84 coordinates
baro_altitude — barometric altitude (meters)
on_ground — boolean ground status
velocity — ground speed (m/s)
true_track — heading (degrees from north)
vertical_rate — climb/descent rate (m/s)
sensors — list of receiver IDs
geo_altitude — GPS altitude (meters)
squawk — transponder code
spi — special purpose indicator

The raw JSON is written directly to MinIO under the bronze/ prefix, partitioned by run ID.
No transformation is applied at this stage.

# Authentication
OpenSky provides a free unauthenticated tier with rate limits. Set OPENSKY_USERNAME and OPENSKY_PASSWORD in .env to use an authenticated account with higher limits.

# Stage 2 - ETL (Silver & Gold Layer)
Tool: Apache Spark (embedded in the Airflow container)
Source: aviation-lake/bronze/
Destination: aviation-lake/silver/, aviation-lake/gold

Silver Layer: Data preprocessing and Cleaning

src/processing/bronze_to_silver.py reads bronze JSON, applies these transformations:

1. Drop records with null ica024, latitude or longitude
2. Drop duplicate icao24 within the same run(keep latest timestamp)
3. Cast all numeric fields to correct types.
4. Rename columns to consistent snake_case names (velocity, speed_ms, baro_altitude, altitude_m, vertical_rate, vertical_rate_ms)
5. Compute distance_10m_km: estimated distance travelled using speed x time window
6. Write as partitioned parquet to silver/flights

Gold Layer - Feature Engineering

Three separate jobs produce gold feature tables:

gold/flights

src/processing/silver_to_gold.py:

1. 

gold/congestion


gold/routes






