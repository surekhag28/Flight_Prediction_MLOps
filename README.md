# Flight Prediction MLOps Platform

An end-to-end MLOps system that turns live flight telemetry from the OpenSky Network into three production grade predictions:

- Flight predictions
- Flight Anomalies

It covers the full lifecyle right from streaming, feature ngineering on a lakehouse, training with hyperparameters search, experiment tracking, real-time inference with caching and A/B testing, monitoring, drift detection, and automated retraining.

---

## Architecture


---

## What It Does

1. Ingest live flight states from OpenSky REST API every 15 minutes and from a Kafka stream in real time.
2. Cleans and transforms the data through Bronze → Silver → Gold layers on MinIO (a medallion lakehouse).

---

## Services

All services are managed using Docker Compose: `docker compose up`


| Service | URL | Credentials |
|----------|-----|--------------|
| Airflow | http://localhost:8081 | airflow / airflow |
| MinIO Console | http://localhost:9003 | minioadmin / minioadmin |


---

## Project Structure

```
flight_prediction_mlops/
├── airflow/dags/            # 5 Airflow DAGs (ingest, retrain, monitor, ab, feast)
├── src/
│   ├── config/              # Pydantic settings + settings.yaml
│   ├── ingestion/           # OpenSky REST client → bronze writes
│   ├── data_quality/       # Great Expectations suites + validator
│   ├── processing/          # Spark ETL: bronze→silver, silver→gold (×3)
│   ├── feature_store/       # Feast entities, views, data sources
│   ├── streaming/           # Kafka producer + aiokafka feature processor
│   ├── ml/                  # Training, HPO, AutoML, evaluation, registry, SHAP
│   ├── inference/           # FastAPI app, auth, cache, A/B router, schemas
│   ├── drift/               # Evidently drift detection
│   ├── monitoring/          # Performance monitoring
│   └── utils/               # Logging, exceptions, Spark / MinIO / MLflow helpers
├── tests/                   # 169 unit + contract tests
├── docs/                    # Architecture, API ref, data pipeline, ML, ops, dev guides
├── Dockerfile
└── docker-compose.yml       # 12-service stack
```