-- Creates the mlflow database alongside the airflow database
-- Runs automatically on first postgres container start
CREATE DATABASE mlflow OWNER airflow;