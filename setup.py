import os

dirs = [
    "airflow/dags",
    "dashboard",
    "docs",
    "feature_repo",
    "monitoring" "src/config",
    "src/data_quality",
    "src/drift",
    "src/feature_store",
    "src/inference",
    "src/ingestion",
    "src/ml",
    "src/monitoring",
    "src/processing",
    "src/streaming",
    "src/utils",
    "notebooks",
    "tests",
]

files = ["docker-compose.yml", "README.md"]

for d in dirs:
    os.makedirs(d, exist_ok=True)

for f in files:
    open(f, "a").close()

print("Project scaffold created")
