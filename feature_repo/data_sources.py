"""
Feast data sources pointing to gold parquet files in MinIO.
Uses s3fs-compatible file paths.

"""

import os
from feast import FileSource
from feast.data_format import ParquetFormat

BUCKET = os.getenv("MINIO__BUCKET", "aviation-lake")

# Gold parquet file paths accessible via s3fs
flight_state_source = FileSource(
    name="flight_state_source",
    path=f"s3://{BUCKET}/gold/flight_state_features",
    s3_endpoint_override="http://minio:9000",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Per-aircraft flight state features from silver layer Spark Job.",
)

airport_congestion_source = FileSource(
    name="airport_congestion_source",
    path=f"s3://{BUCKET}/gold/airport_congestion_features",
    s3_endpoint_override="http://minio:9000",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Per-airport congestion features aggregated from aircraft within 50km",
)

route_features_source = FileSource(
    name="route_airport_source",
    path=f"s3://{BUCKET}/gold/route_features/",
    s3_endpoint_override="http://minio:9000",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Per-route aggregated features derived from callsign patterns.",
)
