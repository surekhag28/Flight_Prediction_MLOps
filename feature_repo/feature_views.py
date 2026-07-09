"""
Feast feature view definitions for flight prediction platform.
Three feature views covering all three ML products.
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float32, Float64, Int64, String

from feature_repo.data_sources import (
    airport_congestion_source,
    flight_state_source,
    route_features_source,
)

from feature_repo.entities import aircraft, airport, route

# Per aircraft flight state features
flight_state_features = FeatureView(
    name="fligt_state_features",
    entities=[aircraft],
    ttl=timedelta(
        hours=2,
    ),
    schema=[
        Field(name="speed_ms", dtype=Float32),
        Field(name="altitude_m", dtype=Float32),
        Field(name="vertical_rate_ms", dtype=Float32),
        Field(name="heading_deg", dtype=Float32),
        Field(name="heading_change_5m", dtype=Float32),
        Field(name="avg_speed_15m", dtype=Float32),
        Field(name="distance_10m_km", dtype=Float32),
        Field(name="hour_of_day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="is_on_ground", dtype=Int64),
        Field(name="latitude", dtype=Float64),
        Field(name="longitude", dtype=Float64),
        Field(name="callsign", dtype=String),
        Field(name="origin_country", dtype=String),
    ],
    source=flight_state_source,
    description="Per aircraft telemetry features: speed, altitude, heading changes, rolling averages",
    tags={"product": "A,C", "team": "aviation_ml"},
)


# Per airport congestion features
airport_congestion_features = FeatureView(
    name="airport_congestion_features",
    entities=[airport],
    ttl=timedelta(hours=1),
    schema=[
        Field(name="aircraft_count_50km", dtype=Int64),
        Field(name="arrivals_last_30m", dtype=Int64),
        Field(name="departures_last_30m", dtype=Int64),
        Field(name="avg_altitude_50km", dtype=Float32),
        Field(name="congestion_score", dtype=Float32),
        Field(name="airport_name", dtype=String),
    ],
    source=airport_congestion_source,
    description="Per airport congestion metrics: aircraft count within 50km radius, arrivals, departure",
    tags={"product": "B", "team": "aviation_ml"},
)

route_features = FeatureView(
    name="route_features",
    entities=[route],
    ttl=timedelta(days=7),
    schema=[
        Field(name="route_density", dtype=Int64),
        Field(name="avg_cruise_speed", dtype=Float32),
        Field(name="historical_avg_delay_proxy", dtype=Float32),
        Field(name="unique_aircraft_count", dtype=Int64),
    ],
    source=route_features_source,
    description="Per route historical aggregates: density, cruise speed, delay proxy",
    tags={"product": "A", "team": "aviation_ml"},
)
