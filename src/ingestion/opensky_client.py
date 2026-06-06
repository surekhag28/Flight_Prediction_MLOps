"""
OpenSky Network API client.
Fetches global flight state vectors and stores raw JSON to MinIO bronze layer.

Free API: https://opensky-network.org/api/states/all
- No auth required for basic access
- Rate limit: 10s between requests (we poll every 15 min)
- Returns ~7000-10000 aircraft state vectors globally

State vector fields (positional array):
[0]  icao24         - ICAO 24-bit hex address
[1]  callsign       - callsign
[2]  origin_country - registering country
[3]  time_position  - unix timestamp of last position update
[4]  last_contact   - unix timestamp of last signal received
[5]  longitude      - degrees
[6]  latitude       - degrees
[7]  baro_altitude  - barometric altitude (meters)
[8]  on_ground      - bool
[9]  velocity       - ground speed (m/s)
[10] true_track     - heading clockwise from north (degrees)
[11] vertical_rate  - climb rate (m/s), negative = descending
[12] sensors        - list of sensor IDs (may be None)
[13] geo_altitude   - geometric altitude (meters)
[14] squawk         - transponder code
[15] spi            - special purpose indicator
[16] position_source - 0=ADS-B, 1=ASTERIX, 2=MLAT, 3=FLARM
"""

from __future__ import annotations

import os
import json
import time
import requests
from datetime import UTC, datetime
from dotenv import load_dotenv

from src.core.logger import get_logger
from src.utils.minio_utils import ensure_bucket, put_json

logger = get_logger(__name__)
load_dotenv()

OPENSKY_URL = "https://opensky-network.org/api/states/all"
OPENSKY_TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
OPENSKY_USERNAME = os.getenv("OPENSKY__USERNAME", "")
OPENSKY_PASSWORD = os.getenv("OPENSKY__PASSWORD", "")
OPENSKY_CLIENT_ID = os.getenv("OPENSKY__CLIENT_ID", "")
OPENSKY_CLIENT_SECRET = os.getenv("OPENSKY__CLIENT_SECRET", "")

_token_cache: dict = {}

BUCKET = os.getenv("MINIO__BUCKET", "aviation-lake")

# Column names matching OpenSky state vector positional fields
STATE_COLUMNS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source",
]


def _get_bearer_token() -> str | None:
    """Fetch OAuth2 bearer token using client credentials and cache it until expiry"""

    if not OPENSKY_CLIENT_ID:
        return None
    now = time.time()
    if _token_cache.get("expires_at", 0) > now + 30:
        return _token_cache["access_token"]
    resp = requests.post(
        OPENSKY_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": OPENSKY_CLIENT_ID,
            "client_secret": OPENSKY_CLIENT_SECRET,
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    logger.info(data)
    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = now + data.get("expires_in", 300)

    return _token_cache["access_token"]


def fetch_states(timeout: int = 30) -> dict:
    """Fetch all aircraft states from OpenSky API."""

    headers = {}
    auth = None

    try:
        token = _get_bearer_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif OPENSKY_USERNAME:
            auth = (OPENSKY_USERNAME, OPENSKY_PASSWORD)
    except Exception as e:
        logger.warning(f"Failed to get bearer token, falling back to anonymous: {e}")

    try:
        response = requests.get(
            OPENSKY_URL, auth=auth, headers=headers, timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning("Rate limit exceeded by OpenSky API, sleeping 60secs")
            logger.info(e.response)
            time.sleep(60)
        raise


def parse_states(raw: dict) -> list[dict]:
    """Convert raw OpenSky state vectors to list of dict object"""

    states = raw.get("states") or []
    parsed = []

    for state in states:
        if state is None:
            continue
        record = {
            col: state[i] if i < len(state) else None
            for i, col in enumerate(STATE_COLUMNS)
        }
        if record.get("callsign"):
            record["callsign"] = record["callsign"].strip()
        parsed.append(record)

    return parsed


def build_bronze_key(ts: datetime) -> str:
    return f"bronze/opensky/year={ts.year}/month={ts.month:02d}/day={ts.day:02d}/batch_{ts.strftime('%Y%m%d_%H%M%S')}.json"


def ingest(run_ts: datetime | None = None) -> dict:
    """
    Fetch > Validate > Store to MinIO bronze layer
    """

    if run_ts is None:
        run_ts = datetime.now(tz=UTC)

    start = time.time()
    raw = fetch_states()
    fetch_ts = raw.get("time", int(run_ts.timestamp()))
    states = parse_states(raw)
    states = validate_states(states)

    payload = {
        "fetch_ts": fetch_ts,
        "ingest_timestamp": run_ts.isoformat(),
        "aircraft_count": len(states),
        "states": states,
    }

    ensure_bucket(BUCKET)
    key = build_bronze_key(run_ts)
    put_json(key, payload, bucket=BUCKET)

    duration = round(time.time() - start, 2)
    summary = {
        "bronze_key": key,
        "aircraft_count": len(states),
        "fetch_timestamp": fetch_ts,
        "duration_sec": duration,
        "status": "success",
    }

    logger.info("fOpensky ingestion layer : {key}")
    return summary


def validate_states(states: list[dict]) -> list[dict]:
    valid = [
        state
        for state in states
        if state.get("latitude") is not None and state.get("longitude") is not None
    ]
    if len(valid) != len(states):
        dropped = len(states) - len(valid)
        logger.info(f"Dropped {dropped} states because of missing lat/long")

    return valid


# if __name__ == "__main__":
#     result = fetch_states()
#     summary, payload = ingest()
#     # print(json.dumps(result, indent=2))

#     import pandas as pd

#     df = pd.DataFrame(payload)

#     from src.data_quality.validate import validate_bronze

#     validate_bronze(df)
