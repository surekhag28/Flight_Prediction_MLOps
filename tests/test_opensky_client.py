"""
Test suite for OpenSky ingestion client
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from src.ingestion.opensky_client import (
    parse_states,
    validate_states,
    build_bronze_key,
    ingest,
)


def test_parse_states_normal():

    raw = {
        "states": [
            [
                "3c4b26",
                "DLH2A   ",
                "Germany",
                1700000000,
                1700000001,
                8.55,
                50.03,
                9500.0,
                False,
                240.0,
                90.0,
                0.0,
                None,
                9800.0,
                "1234",
                False,
                0,
            ]
        ]
    }

    result = parse_states(raw)
    print(result)

    assert len(result) == 1
    assert result[0]["icao24"] == "3c4b26"
    assert result[0]["callsign"] == "DLH2A"
    assert result[0]["longitude"] == 8.55
    assert result[0]["latitude"] == 50.03
    assert result[0]["baro_altitude"] == 9500.0
    assert result[0]["on_ground"] is False


def test_parse_states_empty():
    assert parse_states({}) == []


def test_parse_states_none_values():
    raw = {
        "states": [
            [
                "3c4b26",
                "DLH2A   ",
                "Germany",
                1700000000,
                1700000001,
                8.55,
                50.03,
                9500.0,
                False,
                240.0,
                90.0,
                0.0,
                None,
                9800.0,
                "1234",
                False,
                0,
            ],
            None,
        ]
    }

    result = parse_states(raw)
    assert len(result) == 1

    raw = {"states": [None]}
    assert (parse_states(raw)) == []


def test_parse_states_callsign_strip():
    raw = {"states": [["abc123", "   DLH2A   ", "USA", 1, 2, 0, 0]]}

    result = parse_states(raw)
    assert result[0]["callsign"] == "DLH2A"


def test_parse_empty_states():
    assert parse_states({}) == []


def test_validate_states_filters_none():
    states = [
        {"latitude": 1, "longitude": 2},
        {"latitude": None, "longitude": None},
        {"latitude": 1, "longitude": None},
    ]
    result = validate_states(states)
    assert len(result) == 1


def test_validate_logs_dropped(mocker):
    mock_logger = mocker.patch("src.ingestion.opensky_client.logger")
    states = [
        {"latitude": 1, "longitude": 2},
        {"latitude": 1, "longitude": None},
    ]

    validate_states(states)
    mock_logger.info.assert_called_once()


def test_bronze_key():
    # f"bronze/opensky/year={ts.year}/month={ts.month:02d}/day={ts.day:02d}/batch_{ts.strftime('%Y%m%d_%H%M%S')}.json"
    ts = datetime(2026, 6, 6, 12, 30, 15, tzinfo=timezone.utc)
    key = build_bronze_key(ts)

    assert key == "bronze/opensky/year=2026/month=06/day=06/batch_20260606_123015.json"


def test_ingest(mocker):
    mock_fetch = mocker.patch("src.ingestion.opensky_client.fetch_states")
    mock_parse = mocker.patch("src.ingestion.opensky_client.parse_states")
    mock_validate = mocker.patch("src.ingestion.opensky_client.validate_states")
    mock_bucket = mocker.patch("src.ingestion.opensky_client.ensure_bucket")
    mock_put = mocker.patch("src.ingestion.opensky_client.put_json")

    mock_fetch.return_value = {"time": 123, "states": []}
    mock_parse.return_value = [
        {"latitude": 1, "longitude": 2},
        {"latitude": 1, "longitude": None},
    ]
    mock_validate.return_value = [{"latitude": 1, "longitude": 2}]

    summary = ingest(datetime(2026, 6, 6, tzinfo=timezone.utc))

    mock_bucket.assert_called_once()
    mock_put.assert_called_once()

    assert summary["status"] == "success"
    assert summary["aircraft_count"] == 1


def test_ingest_pipeline(mocker):

    mock_get = mocker.patch("src.ingestion.opensky_client.requests.get")
    mock_put = mocker.patch("src.ingestion.opensky_client.put_json")

    SAMPLE_RESPONSE = {
        "time": 1700000000,
        "states": [
            [
                "3c4b26",
                "DLH2A   ",
                "Germany",
                1700000000,
                1700000001,
                8.55,
                50.03,
                9500.0,
                False,
                240.0,
                90.0,
                0.0,
                None,
                9800.0,
                "1234",
                False,
                0,
            ],
            [
                "a12345",
                "        ",
                "USA",
                1700000000,
                1700000001,
                None,
                None,
                0.0,
                True,
                0.0,
                0.0,
                0.0,
                None,
                None,
                None,
                False,
                0,
            ],
        ],
    }
    mock_resp = mocker.Mock()
    mock_resp.json.return_value = SAMPLE_RESPONSE
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp

    result = ingest()

    mock_put.assert_called_once()
    assert result["status"] == "success"
    assert result["aircraft_count"] == 1
