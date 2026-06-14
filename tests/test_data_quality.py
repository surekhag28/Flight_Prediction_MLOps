from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


def test_success_rate_full_pass():
    from src.data_quality.validate import ValidationbResult

    result = ValidationbResult("bronze_suite", True, 10, 10, [])
    assert result.success_rate == 1.0


def test_success_rate_partial_pass():
    from src.data_quality.validate import ValidationbResult

    result = ValidationbResult("bronze_suite", True, 10, 8, [])
    assert result.success_rate == pytest.approx(0.8)


def test_success_rate_zero_evaluated():
    from src.data_quality.validate import ValidationbResult

    result = ValidationbResult("bronze_suite", True, 0, 0, [])
    assert result.success_rate == 1.0


def test_to_dict():
    from src.data_quality.validate import ValidationbResult

    result = ValidationbResult("bronze_suite", True, 10, 8, [])

    expected = {
        "suite_name": "bronze_suite",
        "passed": True,
        "evaluated_expectations": 10,
        "successful_expectations": 8,
        "failed_expectations": [],
        "success_rate": round(0.8, 4),
        "statistics": {},
    }

    assert result.to_dict() == expected


def _make_bronze_df(n: int = 200, **overrides) -> pd.DataFrame:
    """Minimal valid bronze dataset (airborne aircraft)."""
    rng = np.random.default_rng(42)
    data = {
        "icao24": [f"abc{i:03d}" for i in range(n)],
        "callsign": [f"FLT{i:04d}" for i in range(n)],
        "origin_country": ["GB"] * n,
        "longitude": rng.uniform(-180, 180, n).tolist(),
        "latitude": rng.uniform(-90, 90, n).tolist(),
        "baro_altitude": rng.uniform(1000, 12000, n).tolist(),
        "on_ground": [False] * n,
        "velocity": rng.uniform(50, 300, n).tolist(),
        "true_track": rng.uniform(0, 360, n).tolist(),
        "vertical_rate": rng.uniform(-10, 10, n).tolist(),
        "time_position": [1_700_000_000.0 + i * 15 for i in range(n)],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_valid_data_pass():
    from src.data_quality.validate import validate_bronze

    result = validate_bronze(_make_bronze_df())
    assert result.passed is True
    assert result.evaluated_expectations > 0
    assert len(result.failed_expectations) == 0


def test_bronze_suite_name():
    from src.data_quality.validate import validate_bronze

    result = validate_bronze(_make_bronze_df())
    assert result.suite_name == "bronze_suite"


def test_missing_required_column_fails():
    from src.data_quality.validate import validate_bronze

    df = _make_bronze_df().drop(columns=["icao24"])
    result = validate_bronze(df)
    assert result.passed is False
    types = [exp["expectation_type"] for exp in result.failed_expectations]
    assert "expect_table_columns_to_match_set" in types


def test_all_null_icao24_faills():
    from src.data_quality.validate import validate_bronze

    df = _make_bronze_df()
    df["icao24"] = None
    result = validate_bronze(df)
    assert result.passed is False
    types = [exp["expectation_type"] for exp in result.failed_expectations]
    assert "expect_column_values_to_not_be_null" in types


def test_row_count_between_minimum_fails():
    from src.data_quality.validate import validate_bronze

    df = _make_bronze_df(n=50)
    df.loc[:4, "velocity"] = -10
    result = validate_bronze(df)
    assert result.passed is False

    types = [exp["expectation_type"] for exp in result.failed_expectations]
    assert "expect_column_values_to_be_between" in types


def test_result_has_statistics():
    from src.data_quality.validate import validate_bronze

    df = _make_bronze_df()
    result = validate_bronze(df)
    assert result.statistics["row_count"] == len(df)
    assert result.statistics["column_count"] == len(df.columns)
