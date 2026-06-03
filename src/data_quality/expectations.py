"""
Great Expectations rules for three data layers.
Suites:
    bronze: raw API data: schema, physical bounds, row count
    silver: cleaned data: completeness, deduplication
    gold: feature-ready data: distribution bounds, derived-feature ranges
"""

from __future__ import annotations
import great_expectations.expectations as gxe


def add_bronze_expectations():
    """
    Raw OpenSky API data expectations
    Validates schema, physical bounds and minimu row count
    Critical failure should absort the pipeline

    Returns:
        _type_: _description_
    """

    required_columns = [
        "icao24",
        "callsign",
        "origin_country",
        "longitude",
        "latitude",
        "baro_altitude",
        "on_ground",
        "velocity",
        "true_track",
        "vertical_rate",
        "timestamp",
    ]

    expectations = [
        gxe.ExpectTableColumnsToMatchSet(
            column_set=required_columns, exact_match=False
        ),
        gxe.ExpectColumnValuesToNotBeNull(column="icao24"),
    ]

    bounds = {
        "velocity": (0, 600),
        "baro_latitude": (-500, 15000),
        "latittude": (-90, 90),
        "longitude": (-180, 180),
        "true_track": (0, 360),
    }

    expectations.extend(
        [
            gxe.ExpectColumnValuesToBeBetween(
                column=col, min_value=min_val, max_value=max_val, mostly=0.99
            )
            for col, (min_val, max_val) in bounds.items()
        ]
    )

    return expectations
