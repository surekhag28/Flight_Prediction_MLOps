"""
Tests for ML training modules.

Focuses only on contract tests (input/output) rather than model accuracy.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

DELAY_FEATURES = [
    "speed_ms",
    "altitude_m",
    "vertical_rate_ms",
    "heading_change_5m",
    "avg_speed_15m",
    "aircraft_count_50km",
    "congestion_score",
    "hour_of_day",
    "day_of_week",
]


def make_delay_data(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "speed_ms": rng.uniform(0, 300, n),
            "altitude_m": rng.uniform(0, 12000, n),
            "vertical_rate_ms": rng.uniform(-20, 20, n),
            "heading_change_5m": rng.uniform(0, 90, n),
            "avg_speed_15m": rng.uniform(0, 300, n),
            "aircraft_count_50km": rng.integers(0, 80, n),
            "congestion_score": rng.uniform(0, 1, n),
            "hour_of_day": rng.integers(0, 24, n),
            "day_of_week": rng.integers(1, 8, n),
            "delay_risk": rng.integers(0, 2, n),
        }
    )

    return df


class TestDelayModelContract:
    def test_lightgbm_trains_and_predicts_proba(self):
        from lightgbm import LGBMClassifier

        df = make_delay_data(300)
        X = df[DELAY_FEATURES]
        y = df["delay_risk"]
        model = LGBMClassifier(n_estimators=20, random_state=42, verbose=1)
        model.fit(X, y)
        probas = model.predict_proba(X[:5])
        assert probas.shape == (5, 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)

    def test_delay_proba_in_range(self):
        from lightgbm import LGBMClassifier

        df = make_delay_data(300)
        X = df[DELAY_FEATURES]
        y = df["delay_risk"]
        model = LGBMClassifier(n_estimators=20, random_state=42, verbose=1)
        model.fit(X, y)
        proba = model.predict_proba(X[:1])[:, 1][0]
        assert 0 <= proba <= 1

    def test_risk_label_mapping(self):
        def risk_mapping(p: float) -> str:
            if p >= 0.7:
                return "HIGH"
            if p >= 0.4:
                return "MEDIUM"
            return "LOW"

        assert risk_mapping(0.8) == "HIGH"
        assert risk_mapping(0.5) == "MEDIUM"
        assert risk_mapping(0.3) == "LOW"
        assert risk_mapping(0.4) == "MEDIUM"
        assert risk_mapping(0.7) == "HIGH"
