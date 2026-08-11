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


CONGESTION_FEATURES = [
    "aircraft_count_50km",
    "arrivals_last_30m",
    "departures_last_30m",
    "hour_of_day",
    "day_of_week",
    "avg_altitude_50km",
]


def make_congestion_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    aircraft_count = rng.integers(0, 80, n).astype(float)
    df = pd.DataFrame(
        {
            "aircraft_count_50km": aircraft_count,
            "arrivals_last_30m": rng.integers(0, 40, n).astype(float),
            "departures_last_30m": rng.integers(0, 40, n).astype(float),
            "hour_of_day": rng.integers(0, 24, n).astype(float),
            "day_of_week": rng.integers(1, 8, n).astype(float),
            "avg_altitude_50km": rng.uniform(0, 12000, n),
            "congestion_score": aircraft_count / 80.0,
        }
    )

    return df


class TestCongestionModelContract:

    def test_lgbm_regressor_trains(self):
        from lightgbm import LGBMRegressor

        df = make_congestion_df(300)
        X = df[CONGESTION_FEATURES]
        y = df["congestion_score"]

        model = LGBMRegressor(n_estimators=20, random_state=42, verbose=1)
        model.fit(X, y)
        preds = model.predict(X[:5])

        assert len(preds) == 5


ANOMALY_FEATURES = [
    "heading_change_5m",
    "speed_ms",
    "altitude_m",
    "vertical_rate_ms",
    "distance_10m_km",
]


def make_anomaly_dataframe(n: int = 200) -> pd.DataFrame:

    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "heading_change_5m": rng.uniform(0, 180, n),
            "speed_ms": rng.uniform(0, 300, n),
            "altitude_m": rng.uniform(0, 12000, n),
            "vertical_rate_ms": rng.uniform(-20, 20, n),
            "distance_10m_km": rng.uniform(0, 200, n),
        }
    )


class TestAnomalyModelContract:

    def test_isolation_forest_train(self):
        from sklearn.ensemble import IsolationForest
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        df = make_anomaly_dataframe(300)
        X = df[ANOMALY_FEATURES]
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "iso",
                    IsolationForest(
                        n_estimators=20, contamination=0.05, random_state=42
                    ),
                ),
            ]
        )

        pipe.fit(X)
        scores = pipe.decision_function(X[:5])
        assert len(scores) == 5

    def test_anomaly_flag_conversion(self):
        """IsolationForest returns -1 (anomaly) or 1 (normal); converts to 0/1 flag"""

        raw_preds = [-1, 1, -1, 1, -1]
        flags = [0 if p == -1 else 1 for p in raw_preds]
        assert flags == [0, 1, 0, 1, 0]

    def test_anomaly_rate(self):
        import numpy as np
        from sklearn.ensemble import IsolationForest

        rng = np.random.default_rng(42)
        X = rng.standard_normal((1000, 5))
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X)

        preds = model.predict(X)
        anomaly_rate = (preds == -1).mean()
        assert 0.03 <= anomaly_rate <= 0.07


# evaluate.py contract -------------------------------


class TestEvaluateDelay:
    """Unit tests for evaluate_delay() with mocked Mlflow client."""

    def _make_client(
        self, candidate_auc=0.85, candidate_f1=0.75, prod_auc=None, has_prod=True
    ):
        from unittest.mock import MagicMock

        client = MagicMock()
        candidate_run = MagicMock()
        candidate_run.data.metrics = {"auc_roc": candidate_auc, "f1": candidate_f1}
        client.get_run.return_value = candidate_run
        if has_prod and prod_auc is not None:
            prod_version = MagicMock()
            prod_version.run_id = "prod-run-id"
            prod_run = MagicMock()
            prod_run.data.metrics = {"auc_roc": prod_auc, "f1": 0.70}
            client.get_latest_versions.return_value = [prod_version]
            client.get_run.side_effects = lambda rid: {
                prod_run if rid == "prod-run-id" else candidate_run
            }
        elif not has_prod:
            client.get_latest_versions.return_value = []

        return client

    def test_promotes_when_no_production_baseline(self):
        from unittest.mock import patch

        from src.ml.evaluate import evaluate_delay

        client = self._make_client(
            candidate_auc=0.80, candidate_f1=0.75, has_prod=False
        )
        with (
            patch("src.ml.evaluate.MlflowClient", return_value=client),
            patch("src.ml.evaluate.setup_mlflow"),
        ):
            assert evaluate_delay("cand-run") is True

    def test_promotes_when_candidate_beats_production(self):
        from unittest.mock import patch
        from src.ml.evaluate import evaluate_delay

        client = self._make_client(candidate_auc=0.88, candidate_f1=0.75, prod_auc=0.84)
        with (
            patch("src.ml.evaluate.MlflowClient", return_value=client),
            patch("src.ml.evaluate.setup_mlflow"),
        ):
            assert evaluate_delay("cand-run") is True

    def test_keeps_production_when_candidate_worse(self):
        from unittest.mock import patch
        from src.ml.evaluate import evaluate_delay

        client = self._make_client(candidate_auc=0.80, candidate_f1=0.72, prod_auc=0.85):
        with patch("src.ml.evaluate.MlflowClient", return_value=client), patch("src.ml.evaluate.setup_mlflow"):
            assert evaluate_delay("cand-run") is False

    def test_fails_quality_gate_low_auc(self):
        from unittest.mock import patch
        from src.ml.evaluate import evaluate_delay

        client = self._make_client(
            candidate_auc=0.50, candidate_f1=0.72, has_prod=False
        )
        with (
            patch("src.ml.evaluate.MlflowClient", return_value=client),
            patch("src.ml.evaluate.setup_mlflow"),
        ):
            assert evaluate_delay("cand-run") is False


class TestEvaluateAnomalyAlwaysPromotes:
    def test_always_promotes(self):
        from unittest.mock import patch
        from src.ml.evaluate import evaluate_anomaly

        with patch("src.ml.evaluate.setup_mlflow"):
            assert evaluate_anomaly("cand-run") is True


# algorithm registry and train test contract------------------------

class TestTrainTestWithAlgorithmRegistry:
    """Verify that every registry algorithm can build, fit and predict"""

    def _make_clf_data(self, n=200, n_features=5):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n,n_features))
        y = rng.integers(0,2,n)
        return X,y

    def _make_regressor_data(self,n=200,n_features=5):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n,n_features))
        y = rng.uniform(0,1)
        return X, y

    def _make_anomaly_data(self,n=200,n_features=5):
        rng = np.random.default_rng(0)
        return rng.standard_normal((n,n_features))

    def test_all_classifiers_fit_predict_proba(self):
        from src.ml.algorithms import CLASSIFIER_REGISTRY
        X,y = self._make_clf_data()
        X_train,X_test = X[:160], X[160:]
        for name, spec in CLASSIFIER_REGISTRY.items():
            params = _minimal_clf_params(name)
            model = spec.factory(params, random_state=42)
            model.fit(X_train, y[:160])
            proba = model.predict_proba(X_test)
            assert proba.shape == (40,2), f"{name}: wrong proba shape"
            assert np.all(proba>=0) and np.all(proba<=1), f"{name}: proba out of range"

    def test_all_regressors_fit_predict(self):
        from src.ml.algorithms import REGRESSOR_REGISTRY
        X,y = self._make_clf_data()
        X_train,X_test = X[:160], X[160:]

        for name, spec in REGRESSOR_REGISTRY.items():
            params = _minimal_reg_params(name)
            model = spec.factory(params, random_state=42)
            model.fit(X_train, y[:160])
            preds = model.predict(X_test)
            assert len(preds) == 40, f"{name}: wrong prediction count"

    def test_all_anomaly_detectors_fit_decision_function(self):
        from src.ml.algorithms import ANOMALY_REGISTRY
        X = self._make_anomaly_data()

        for name, spec in ANOMALY_REGISTRY.items():
            params = _minimal_anomaly_params(name)
            model = spec.factory(params, random_state=42)
            model.fit(X)
            scores = model.decision_function(X[:10])
            preds = model.predict(X[:10])
            assert len(scores)==10, f"{name}: wrong scores count"
            assert set(preds).issubset({-1,1}), f"{name}: unexpected predict values"

    def test_feature_importances_available_for_tree_models(self):
        from src.ml.algorithms import CLASSIFIER_REGISTRY
        X,y = self._make_clf_data()
        tree_algos = ["lgbm","xgboost","catboost","random_forest"]

        for name in tree_algos:
            spec = CLASSIFIER_REGISTRY[name]
            params = _minimal_clf_params(name)
            model = spec.factory(params, random_state=42)
            model.fit(X)
            importances = spec.get_feature_importance(model, X)
            assert importances is not None, f"{name}: features importances should not be None."
            assert len(importances) == X.shape[1], f"{name}: importance length mismatch."

    def test_hgb_feature_importances_are_none(self):
        from src.ml.algorithms import CLASSIFIER_REGISTRY
        X,y = self._make_clf_data()
        spec = CLASSIFIER_REGISTRY["hgb"]
        params = _minimal_clf_params("hgb")
        model = spec.factory(params, random_state=42)
        model.fit(X)
        assert spec.get_feature_importance(model,X) is None


        
def _minimal_clf_params(algo:str)-> dict:
    if algo=="lgbm":
        return {"n_estimators":10, "num_leaves":8,"learning_rate":0.1,"class_weight":"balanced"}
    elif algo=="xgboost":
        return {"n_estimators":10, "max_depth":3,"learning_rate":0.1,"scale_pos_weight":1.0}
    if algo == "catboost":
        return {"iterations": 10, "depth": 3, "learning_rate": 0.1,
                "auto_class_weights": "Balanced"}
    if algo == "random_forest":
        return {"n_estimators": 10, "max_depth": 3, "class_weight": "balanced"}
    if algo == "hgb":
        return {"max_iter": 10, "max_depth": 3, "class_weight": "balanced"}
    return {}

def _minimal_reg_params(algo:str)->dict:
    if algo == "lgbm":
        return {"n_estimators": 10, "num_leaves": 8, "learning_rate": 0.1}
    if algo == "xgboost":
        return {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}
    if algo == "catboost":
        return {"iterations": 10, "depth": 3, "learning_rate": 0.1}
    if algo == "random_forest":
        return {"n_estimators": 10, "max_depth": 3}
    if algo == "hgb":
        return {"max_iter": 10, "max_depth": 3}
    return {}

def _minimal_anomaly_params(algo:str)->dict:
    if algo == "isolation_forest":
        return {"n_estimators": 10, "contamination": 0.05, "max_features": 1.0,
                "bootstrap": False}
    if algo == "lof":
        return {"n_neighbors": 5, "contamination": 0.05}
    return {}

