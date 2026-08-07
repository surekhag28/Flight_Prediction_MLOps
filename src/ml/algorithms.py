"""
Algorithm registry for AutoML model selection.

Defines AlgorithmSpec per task type. (classifier/regressor/anomaly_detector).
Each spec provides:
    - factory(params, random_state) -> unfitted model instance
    - search_space(trial,prefix) -> Optuna params dict
    - get_feature_importance(model, X) -> np.ndarray | None
    - supports_feature_importance_flag

Used by auto_hp.py (HPO) and train_**.py (final training)
"""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import numpy as np
import optuna


@dataclass
class AlgorithmSpec:
    name: str
    factory: Callable[..., Any]  # (params, random_state) -> model
    search_space: Callable[[optuna.Trial, str], dict]  # (trial, prefix) -> params
    get_feature_importance: Callable[[Any, np.ndarray | None], np.ndarray | None]
    supports_feature_importance_flag: bool = True
    algorithm_type: str = "tree"  #  "tree" | "ensemble" | "anomaly"


# feature importance helpers


def _lgbm_feature_importance(model: Any, X: np.ndarray | None) -> np.ndarray | None:
    return np.array(model.feature_importances_, dtype=float)


def _xgb_feature_importance(model: Any, X: np.ndarray | None) -> np.ndarray | None:
    return np.array(model.feature_importances_, dtype=float)


def _rf_feature_importance(model: Any, X: np.ndarray | None) -> np.ndarray | None:
    return np.array(model.feature_importances_, dtype=float)


def _catboost_feature_importance(model: Any, X: np.ndarray | None) -> np.ndarray | None:
    try:
        return np.array(model.get_feature_importance(), dtype=float)
    except Exception as e:
        return None


def _hgb_feature_importance(model: Any, X: np.ndarray | None) -> np.ndarray | None:
    # HistGradientBoosting has feature_importances_ via permutations, use None for speed.
    return None


def _no_feature_importance(model: Any, X: np.ndarray | None) -> np.ndarray | None:
    return None


# Classfier Algorithms


def _lgbm_clf_factory(params: dict, random_state: int) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**params, random_state=random_state, verbose=1, n_jobs=-1)


def _lgbm_clf_space(trial: optuna.Trial, prefix: str) -> dict:
    p = f"{prefix}_"
    return {
        "n_estimators": trial.suggest_int(f"{p}n_estimators", 100, 800),
        "learning_rate": trial.suggest_float(f"{p}learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int(f"{p}num_leaves", 20, 150),
        "max_depth": trial.suggest_int(f"{p}max_depth", 3, 12),
        "min_child_samples": trial.suggest_int(f"{p}min_child_samples", 10, 100),
        "subsample": trial.suggest_float(f"{p}subsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float(f"{p}reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float(f"{p}reg_lambda", 1e-4, 10.0, log=True),
        "class_weight": "balanced",
    }


def _xgb_clf_factory(params: dict, random_state: int) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        **params,
        random_state=random_state,
        verbose=1,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
    )


def _xgb_clf_space(trial: optuna.Trial, prefix: str) -> dict:
    p = f"{prefix}_"
    return {
        "n_estimators": trial.suggest_int(f"{p}n_estimators", 100, 800),
        "learning_rate": trial.suggest_float(f"{p}learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int(f"{p}max_depth", 3, 10),
        "min_child_weight": trial.suggest_int(f"{p}min_child_weight", 1, 20),
        "subsample": trial.suggest_float(f"{p}subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float(f"{p}colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float(f"{p}gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float(f"{p}reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float(f"{p}reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float(f"{p}scale_pos_weight", 1.0, 10.0),
    }


def _catboost_clf_factory(params: dict, random_state: int) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(**params, random_seed=random_state, verbose=1)


def _catboost_clf_space(trial: optuna.Trial, prefix: str) -> dict:
    p = f"{prefix}_"
    return {
        "iterations": trial.suggest_int(f"{p}iterations", 100, 600),
        "learning_rate": trial.suggest_float(f"{p}learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int(f"{p}depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float(f"{p}l2_leaf_reg", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int(f"{p}border_count", 32, 255),
        "bagging_temperature": trial.suggest_float(f"{p}bagging_temperature", 0.0, 1.0),
        "auto_class_weights": "Balanced",
    }


def _rf_clf_factory(params: dict, random_state: int) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)


def _rf_clf_space(trial: optuna.Trial, prefix: str) -> dict:
    p = f"{prefix}_"
    return {
        "n_estimators": trial.suggest_int(f"{p}n_estimators", 100, 500),
        "max_depth": trial.suggest_int(f"{p}max_depth", 3, 20),
        "min_samples_split": trial.suggest_int(f"{p}min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int(f"{p}min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            f"{p}max_features", ["sqrt", "log2", 0.5]
        ),
        "class_weight": "balanced",
    }


def _hgb_clf_factory(params: dict, random_state: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(**params, random_state=random_state)


def _hgb_clf_space(trial: optuna.Trial, prefix: str) -> dict:
    p = f"{prefix}_"
    return {
        "max_iter": trial.suggest_int(f"{p}max_iter", 100, 500),
        "learning_rate": trial.suggest_float(f"{p}learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int(f"{p}max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int(f"{p}min_samples_leaf", 10, 100),
        "l2_regularization": trial.suggest_float(
            f"{p}l2_regularization", 1e-4, 10.0, log=True
        ),
        "max_bins": trial.suggest_int(f"{p}max_bins", 64, 255),
        "class_weight": "balanced",
    }


CLASSIFIER_REGISTRY: dict[str, AlgorithmSpec] = {
    "lgbm": AlgorithmSpec(
        name="lgbm",
        factory=_lgbm_clf_factory,
        search_space=_lgbm_clf_space,
        get_feature_importance=_lgbm_feature_importance,
    ),
    "xgboost": AlgorithmSpec(
        name="xgb",
        factory=_xgb_clf_factory,
        search_space=_xgb_clf_space,
        get_feature_importance=_xgb_feature_importance,
    ),
    "catboost": AlgorithmSpec(
        name="catboost",
        factory=_catboost_clf_factory,
        search_space=_catboost_clf_space,
        get_feature_importance=_catboost_feature_importance,
    ),
    "random_forest": AlgorithmSpec(
        name="random_forest",
        factory=_rf_clf_factory,
        search_space=_rf_clf_space,
        get_feature_importance=_rf_feature_importance,
    ),
    "hgb": AlgorithmSpec(
        name="hgb",
        factory=_hgb_clf_factory,
        search_space=_hgb_clf_space,
        get_feature_importance=_hgb_feature_importance,
    ),
}


# Regressor Algorithms


def _lgbm_reg_factory(params: dict, random_state: int) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**params, random_state=random_state, verbose=-1, n_jobs=-1)


def _lgbm_reg_space(trial: optuna.Trial, prefix: str) -> dict:
    p = f"{prefix}_"
    return {
        "n_estimators": trial.suggest_int(f"{p}n_estimators", 100, 800),
        "learning_rate": trial.suggest_float(f"{p}learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int(f"{p}num_leaves", 20, 150),
        "max_depth": trial.suggest_int(f"{p}max_depth", 3, 12),
        "min_child_samples": trial.suggest_int(f"{p}min_child_samples", 10, 100),
        "subsample": trial.suggest_float(f"{p}subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float(f"{p}colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float(f"{p}reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float(f"{p}reg_lambda", 1e-4, 10.0, log=True),
    }


def _xgb_reg_factory(params: dict, random_state: int) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(
        **params, random_state=random_state, verbosity=0, tree_method="hist", n_jobs=-1
    )


def _xgb_reg_space(trial: optuna.Trial, prefix: str) -> dict:
    p = f"{prefix}_"
    return {
        "n_estimators": trial.suggest_int(f"{p}n_estimators", 100, 800),
        "learning_rate": trial.suggest_float(f"{p}learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int(f"{p}max_depth", 3, 10),
        "min_child_weight": trial.suggest_int(f"{p}min_child_weight", 1, 20),
        "subsample": trial.suggest_float(f"{p}subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float(f"{p}colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float(f"{p}gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float(f"{p}reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float(f"{p}reg_lambda", 1e-4, 10.0, log=True),
    }


def _catboost_reg_factory(params: dict, random_state: int) -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**params, random_state=random_state, verbose=0)


def _catboost_reg_space(trial: optuna.Trial, prefix: str) -> dict:
    p = f"{prefix}_"
    return {
        "iterations": trial.suggest_int(f"{p}iterations", 100, 600),
        "learning_rate": trial.suggest_float(f"{p}learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int(f"{p}depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float(f"{p}l2_leaf_reg", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int(f"{p}border_count", 32, 255),
    }


def _rf_reg_factory(params: dict, random_state: int) -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)


def _rf_reg_space(trial: optuna.Trial, prefix: str) -> dict:
    p = f"{prefix}_"
    return {
        "n_estimators": trial.suggest_int(f"{p}n_estimators", 100, 500),
        "max_depth": trial.suggest_int(f"{p}max_depth", 3, 20),
        "min_samples_split": trial.suggest_int(f"{p}min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int(f"{p}min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            f"{p}max_features", ["sqrt", "log2", 0.5]
        ),
    }


def _hgb_reg_factory(params: dict, random_state: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(**params, random_state=random_state)


def _hgb_reg_space(trial: optuna.Trial, prefix: str) -> dict:
    p = f"{prefix}_"
    return {
        "max_iter": trial.suggest_int(f"{p}max_iter", 100, 500),
        "learning_rate": trial.suggest_float(f"{p}learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int(f"{p}max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int(f"{p}min_samples_leaf", 10, 100),
        "l2_regularization": trial.suggest_float(
            f"{p}l2_regularization", 1e-4, 10.0, log=True
        ),
        "max_bins": trial.suggest_int(f"{p}max_bins", 64, 255),
    }


REGRESSOR_REGISTRY: dict[str, AlgorithmSpec] = {
    "lgbm": AlgorithmSpec(
        name="lgbm",
        factory=_lgbm_reg_factory,
        search_space=_lgbm_reg_space,
        get_feature_importance=_lgbm_feature_importance,
    ),
    "xgboost": AlgorithmSpec(
        name="xgboost",
        factory=_xgb_reg_factory,
        search_space=_xgb_reg_space,
        get_feature_importance=_xgb_feature_importance,
    ),
    "catboost": AlgorithmSpec(
        name="catboost",
        factory=_catboost_reg_factory,
        search_space=_catboost_reg_space,
        get_feature_importance=_catboost_feature_importance,
    ),
    "random_forest": AlgorithmSpec(
        name="random_forest",
        factory=_rf_reg_factory,
        search_space=_rf_reg_space,
        get_feature_importance=_rf_feature_importance,
    ),
    "hgb": AlgorithmSpec(
        name="hgb",
        factory=_hgb_reg_factory,
        search_space=_hgb_reg_space,
        get_feature_importance=_hgb_feature_importance,
    ),
}

# default algorithms


def get_classifier_algorithms() -> list[str]:
    """Returns default algorithm names to compete in classifier HPO"""
    return ["lgbm", "xgboost", "catboost", "random_forest", "hgb"]


def get_regressor_algorithms() -> list[str]:
    """Returns deafult algorithm names to compete in regressor HPO"""
    return ["lgbm", "xgboost", "catboost", "random_forest", "hgb"]
