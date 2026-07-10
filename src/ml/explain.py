"""
SHAP Explainability for Flight prediction Model.

Provides:
    - compute_shap_values: TreeExplainer on a trained model, returns (values, base_value)
    - get_top_features: Rank features by mean |SHAP|
    - log_shap_summary: Training-time summary bar chart -> MLflow artifact

Supported model types:
    - LightGBM, XGBost, CatBoost, RandomForest, HistGradientBoosting (TreeExplainer)
    - Falls back gracefully for unsupported models

Binary classifiers: shap_values for class 1 (positive/delay class) are returned
Regressors: shap_values shape (n_samples, n_features)
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import shap

from src.core.logger import get_logger

logger = get_logger(__name__)


def compute_shap_values(model: Any, X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute SHAP values using TreeExplainer.

    Args:
        - model:Any     Trained sklearn-compatible model.
        - X:np.ndarray  Input array of shape (n_samples, n_features)

    Returns:
        (shap_values, base_value) where shap_values has shape (n_samples, n_features).
        For binary classifiers, values correspond to the positive class.

    Raises:
        ValueError: If SHAP computation fails and no fallback is possible.
    """

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)

    # for binary classifiers returns a list [neg_class, pos_class]
    if isinstance(raw, list):
        values = raw[1]
    else:
        values = raw

    # base_value
    ev = explainer.expected_value
    base_value = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)

    return values, base_value


# global explainability
def get_top_features(
    shap_values: np.ndarray, feature_names: list[str], n: int = 5
) -> list[str]:
    """Returns top-n feature names ranked by mean [SHAP] across all samples."""

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = mean_abs.argsort()[::-1][:n]
    return [feature_names[i] for i in top_idx]  # top-n features


# local explainability
def get_feature_contributions(
    shap_values: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    """Map shap values to feature names for single-row prediction."""

    return {name: round(float(val), 6) for name, val in zip(feature_names, shap_values)}


def log_shap_summary(
    model: Any, X_sample: np.ndarray, feature_names: list[str], artifact_dir: Path
) -> None:
    """
    Compute SHAP values on X_sample, create a mean |SHAP| bar chat and log it as mlflow artifact.

    Called at training time after model.fit() method. Silent on any SHAP error so training never fails
    due to explainability code.

    Args:
        model:Any       Trained model
        X_sample:np.ndarray     Training data sample
        feature_names:list[str]     Feature names matching columns of X_sample
        artifact_dir:Path           Local directory to write the plot file before upload.
    """
    try:
        import matplotlib
        import mlflow

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Cap sample size for speed
        if len(X_sample) > 500:
            rng = np.random.default_rng(42)  # random number generator object with seed
            idx = rng.choice(len(X_sample), size=500, replace=False)
            X_sample = X_sample[idx]

        shap_values, base_value = compute_shap_values(model, X_sample)
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = mean_abs.argsort()[
            ::-1
        ]  # indexes in descending order of shap values for features

        fig, ax = plt.subplots(figsize=(10, 12))
        ax.barh(
            [feature_names[idx] for idx in order],
            mean_abs[order[::-1]],
            color="#2196F3",
        )

        ax.set_title(f"Feature Importance (SHAP) values\nbase_value: {base_value:.4f}")
        ax.set_xlabel("Mean | SHAP values")
        ax.tick_params(axis="y", labelsize=9)
        fig.tight_layout()

        artifact_dir.mkdir(parents=True, exist_ok=True)
        plot_path = artifact_dir / "shap_summary.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)

        mlflow.log_artifact(str(plot_path), "explainability")

        # Also log top-10 mean [SHAP] as a JSON dict
        top10 = {feature_names[i]: round(float(mean_abs[i]), 4) for i in order[:10]}
        mlflow.log_dict(top10, "explainability/shap_mean_abs.json")

        logger.info(
            f"SHAP summary logged: features: {len(feature_names)}, base_value: {round(base_value,4)}"
        )
    except Exception as e:
        logger.warning("SHAP summary statistics skipped [non-fatal]")  # silent failure
