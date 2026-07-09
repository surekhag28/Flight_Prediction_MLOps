"""
Multi-algorithm AutoML HPO - best model selection across 5 different algorithms and parameters.

Design:
    - Single optuna study per task `algorithm` as categorical hyperparameter.
    - Optuna's sampler learns which algorithm peforms best and allocates proportionally more trials
        to top contenders (exploration vs exploitation)
    - Each trial is nested MLflow run tagged  with the algorithm name.
    - Returns {"algorithm":str, "params":dict, "best_score":float, "hpo_run_id":str}


MLflow hierarchy (per-level experiments):
    Aviation MLOps/Delay Risk
        delay_<timestamp>   (parent)
            delay_auto      (HPO child)
                trial_0     (algorithm=lgbm, cv_auc=0.81)
                trial_1     (algorithm=xgboost, cv_auc=0.88) <-- best performing model
                trial_5     (algorithm=xgboost, cv_auc=0.85)
        training_delay      (training child)

    Aviation MLOps/Congestion   ->  congestion_auto + training_congestion
    Aviation MLOps/Anomaly Detection    ->  anomaly_auto + training_anomaly
"""

from __future__ import annotations
import os
import json
from datetime import UTC, datetime
from typing import Any
import mlflow
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config.config import get_settings
from src.ml.algorithms import CLASSIFIER_REGISTRY, get_classifier_algorithms
from src.core.logger import get_logger
from src.utils.mlflow_utils import child_run, setup_mlflow
from src.utils.exceptions import InsufficientDataError, HPOError

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

settings = get_settings()

DELAY_EXP = settings.mlflow.experiments.delay
N_TRIALS = settings.hpo.n_trials
HPO_SAMPLE_ROWS = settings.hpo.sample_rows
N_CV_FOLDS = settings.hpo.n_cv_folds
RANDOM_STATE = settings.training.delay.random_state

# classifier AutoHPO (delay risk)


def auto_tune_delay(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    pipeline_run_id: str,
    hpo_parent_run_id: str,
    n_trials: int = N_TRIALS,
    sample_rows: int = HPO_SAMPLE_ROWS,
    algorithms: list[str] | None = None,
) -> dict[str, Any]:
    """
    Multi-algorithm HPO for the delay risk classifier

    Returns {"algorithm":str, "params":dict, "best_auc":float, "hpo_run_id":str}
    """

    if algorithms is None:
        algorithms = get_classifier_algorithms()

    if len(X) > sample_rows:
        idx = np.random.default_rng(RANDOM_STATE).choice(
            len(X), sample_rows, replace=False
        )
        X_s, y_s = X[idx], y[idx]
    else:
        X_s, y_s = X, y

    logger.info(
        f"Auto HPO delay started: samples: {len(X_s)}, trials: {n_trials}, algorithms: {algorithms}"
    )

    try:
        with child_run(
            hpo_parent_run_id,
            "delay_auto",
            DELAY_EXP,
            tags={
                "model_type": "classifier",
                "pipeline_run_id": pipeline_run_id,
                "hpo_mode": "multi_algorithm",
                "algorithms": ",".join(algorithms),
            },
        ) as run:

            # child run id
            hpo_run_id = run.info.run_id

            def objective(trial: optuna.Trial) -> float:

                algo_name = trial.suggest_categorical("algorithm", algorithms)
                spec = CLASSIFIER_REGISTRY[algo_name]  # AlgorithmSpec object
                params = spec.search_space(trial, algo_name)

                n_pos = int(sum(y_s))  # total positive samples
                n_splits = min(n_pos, max(2, n_pos))  #
                cv = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
                )

                aucs = []
                for train_idx, val_idx in cv.split(X_s, y_s):
                    if len(
                        np.unique(y_s[val_idx]) < 2
                    ):  # ignore the fold containing only one class
                        continue
                    if len(np.unique(train_idx) < 2):
                        continue

                    m = spec.factory(params, RANDOM_STATE)
                    m.fit(X_s[train_idx], y_s[train_idx])

                    prob = m.predict_proba(X_s[val_idx])[:, 1]
                    auc_val = roc_auc_score(y_s[val_idx], prob)
                    if np.isnan(auc_val):
                        continue
                    aucs.append(auc_val)

                if not aucs:
                    return 0.0
                mean_auc = float(np.mean(aucs))  # avg auc across folds for single trial

                with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                    mlflow.log_params(
                        {
                            "algorithm": algo_name,
                            **{k: str(v) for k, v in params.items()},
                        }
                    )
                    mlflow.log_metric("cv_auc_roc", mean_auc)
                    mlflow.log_metric("trial_number", trial.number)
                    mlflow.set_tag("algorithm", algo_name)

                return mean_auc

            sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
            study = optuna.create_study(
                direction="maximize", sampler=sampler, pruner=pruner
            )

            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            best_trial = study.best_trial
            best_algo = best_trial.params["algorithm"]
            best_params = {
                k.removeprefix(f"{best_algo}_"): v
                for k, v in best_trial.params.items()
                if k != "algorithm"
            }
            best_auc = study.best_value

            algo_stats = _summarise_algo_trials(study, algorithms)
            mlflow.log_dict(algo_stats, "algorithms_comparison.json")
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_cv_auc_roc", best_auc)
            mlflow.log_metric("n_trials_completed", len(study.trials))
            mlflow.set_tag("best_algorithm", best_algo)

            logger.info(
                f"Auto HPO delay done: best_model= {best_algo}, best_auc = {round(best_auc,4)}"
            )

        return {
            "algorithm": best_algo,
            "params": best_params,
            "best_auc": best_auc,
            "hpo_run_id": hpo_run_id,
        }
    except Exception as e:
        raise HPOError("Auto HPO failed for delay risk classifier model") from e


def _summarise_algo_trials(
    study: optuna.Study,
    algorithms: list[str],
    metric_name: str = "auc",
    default_empty: float = 0.0,
) -> dict:
    """Return per-algorithm best/mean score and HPO trials."""

    result = {}
    for algo in algorithms:
        algo_trials = [
            t
            for t in study.trials
            if t.params.get("algorithm") == algo and t.value is not None
        ]

        if not algo_trials:
            result[algo] = {
                "best_{metric_name}": default_empty,
                "mean_{metric_name}": default_empty,
            }
            continue

        values = [t.value for t in algo_trials]
        result[algo] = {
            f"best_{metric_name}": round(max(values), 4),
            f"mean_{metric_name}": round(float(np.mean(values)), 4),
            "n_trials": len(algo_trials),
        }

        return result


def run_delay_auto_hpo(pipeline_run_id: str, hpo_parent_run_id: str) -> dict:
    """Loads data from MinIO and runs multi-algorithm delay HPO, orchestrated by Airflow."""

    from src.ml.training_utils import load_parquet_data, get_fs

    setup_mlflow()
    fs = get_fs()
    cfg = settings.training.delay
    bucket = settings.miniosettings.bucket

    try:
        df = load_parquet_data(f"{bucket}/gold/labels", fs, "delay label")
    except (FileNotFoundError, OSError, InsufficientDataError) as e:
        logger.error(
            f"Auto HPO delay: no labelled data found, falling back to default lgbm config"
        )
        return {"algorithm": "lgbm", "params": cfg.lgbm_params, "skipped": True}

    if len(df) < settings.hpo.min_sample_rows_delay:
        logger.warning(f"Auto HPO delay: insufficient rows, falling back to default")
        return {"algorithm": "lgbm", "params": cfg.lgbm_params, "skipped": True}

    feature_cols = [col for col in cfg.feature_columns if col in df.columns]
    df = df[feature_cols + [cfg.target_column]].dropna()

    X = df[feature_cols].values
    y = df[cfg.target_column].values
    return auto_tune_delay(X, y, feature_cols, pipeline_run_id, hpo_parent_run_id)


if __name__ == "__main__":

    from src.utils.mlflow_utils import create_parent_run

    setup_mlflow()

    pipeline_run_id = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    model = "delay"
    exp = {"delay": DELAY_EXP}.get(model, DELAY_EXP)
    hpo_parent = create_parent_run(
        exp, f"hpo_delay_{pipeline_run_id}", tags={"pipeline_run_id": pipeline_run_id}
    )
    result = run_delay_auto_hpo(pipeline_run_id, hpo_parent)

    print(json.dumps({k: str(v) for k, v in result.items()}, indent=2))
