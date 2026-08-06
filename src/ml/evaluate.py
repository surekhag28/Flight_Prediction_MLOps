"""
Model evaluation: compare candidate model run metrics against the production.

'evaluation' runs under pipeline parent run and logs comparison results
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from src.config.config import get_settings
from src.utils.exceptions import MlflowError
from src.core.logger import get_logger
from src.utils.mlflow_utils import child_run, setup_mlflow

logger = get_logger(__name__)
settings = get_settings()

PIPELINE_EXP = settings.mlflow.experiments.pipeline


def _get_run_metrics(run_id: str):
    try:
        return MlflowClient().get_run(run_id).data.metrics
    except Exception as e:
        raise MlflowError(
            f"Failed to fetch mlflow run metrics for run :{run_id}"
        ) from e


def _get_production_run_id(model_name: str):
    try:
        versions = MlflowClient().get_latest_versions(model_name, stages=["Production"])
        return versions[0].run_id
    except Exception as e:
        return None


def evaluate_delay(candidate_run_id: str):
    cfg = settings.training.delay
    metrics = _get_run_metrics(candidate_run_id)
    candidate_auc = metrics.get("auc_roc", 0.0)
    candidate_f1 = metrics.get("f1", 0.0)

    # gate quality check
    if candidate_auc < cfg.min_auc and candidate_auc < cfg.min_f1:
        logger.warning(
            f"Delay candidate model failed to pass gate quality check, auc={round(candidate_auc,3)}, min_auc={cfg.min_auc}, f1={round(candidate_f1,3)}, min_f1={cfg.min_f1}"
        )
        return False

    prod_run_id = _get_production_run_id(settings.mlflow.model_names.delay)
    if prod_run_id is None:
        logger.info("No production model deployed - promoting candidate model")
        return True

    prod_auc = _get_run_metrics(prod_run_id).get("auc_roc", 0.0)
    better = candidate_auc > prod_auc
    logger.info(
        f"Delay eval: candidate_auc={round(candidate_auc,3)}, prod_auc={round(prod_auc,3)}, decision={'PROMOTE' if better else 'KEEP'}"
    )

    return better


def evaluate_all(
    delay_run_id: str, pipeline_parent_run_id: str, pipeline_run_id: str
) -> dict[str, bool]:
    """Evaluate candidate models under one 'evaluation' child run.

    Logs per-model comparison results in mlflow and returns a dict of {model_key: should_promote}.
    """

    setup_mlflow()

    with child_run(
        pipeline_parent_run_id,
        "evaluation",
        PIPELINE_EXP,
        tags={"pipeline_run_id": pipeline_run_id},
    ) as run:
        delay_promote = evaluate_delay(delay_run_id)

        results = {"delay": delay_promote}

        delay_metrics = _get_run_metrics(delay_run_id)

        mlflow.log_metrics(
            {
                "delay_candidate_auc": delay_metrics.get("auc_roc", 0.0),
                "delay_candidate_f1": delay_metrics.get("f1", 0.0),
            }
        )

        mlflow.log_params(
            {"delay_promote": str(delay_promote), "delay_run_id": delay_run_id}
        )

        mlflow.log_dict(results, "promotion_decision.json")

        logger.info(
            f"Evaluation complete, evaluation_run_id: {run.info.run_id}, delay={delay_promote}"
        )

    return results
