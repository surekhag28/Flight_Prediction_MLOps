"""
Mlflow model registration.

'registry' child runs under the pipeline parent run and promotes models that passed evaluation.
"""

from __future__ import annotations
from typing import NamedTuple
import mlflow
from src.config.config import get_settings
from src.core.logger import get_logger
from src.utils.mlflow_utils import child_run, setup_mlflow, promote_to_production

logger = get_logger(__name__)
settings = get_settings()
PIPELINE_EXP = settings.mlflow.experiments.pipeline


class RegistrationResult(NamedTuple):
    model_name: str
    run_id: str
    promoted: bool
    version: str | None


def register_delay_if_better(run_id: str, should_promote: bool) -> RegistrationResult:
    model_name = settings.mlflow.model_names.delay
    if should_promote:
        version = promote_to_production(model_name, run_id, artifact_path="model")
        logger.info(f"Delay model promoted with version: {version}")
        return RegistrationResult(model_name, run_id, True, version)
    return RegistrationResult(model_name, run_id, False, None)


def register_congestion_if_better(
    run_id: str, should_promote: bool
) -> RegistrationResult:
    model_name = settings.mlflow.model_names.congestion
    if should_promote:
        version = promote_to_production(model_name, run_id, artifact_path="/model")
        logger.info(f"Congestion model promoted with version: {version}")
        return RegistrationResult(
            model_name=model_name,
            run_id=run_id,
            promoted=should_promote,
            version=version,
        )
    return RegistrationResult(
        model_name=model_name, run_id=run_id, promoted=False, version=None
    )


def register_all(
    delay_run_id: str,
    congestion_run_id: str,
    eval_results: dict[str, bool],
    pipeline_parent_run_id: str,
    pipeline_run_id: str,
) -> dict[str, RegistrationResult]:
    """
    Promotes models that passed evaluation under a 'registry' child run.

    Args:
        delay_run_id:str            Mlflow run ID of the delay training child run
        congestion_run_id:str       Mlflow run ID of the congestion regressor child run
        eval_results:dict           {model_key: should_promote} from evaluate_all
        pipeline_parent_run_id      Mlflow run ID of the pipeline parent run
        pipeline_run_id             Shared DAG timestamp string

    Returns:
        {"delay":RegistrationResult, "congestion":RegistrationResult}
    """

    setup_mlflow()

    with child_run(
        pipeline_parent_run_id,
        "registry",
        PIPELINE_EXP,
        tags={"pipeline_run_id": pipeline_run_id},
    ) as run:
        delay_result = register_delay_if_better(
            delay_run_id, eval_results.get("delay", False)
        )

        congestion_result = register_congestion_if_better(
            congestion_run_id, eval_results.get("congestion", False)
        )

        mlflow.log_params(
            {
                "delay_promoted": delay_result.promoted,
                "delay_version": delay_result.version,
                "congestion_promoted": congestion_result.promoted,
                "congestion_version": congestion_result.version,
            }
        )

        mlflow.log_metric(
            "models_promoted", sum([delay_result.promoted, congestion_result.promoted])
        )

        logger.info(
            f"Registration completed, registry_run_id:{run.info.run_id}, delay_promoted:{delay_result.promoted}, congestion_promoted: {congestion_result.promoted}"
        )

    return {"delay": delay_result, "congestion": congestion_result}
