"""
DAG: Retrain all models

Three triggers (all runs the same full pipeline)
    - Monthly               -- cron "0 2 * *" (1st of each month at 2:00 pm UTC)
    - Drift                 -- triggered by monitor_dag when drift detected
    - Performance drop      -- triggered by monitor_dag when model

Full pipeline run per execution:
    set_pipeline_run_id         -- creates HPO + pipeline parent (delay + congestion + anomaly) run
        - create_proxy_labels
        - [hpo_delay, hpo_congestion, hpo_anomaly] (parallel run for 30 trials each)
        - [train_delay, train_congestion, train_anomaly] (parallel run for full dataset)
        - evaluate_all_models
        - register_all_models
        - finalise_runs
"""

from __future__ import annotations
import sys
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow import DAG
from src.core.logger import get_logger
from dag_utils import get_pipeline_run_id

logger = get_logger(__name__)

sys.path.insert(0, "/opt/src")

DEFAULT_ARGS = {
    "owner": "aviation_ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(hours=4),
}


# ------- Setting pipeline run and creating MLflow parent run id --------- #
def _set_run_id(**context):
    """
    Generate pipeline_run_id and create both MLflow parent run.
    """

    from dag_utils import set_pipeline_run_id
    from src.config.config import get_settings
    from src.utils.mlflow_utils import setup_mlflow, create_parent_run

    pipeline_run_id = set_pipeline_run_id(**context)  # based on datetime
    setup_mlflow()

    settings = get_settings()

    dag_run = context.get("dag_run")
    trigger_reason = "monthly_schedule"

    if dag_run and dag_run.conf:
        trigger_reason = dag_run.conf.get("triggered_by", "monthly_schedule")

    _tags = {
        "pipeline_run_id": pipeline_run_id,
        "dag": "retrain_dag",
        "trigger_reason": trigger_reason,
    }

    # mlflow create parent run

    delay_parent_run_id = create_parent_run(
        settings.mlflow.experiments.delay, f"delay_{pipeline_run_id}", tags=_tags
    )  # for delay risk classifier
    pipeline_parent_run_id = create_parent_run(
        settings.mlflow.experiments.pipeline, f"pipeline_{pipeline_run_id}", tags=_tags
    )

    ti = context["task_instance"]
    ti.xcom_push(key="delay_parent_run_id", value=delay_parent_run_id)
    ti.xcom_push(key="pipeline_parent_run_id", value=pipeline_parent_run_id)

    logger.info(
        f"Retrain dag started: pipeline_run_id:{pipeline_run_id}, trigger_reason: {trigger_reason}, delay_parent_run_id: {delay_parent_run_id}, pipeline_parent_run_id: {pipeline_parent_run_id}"
    )

    return pipeline_run_id


def _create_labels(**context):

    from src.ml.create_labels import run

    run_id = get_pipeline_run_id(**context)
    result = run(pipeline_run_id=run_id)

    logger.info(
        f"Proxy labels created: total rows: {result['total_rows']}, delay_risk_rate: {result['delay_risk_rate']}"
    )


# ------------ HPO tasks --------------#


def _hpo_delay(**context):

    from src.ml.auto_hpo import run_delay_auto_hpo

    pipeline_run_id = get_pipeline_run_id(**context)
    ti = context["task_instance"]

    hpo_parent_id = ti.xcom_pull(task_ids="set_run_id", key="delay_parent_run_id")
    result = run_delay_auto_hpo(pipeline_run_id, hpo_parent_id)
    ti.xcom_push(key="delay_best_params", value=result["params"])
    ti.xcom_push(key="delay_best_algorithm", value=result.get("algorithm", "lgbm"))

    logger.info(
        f"Auto HPO delay complete: best algorithm: {result.get('algorithm')}, best_auc: {result.get('best_auc')}"
    )

    return result


def _train_delay(**context):

    from src.ml.train_delay import run

    pipeline_run_id = get_pipeline_run_id(**context)
    ti = context["task_instance"]
    parent_run_id = ti.xcom_pull(task_ids="set_run_id", key="delay_parent_run_id")
    best_params = ti.xcom_pull(task_ids="hpo_delay", key="delay_best_params")
    algorithm = ti.xcom_pull(task_ids="hpo_delay", key="delay_best_algorithm") or "lgbm"
    result = run(pipeline_run_id, parent_run_id, best_params, algorithm)
    ti.xcom_push(key="delay_run_id", value=result["run_id"])

    logger.info(
        f"Delay model training completed: algorithm: {result['algorithm']}, auc_roc: {round(result['metrics'].get('auc_roc',0),3)}"
    )

    return result


def _evaluate_all(**context):
    from dag_utils import get_pipeline_run_id
    from src.ml.evaluate import evaluate_all

    run_id = get_pipeline_run_id(**context)
    ti = context["task_instance"]
    eval_results = evaluate_all(
        delay_run_id=ti.xcom_pull(task_ids="train_delay", key="delay_run_id"),
        pipeline_parent_run_id=ti.xcom_pull(
            task_ids="set_run_id", key="pipeline_parent_run_id"
        ),
        pipeline_run_id=run_id,
    )

    ti.xcom_push(key="eval_results", value=eval_results)
    logger.info(f"Evaluation complete: eval_results={eval_results}")

    return eval_results


def _register_all(**context):
    from dag_utils import get_pipeline_run_id
    from src.ml.registry import register_all

    run_id = get_pipeline_run_id(**context)
    ti = context["task_instance"]
    results = register_all(
        delay_run_id=ti.xcom_pull(task_ids="train_delay", key="delay_run_id"),
        eval_results=ti.xcom_pull(task_ids="evaluate_all", key="eval_results"),
        pipeline_parent_run_id=ti.xcom_pull(
            task_ids="set_run_id", key="pipeline_parent_run_id"
        ),
        pipeline_run_id=run_id,
    )

    promoted = [k for k, v in results.items() if v.promoted]  # promoted models
    logger.info(f"Registry complete, models promoted={promoted}")
    return {
        k: {"promoted": v.promoted, "version": v.version} for k, v in results.items()
    }


def _finalise_run(**context):
    from src.utils.mlflow_utils import finish_run

    ti = context["task_instance"]
    delay_run_id = ti.xcom_pull(task_ids="set_run_id", key="delay_parent_run_id")

    finish_run(delay_run_id)
    logger.info(f"MLflow parent run finalised:o")


with DAG(
    dag_id="retrain_dag",
    description="Model retrain: monthly + drift-triggered + perf-drop triggered",
    schedule_interval="0 2 1 * *",  # monthly on 1st at 02:00 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["retrain", "hpo", "mlflow", "monthly"],
) as dag:

    set_run_id = PythonOperator(task_id="set_run_id", python_callable=_set_run_id)
    create_labels = PythonOperator(
        task_id="create_labels", python_callable=_create_labels
    )

    hpo_delay = PythonOperator(task_id="hpo_delay", python_callable=_hpo_delay)
    train_delay = PythonOperator(task_id="train_delay", python_callable=_train_delay)
    evaluate_all_task = PythonOperator(
        task_id="evaluate_all", python_callable=_evaluate_all
    )
    register_all_task = PythonOperator(
        task_id="register_all", python_callable=_register_all
    )
    finalise_run = PythonOperator(task_id="finalise_run", python_callable=_finalise_run)

    set_run_id >> create_labels >> hpo_delay
    hpo_delay >> train_delay
    train_delay >> evaluate_all_task >> register_all_task >> finalise_run
