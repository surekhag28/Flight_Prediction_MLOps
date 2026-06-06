from __future__ import annotations
import os
from datetime import UTC, datetime


def set_pipeline_run_id(**context):
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}

    if "pipeline_run_id" in conf:
        run_id = conf["pipeline_run_id"]
    else:
        exec_date = context.get("execution_date") or datetime.now(tz=UTC)
        run_id = exec_date.strftime("%Y%m%d_%H%M%S")

    os.environ["PIPELINE_RUN_ID"] = run_id
    context["task_instance"].xcom_push(key="pipeline_run_id", value=run_id)
    return run_id


def get_pipeline_run_id(**context) -> str:
    """Retrieve the PIPELINE_RUN_ID set by set_pipeline_run_id"""

    run_id = context["task_instance"].xcom_pull(
        task_ids="set_pipeline_run_id", key="pipeline_run_id"
    )

    if run_id:
        os.environ["PIPELINE_RUN_ID"] = run_id

    return run_id or os.getenv("PIPELINE_RUN_ID", "UNKNOWN")
