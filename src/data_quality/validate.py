from __future__ import annotations
from dataclasses import dataclass, field

from typing import Any
import pandas as pd
import great_expectations as gx

from src.core.logger import get_logger
from src.data_quality.expectations import add_bronze_expectations

logger = get_logger(__name__)


@dataclass
class ValidationbResult:
    suite_name: str
    passed: bool
    evaluated_expectations: int
    successful_expecatations: int
    failed_expectations: list[dict[str, Any]]
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.evaluated_expectations == 0:
            return 1.0
        return self.successful_expecatations / self.evaluated_expectations

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "passd": self.passed,
            "evaluated_expectations": self.evaluated_expectations,
            "successful_expecatations": self.successful_expecatations,
            "failed_expectations": self.failed_expectations,
            "success_rate": round(self.success_rate, 4),
            "statistics": self.statistics,
        }


def _run_suite(df: pd.DataFrame, suite_name: str, add_expectations_fn):

    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas(f"pandas_{suite_name}")
    asset = datasource.add_dataframe_asset(suite_name)
    batch_def = asset.add_batch_definition_whole_dataframe(f"{suite_name}_bd")

    suite = gx.ExpectationSuite(name=suite_name)

    for expectation in add_expectations_fn():
        suite.add_expectation(expectation)

    context.suites.add(suite)

    validation = gx.ValidationDefinition(name=suite_name, data=batch_def, suite=suite)

    result = validation.run(batch_parameters={"dataframe": df})

    passed = bool(result["success"])
    stats = result.statistics or {}
    evaluated = stats.get("evaluated_expectations", 0)
    successful = stats.get("successful_expectations", 0)

    failed = []
    for er in result.results:
        if not er.success:
            cfg = er.expectation_config
            expectation_type = getattr(cfg, "type", None) or getattr(
                cfg, "expectation_type", "unknown"
            )

            failed.append(
                {
                    "expectation_type": expectation_type,
                    "column": cfg.kwargs.get("column") or cfg.kwargs.get("column_set"),
                    "kwargs": cfg.kwargs,
                    "result": er.result,
                }
            )

    validation_result = ValidationbResult(
        suite_name=suite_name,
        passed=passed,
        evaluated_expectations=evaluated,
        successful_expecatations=successful,
        failed_expectations=failed,
        statistics={
            "row_count": len(df),
            "column_count": len(df.columns),
            "success_percent": round(stats.get("success_percent", 0.0), 2),
        },
    )
    _log_result(validation_result)
    return validation_result


def _log_result(result: ValidationbResult) -> None:
    log = logger.info if result.passed else logger.warning
    log(
        f"Data validation checks: {'PASSED' if result.passed else 'FAILED'}",
        extra={
            "suite": result.suite_name,
            "evaluated": result.evaluated_expectations,
            "passed_count": result.successful_expecatations,
            "failed_count": result.failed_expectations,
            "success_rate": result.success_rate,
        },
    )

    for f in result.failed_expectations:
        log(f"Failed expectations: {f['expectation_type'], f.get('column','N/A')}")


def validate_bronze(df: pd.DataFrame):
    # print(df.head(2))
    return _run_suite(df, "bronze_suite", add_bronze_expectations)


# if __name__ == "__main__":
#     import boto3
#     import json

#     s3 = boto3.client(
#         "s3",
#         endpoint_url="http://localhost:9002",
#         aws_access_key_id="minioadmin",
#         aws_secret_access_key="minioadmin",
#     )

#     obj = s3.get_object(
#         Bucket="aviation-lake",
#         Key="bronze/opensky/year=2026/month=06/day=02/batch_20260602_045828.json",
#     )
#     import pandas as pd

#     content = json.loads(obj["Body"].read().decode("utf-8"))["states"]
#     df = pd.DataFrame(content)
#     print(df.head())
#     validate_bronze(df)
