"""
Great expectations checkpoint runner for the flight prediction data pipeline.

Functions:
validate_bronze(df) -> data quality checks of the raw opensky api ingested data -> ValidationResult

ValidationResult carries passed/failed status for the expectations, per-expectation dtails, and
summary statistics that are logged and published as HTML report to MinIO bucket.

"""

from __future__ import annotations
from dataclasses import dataclass, field

from typing import Any
import pandas as pd
import great_expectations as gx

from src.core.logger import get_logger
from src.data_quality.expectations import add_bronze_expectations

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    suite_name: str
    passed: bool
    evaluated_expectations: int
    successful_expectations: int
    failed_expectations: list[dict[str, Any]]
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.evaluated_expectations == 0:
            return 1.0
        return self.successful_expectations / self.evaluated_expectations

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "passed": self.passed,
            "evaluated_expectations": self.evaluated_expectations,
            "successful_expectations": self.successful_expectations,
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

    validation_result = ValidationResult(
        suite_name=suite_name,
        passed=passed,
        evaluated_expectations=evaluated,
        successful_expectations=successful,
        failed_expectations=failed,
        statistics={
            "row_count": len(df),
            "column_count": len(df.columns),
            "success_percent": round(stats.get("success_percent", 0.0), 2),
        },
    )
    _log_result(validation_result)
    return validation_result


def _log_result(result: ValidationResult) -> None:
    log = logger.info if result.passed else logger.warning
    log(
        f"Data validation checks: {'PASSED' if result.passed else 'FAILED'}",
        extra={
            "suite": result.suite_name,
            "evaluated": result.evaluated_expectations,
            "passed_count": result.successful_expectations,
            "failed_count": result.failed_expectations,
            "success_rate": result.success_rate,
        },
    )

    for f in result.failed_expectations:
        log(f"Failed expectations: {f['expectation_type'], f.get('column','N/A')}")


def validate_bronze(df: pd.DataFrame):
    # print(df.head(2))
    return _run_suite(df, "bronze_suite", add_bronze_expectations)


# ----- Validation Result publisher


def publish_data_docs(results: list[ValidationResult], run_id: str) -> str | None:
    """
    Serialize validation results as HTML report and upload to MinIO.

    Returns MinIO url for uploaded report, or None if upload fails.
    """

    try:
        import boto3
        from botocore.config import Config
        from src.config.config import get_settings

        html = _build_html_report(results, run_id)
        key = f"data-docs/{run_id}/index.html"

        settings = get_settings()

        s3 = boto3.client(
            "s3",
            endpoint_url=settings.miniosettings.endpoint,
            aws_access_key_id=settings.miniosettings.access_key,
            aws_secret_access_key=settings.miniosettings.secret_key,
            config=Config(signature_version="s3v4"),
        )

        s3.put_object(
            Bucket=settings.miniosettings.bucket,
            Key=key,
            Body=html.encode("utf-8"),
            ContentType="text/html",
        )

        url = f"{settings.miniosettings.endpoint}/{settings.miniosettings.bucket}/{key}"
        logger.info(
            f"Data quality checks doc published to MinIO: {url}, {run_id}",
            extra={"url": url, "run_id": run_id},
        )
        return url

    except Exception as e:
        logger.warning(f"Data quality checks doc failed to publish to MinIO: str{e}")
        return None


def _build_html_report(results: list[ValidationResult], run_id: str) -> str:
    """Build a minimal HTML summary of all validation results."""
    rows = []
    for r in results:
        status_icon = "OK" if r.passed else "FAIL"
        failed_html = ""
        if r.failed_expectations:
            items = "".join(
                f"<li><code>{f['expectation_type']}</code> "
                f"on <b>{f.get('column', 'table')}</b></li>"
                for f in r.failed_expectations
            )
            failed_html = f"<ul>{items}</ul>"
        rows.append(f"""
            <tr>
                <td>[{status_icon}] {r.suite_name}</td>
                <td>{r.evaluated_expectations}</td>
                <td>{r.successful_expectations}</td>
                <td>{len(r.failed_expectations)}</td>
                <td>{r.success_rate:.1%}</td>
                <td>{failed_html or "None"}</td>
            </tr>""")

    table_rows = "\n".join(rows)
    overall = all(r.passed for r in results)
    banner_color = "#2ecc71" if overall else "#e74c3c"
    banner_text = "ALL SUITES PASSED" if overall else "VALIDATION FAILURES DETECTED"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Data Quality Report - {run_id}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; }}
    h1 {{ color: #2c3e50; }}
    .banner {{ background: {banner_color}; color: white; padding: 1rem; border-radius: 4px; margin-bottom: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    code {{ background: #eee; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>Aviation Data Quality Report</h1>
  <p><b>Run ID:</b> {run_id}</p>
  <div class="banner">{banner_text}</div>
  <table>
    <tr>
      <th>Suite</th><th>Evaluated</th><th>Passed</th>
      <th>Failed</th><th>Success Rate</th><th>Failed Expectations</th>
    </tr>
    {table_rows}
  </table>
  <p style="color:#888; font-size:0.85em; margin-top:2rem;">
    Generated by OpenSky MLOps - great-expectations
  </p>
</body>
</html>"""


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
