"""Weekly lead conversion training.

extract_crm_snapshot → build_features → train_candidate_models → evaluate_models
→ register_best_candidate → assign_challenger_alias

Every task runs in the isolated RevIntel environment (REVINTEL_PYTHON) and calls one step
of `ml.pipeline.lead_conversion`, which reuses the existing feature, training, evaluation
and MLflow code. Tasks exchange only small metadata through XCom (relative paths, hashes,
MLflow IDs, versions, validation metrics); data and models stay in the run's directory
under REVINTEL_PIPELINE_ROOT.

The best candidate is chosen on validation metrics only, registered once per DAG run and
marked `challenger`. Nothing here touches `champion`: promotion is a manual decision
(`python -m ml.training.promote --version <N>`).
"""

import os
from datetime import timedelta
from typing import Any

import pendulum
from airflow.sdk import dag, task

REVINTEL_PYTHON = os.environ.get("REVINTEL_PYTHON", "/opt/revintel-venv/bin/python")
# One key per DAG run: names the run's artifact directory and tags its MLflow records, so
# retries of the same run reuse them instead of duplicating them.
RUN_KEY = "lead_conversion_training/{{ run_id }}"


@dag(
    dag_id="lead_conversion_training",
    schedule="0 3 * * 1",  # weekly, Mondays 03:00 UTC
    start_date=pendulum.datetime(2026, 11, 2, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    tags=["revintel", "ml", "lead-conversion"],
    doc_md=__doc__,
)
def lead_conversion_training() -> None:
    # Every task runs one pipeline step in the RevIntel virtualenv and returns one small dict
    # as its XCom (multiple_outputs=False: not one XCom per key).
    @task.external_python(python=REVINTEL_PYTHON, expect_airflow=False, multiple_outputs=False)
    def extract_crm_snapshot(run_key: str) -> dict[str, Any]:
        from ml.pipeline.lead_conversion import extract_crm_snapshot as step

        return step(run_key)

    @task.external_python(python=REVINTEL_PYTHON, expect_airflow=False, multiple_outputs=False)
    def build_features(run_key: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        from ml.pipeline.lead_conversion import build_features as step

        return step(run_key)

    @task.external_python(python=REVINTEL_PYTHON, expect_airflow=False, multiple_outputs=False)
    def train_candidate_models(run_key: str, features: dict[str, Any]) -> dict[str, Any]:
        from ml.pipeline.lead_conversion import train_candidate_models as step

        return step(run_key)

    @task.external_python(python=REVINTEL_PYTHON, expect_airflow=False, multiple_outputs=False)
    def evaluate_models(run_key: str, models: dict[str, Any]) -> dict[str, Any]:
        from ml.pipeline.lead_conversion import evaluate_models as step

        return step(run_key)

    @task.external_python(python=REVINTEL_PYTHON, expect_airflow=False, multiple_outputs=False)
    def register_best_candidate(run_key: str, evaluation: dict[str, Any]) -> dict[str, Any]:
        from ml.pipeline.lead_conversion import register_best_candidate as step

        return step(run_key, evaluation)

    @task.external_python(python=REVINTEL_PYTHON, expect_airflow=False, multiple_outputs=False)
    def assign_challenger_alias(run_key: str, registration: dict[str, Any]) -> dict[str, Any]:
        from ml.pipeline.lead_conversion import assign_challenger_alias as step

        return step(run_key, registration)

    snapshot = extract_crm_snapshot(RUN_KEY)
    features = build_features(RUN_KEY, snapshot)
    models = train_candidate_models(RUN_KEY, features)
    evaluation = evaluate_models(RUN_KEY, models)
    registration = register_best_candidate(RUN_KEY, evaluation)
    assign_challenger_alias(RUN_KEY, registration)


lead_conversion_training()
