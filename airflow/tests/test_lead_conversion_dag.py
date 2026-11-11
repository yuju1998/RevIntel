"""DAG structure tests. Run inside the Airflow image (Airflow is not in the RevIntel env):

    docker compose -f airflow/docker-compose.yaml run --rm airflow-cli \
        bash -c "python -m pytest -q /opt/airflow/tests"
"""

from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from airflow.dag_processing.dagbag import DagBag

DAGS = Path("/opt/airflow/dags")
DAG_ID = "lead_conversion_training"
CHAIN = [
    "extract_crm_snapshot",
    "build_features",
    "train_candidate_models",
    "evaluate_models",
    "register_best_candidate",
    "assign_challenger_alias",
]


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    return DagBag(dag_folder=str(DAGS))


@pytest.fixture(scope="module")
def dag(dagbag: DagBag) -> Any:
    return dagbag.get_dag(DAG_ID)


def test_dag_imports_without_errors(dagbag: DagBag, dag: Any) -> None:
    assert dagbag.import_errors == {}
    assert dag is not None


def test_task_graph_is_the_expected_linear_chain(dag: Any) -> None:
    assert sorted(dag.task_ids) == sorted(CHAIN)
    for upstream, downstream in pairwise(CHAIN):
        assert dag.get_task(upstream).downstream_task_ids == {downstream}
    assert dag.get_task(CHAIN[0]).upstream_task_ids == set()
    assert dag.get_task(CHAIN[-1]).downstream_task_ids == set()


def test_weekly_schedule_without_catchup_or_overlap(dag: Any) -> None:
    assert dag.timetable.expression == "0 3 * * 1"  # weekly, Mondays 03:00 UTC
    assert str(dag.timetable.timezone) == "UTC"
    assert dag.catchup is False
    assert dag.max_active_runs == 1


def test_no_champion_promotion(dag: Any) -> None:
    assert not [task_id for task_id in dag.task_ids if "champion" in task_id]
    source = (DAGS / "lead_conversion_training.py").read_text()
    assert "promote_challenger" not in source and "set_registered_model_alias" not in source


def test_tasks_run_in_the_revintel_environment(dag: Any) -> None:
    for task_id in CHAIN:
        task = dag.get_task(task_id)
        assert task.python == "/opt/revintel-venv/bin/python"
        assert task.retries == 2
        assert task.multiple_outputs is False  # one small XCom per task
