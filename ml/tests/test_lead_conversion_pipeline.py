"""Pipeline steps for the weekly training DAG (no Airflow, temporary SQLite MLflow registry)."""

import dataclasses
import json
import os
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

import ml.training.train as training  # noqa: E402
from ml.features.build_features import (  # noqa: E402
    SourceExtract,
    SourceManifest,
    build_from_sources,
)
from ml.features.source_snapshot import read_source_snapshot, write_source_snapshot  # noqa: E402
from ml.features.sources import activities_frame, leads_frame  # noqa: E402
from ml.pipeline import lead_conversion as pipeline  # noqa: E402
from ml.synthetic.generator import SyntheticDataset  # noqa: E402
from ml.tests.conftest import FAST_XGBOOST  # noqa: E402
from ml.training.dataset import FeatureSchemaError, FeatureTable  # noqa: E402
from ml.training.train import (  # noqa: E402
    evaluate_candidates,
    fit_candidates,
    load_fitted,
    save_fitted,
    train_candidates,
)

MODEL = "lead-model"
RUN_KEY = pipeline.run_key_for("manual__2026-11-11T00:00:00+00:00")


def _extract(dataset: SyntheticDataset) -> tuple[SourceExtract, SourceManifest]:
    leads = leads_frame(dataset.leads)
    activities = activities_frame(dataset.activities)
    counts = {
        "leads": len(dataset.leads),
        "activities": len(dataset.activities),
        "lead_activities": len(activities),
    }
    manifest = SourceManifest(
        path=Path("manifest.json"),
        dataset_sha256="0" * 64,
        generator_version="test",
        as_of=dataset.config.as_of,
        max_lead_age_days=dataset.config.max_lead_age_days,
        row_counts=counts,
    )
    return SourceExtract(leads, activities, counts), manifest


# --- building blocks ------------------------------------------------------------------------------


def test_source_snapshot_reproduces_the_same_features(
    small_dataset: SyntheticDataset, tmp_path: Path
) -> None:
    extract, manifest = _extract(small_dataset)
    direct = build_from_sources(extract.leads, extract.activities, manifest, progress=print)

    metadata = write_source_snapshot(
        tmp_path, extract, manifest, "postgresql+psycopg://u:secret@db/revintel_ml"
    )
    leads, activities, saved_manifest = read_source_snapshot(tmp_path)
    rebuilt = build_from_sources(leads, activities, saved_manifest, progress=print)

    assert rebuilt.metadata["features_sha256"] == direct.metadata["features_sha256"]
    assert metadata["source_database"] == "revintel_ml"
    assert "secret" not in (tmp_path / "snapshot.json").read_text()


def test_fit_then_evaluate_equals_one_pass_training(
    training_table: FeatureTable, tmp_path: Path
) -> None:
    one_pass = train_candidates(training_table, **FAST_XGBOOST)
    fitted = fit_candidates(training_table, **FAST_XGBOOST)
    save_fitted(fitted, training_table, tmp_path)

    reloaded = evaluate_candidates(training_table, load_fitted(tmp_path, training_table))

    assert reloaded.report == one_pass.report
    other = dataclasses.replace(
        training_table, metadata={**training_table.metadata, "features_sha256": "x"}
    )
    with pytest.raises(FeatureSchemaError, match="different feature table"):
        load_fitted(tmp_path, other)


VALIDATION = {
    "logistic_regression": {"average_precision": 0.13, "brier": 0.039, "roc_auc": 0.71},
    "xgboost": {"average_precision": 0.15, "brier": 0.038, "roc_auc": 0.74},
}


def test_selection_uses_validation_pr_auc_among_baseline_passers() -> None:
    passed = {"logistic_regression": "passed", "xgboost": "passed"}

    assert pipeline.select_best_candidate(VALIDATION, passed).candidate == "xgboost"
    only_lr = {"logistic_regression": "passed", "xgboost": "failed"}
    assert pipeline.select_best_candidate(VALIDATION, only_lr).candidate == "logistic_regression"
    none = pipeline.select_best_candidate(VALIDATION, {"logistic_regression": "failed"})
    assert none.candidate is None and "baseline" in none.reason


def test_selection_tie_breaks_are_deterministic() -> None:
    tied = {
        "a": {"average_precision": 0.15, "brier": 0.040, "roc_auc": 0.70},
        "b": {"average_precision": 0.15, "brier": 0.038, "roc_auc": 0.70},
        "c": {"average_precision": 0.15, "brier": 0.038, "roc_auc": 0.75},
    }
    checks = dict.fromkeys(tied, "passed")

    assert pipeline.select_best_candidate(tied, checks).candidate == "c"
    tied["c"]["roc_auc"] = 0.70
    assert pipeline.select_best_candidate(tied, checks).candidate == "b"  # then by name


def test_selection_ignores_test_metrics() -> None:
    evaluation = {
        "session_run_id": "s",
        "candidates": {
            name: {
                "validation": metrics,
                "validation_baseline_check": "passed",
                # Test metrics favouring the other model must not matter.
                "test": {"average_precision": 0.99 if name == "logistic_regression" else 0.01},
            }
            for name, metrics in VALIDATION.items()
        },
    }
    validation_only = {n: c["validation"] for n, c in evaluation["candidates"].items()}
    checks = {n: "passed" for n in evaluation["candidates"]}

    assert pipeline.select_best_candidate(validation_only, checks).candidate == "xgboost"


def test_step_publishes_atomically_and_reuses_its_result(tmp_path: Path) -> None:
    artifacts = pipeline.RunArtifacts(tmp_path, RUN_KEY)
    calls: list[int] = []

    def compute(output: Path) -> dict[str, Any]:
        calls.append(1)
        (output / "data.txt").write_text("x")
        return {"value": 1}

    def fail(output: Path) -> dict[str, Any]:
        (output / "partial.txt").write_text("x")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        artifacts.step("step", fail)
    assert not artifacts.path("step").exists()  # nothing published by the failed attempt
    assert artifacts.step("step", compute) == {"value": 1, "reused": False}
    assert artifacts.step("step", compute) == {"value": 1, "reused": True}
    assert len(calls) == 1
    assert sorted(p.name for p in artifacts.directory.iterdir()) == ["step"]


# --- the whole pipeline ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> Iterator[pipeline.PipelineConfig]:
    directory = tmp_path_factory.mktemp("pipeline")
    uri = f"sqlite:///{directory / 'mlflow.db'}"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("lead-conversion", artifact_location=(directory / "mlruns").as_uri())
    mlflow.set_tracking_uri(previous)
    yield pipeline.PipelineConfig(
        pipeline_root=directory / "runs",
        source_database_url=None,
        manifest_path=directory / "manifest.json",
        tracking_uri=uri,
        registered_model=MODEL,
        experiment="lead-conversion",
    )


@pytest.fixture(scope="module")
def first_run(
    small_dataset: SyntheticDataset,
    config: pipeline.PipelineConfig,
    monkeypatch_module: pytest.MonkeyPatch,
) -> dict[str, dict[str, Any]]:
    # Extraction itself is covered against PostgreSQL in the integration tests; here the
    # snapshot is written from the in-memory dataset.
    extract, manifest = _extract(small_dataset)
    artifacts = pipeline.RunArtifacts(config.pipeline_root, RUN_KEY)
    artifacts.step(
        "snapshot",
        lambda out: (
            {"snapshot_dir": "snapshot"}
            | write_source_snapshot(out, extract, manifest, "postgresql+psycopg://u@h/revintel_ml")
        ),
    )
    # Small XGBoost search keeps the test fast; production defaults live in pipelines.py.
    monkeypatch_module.setattr(
        training, "fit_candidates", partial(training.fit_candidates, **FAST_XGBOOST)
    )
    features = pipeline.build_features(RUN_KEY, config)
    models = pipeline.train_candidate_models(RUN_KEY, config)
    evaluation = pipeline.evaluate_models(RUN_KEY, config)
    registration = pipeline.register_best_candidate(RUN_KEY, evaluation, config)
    challenger = pipeline.assign_challenger_alias(RUN_KEY, registration, config)
    return {
        "features": features,
        "models": models,
        "evaluation": evaluation,
        "registration": registration,
        "challenger": challenger,
    }


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    patcher = pytest.MonkeyPatch()
    yield patcher
    patcher.undo()


def _client(config: pipeline.PipelineConfig) -> MlflowClient:
    return MlflowClient(tracking_uri=config.tracking_uri, registry_uri=config.tracking_uri)


def _run_count(client: MlflowClient, config: pipeline.PipelineConfig) -> int:
    experiment = client.get_experiment_by_name(config.experiment)
    return len(client.search_runs([experiment.experiment_id]))


def test_pipeline_registers_the_best_candidate_as_challenger(
    first_run: dict[str, dict[str, Any]], config: pipeline.PipelineConfig
) -> None:
    evaluation, registration = first_run["evaluation"], first_run["registration"]
    client = _client(config)

    # XCom-sized results: paths, hashes, IDs and validation metrics only.
    assert len(json.dumps(first_run)) < 10_000
    for candidate in evaluation["candidates"].values():
        assert set(candidate) == {
            "validation",
            "run_id",
            "model_uri",
            "model_id",
            "validation_baseline_check",
        }
    expected = pipeline.select_best_candidate(
        {n: c["validation"] for n, c in evaluation["candidates"].items()},
        {n: c["validation_baseline_check"] for n, c in evaluation["candidates"].items()},
    )
    assert registration["selected"] == expected.candidate
    (version,) = client.search_model_versions(f"name='{MODEL}'")
    assert str(version.version) == registration["version"]
    assert version.tags["pipeline_run_key"] == RUN_KEY
    assert version.tags["algorithm"] == registration["selected"]
    aliases = client.get_registered_model(MODEL).aliases
    assert {k: str(v) for k, v in aliases.items()} == {"challenger": registration["version"]}
    assert first_run["challenger"]["previous_challenger"] is None


def test_retrying_every_step_creates_no_duplicates(
    first_run: dict[str, dict[str, Any]], config: pipeline.PipelineConfig
) -> None:
    client = _client(config)
    runs_before = _run_count(client, config)

    features = pipeline.build_features(RUN_KEY, config)
    models = pipeline.train_candidate_models(RUN_KEY, config)
    evaluation = pipeline.evaluate_models(RUN_KEY, config)
    registration = pipeline.register_best_candidate(RUN_KEY, evaluation, config)
    challenger = pipeline.assign_challenger_alias(RUN_KEY, registration, config)

    assert features["reused"] and models["reused"] and evaluation["reused"]
    assert evaluation["session_run_id"] == first_run["evaluation"]["session_run_id"]
    assert _run_count(client, config) == runs_before  # no duplicate MLflow runs
    assert registration["version"] == first_run["registration"]["version"]
    assert len(client.search_model_versions(f"name='{MODEL}'")) == 1  # no duplicate version
    assert challenger["previous_challenger"] == challenger["challenger"]


def test_champion_is_never_changed(
    first_run: dict[str, dict[str, Any]], config: pipeline.PipelineConfig
) -> None:
    client = _client(config)
    version = first_run["registration"]["version"]
    client.set_registered_model_alias(MODEL, "champion", version)  # simulate a manual promotion
    try:
        from ml.training.register import AliasWorkflowError

        # The champion version is never made challenger, and champion never moves.
        with pytest.raises(AliasWorkflowError, match="current champion"):
            pipeline.assign_challenger_alias(RUN_KEY, first_run["registration"], config)
        assert str(client.get_registered_model(MODEL).aliases["champion"]) == version
    finally:
        client.delete_registered_model_alias(MODEL, "champion")
        client.set_registered_model_alias(MODEL, "challenger", version)


def test_no_eligible_candidate_registers_nothing(config: pipeline.PipelineConfig) -> None:
    evaluation = {
        "session_run_id": "s",
        "candidates": {
            name: {"validation": metrics, "validation_baseline_check": "failed"}
            for name, metrics in VALIDATION.items()
        },
    }

    registration = pipeline.register_best_candidate("other-run", evaluation, config)
    challenger = pipeline.assign_challenger_alias("other-run", registration, config)

    assert registration["version"] is None
    assert challenger["challenger"] == "unchanged"
