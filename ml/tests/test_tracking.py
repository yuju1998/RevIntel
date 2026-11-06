"""MLflow tracking and registration against a temporary SQLite store (no server needed)."""

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.types import TensorSpec  # noqa: E402

from ml.features.build_features import FeatureBuild, write_outputs  # noqa: E402
from ml.features.feature_spec import ACTIVE_FEATURES  # noqa: E402
from ml.training import tracking, train  # noqa: E402
from ml.training.dataset import FeatureTable  # noqa: E402
from ml.training.train import TrainingResult  # noqa: E402

MODEL = tracking.DEFAULT_REGISTERED_MODEL
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def logged(
    trained: TrainingResult, training_table: FeatureTable, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[MlflowClient, tracking.TrackingSummary]]:
    directory = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{directory / 'mlflow.db'}"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(
        tracking.DEFAULT_EXPERIMENT, artifact_location=(directory / "artifacts").as_uri()
    )
    summary = tracking.log_candidates(
        trained, training_table, tracking_uri=uri, progress=lambda _: None
    )
    yield MlflowClient(tracking_uri=uri, registry_uri=uri), summary
    mlflow.set_tracking_uri(previous)


def test_session_and_candidate_runs(
    logged: tuple[MlflowClient, tracking.TrackingSummary], trained: TrainingResult
) -> None:
    client, summary = logged
    runs = {r.info.run_name: r for r in client.search_runs([summary.experiment_id])}

    assert set(runs) == {"training-session", "logistic_regression", "xgboost"}
    assert runs["training-session"].data.tags["role"] == "training_session"
    assert "baseline_test_pr_auc" in runs["training-session"].data.metrics
    for name in ("logistic_regression", "xgboost"):
        run, metadata = runs[name], trained.models[name].metadata
        params, tags = run.data.params, run.data.tags
        assert tags["mlflow.parentRunId"] == summary.session_run_id
        assert tags["algorithm"] == params["algorithm"] == name
        assert params["data.dataset_sha256"] == metadata["source"]["dataset_sha256"]
        assert params["data.features_sha256"] == metadata["source"]["features_sha256"]
        assert params["features.spec_fingerprint"] == metadata["feature_spec_fingerprint"]
        assert params["features.active"].split(",") == list(ACTIVE_FEATURES)
        assert params["split.test_boundary"] == metadata["split"]["test_start"]
        assert params["split.purged_validation_test"] == str(
            metadata["split"]["purged"]["validation_test"]
        )
        assert float(params["threshold.value"]) == metadata["threshold"]["value"]
        assert params["threshold.rule"] == metadata["threshold"]["rule"]
        assert {f"{s}_{m}" for s in ("validation", "test") for m in tracking.METRICS} <= set(
            run.data.metrics
        )
        assert run.data.metrics["test_pr_auc"] == pytest.approx(
            metadata["metrics"]["test"]["average_precision"]
        )
    assert runs["xgboost"].data.params["hp.max_depth"] == str(
        trained.models["xgboost"].metadata["hyperparameters"]["max_depth"]
    )


def test_candidates_are_registered_without_aliases(
    logged: tuple[MlflowClient, tracking.TrackingSummary],
) -> None:
    client, summary = logged
    versions = client.search_model_versions(f"name='{MODEL}'")

    assert sorted(str(v.version) for v in versions) == sorted(summary.versions.values())
    for version in versions:
        assert version.tags["role"] == "candidate"
        assert version.tags["algorithm"] in summary.versions
        assert version.tags["validation_baseline_check"] in {"passed", "failed"}
        assert "validation_status" not in version.tags  # reserved for a review decision
        assert version.tags["training_session_run_id"] == summary.session_run_id
        assert not version.aliases
    assert client.get_registered_model(MODEL).aliases == {}


def test_registered_model_serves_probabilities(
    logged: tuple[MlflowClient, tracking.TrackingSummary],
    trained: TrainingResult,
    training_table: FeatureTable,
) -> None:
    _, summary = logged
    rows = tracking.to_wire_frame(training_table.frame.iloc[trained.split.test[:4]])
    rows.loc[0, "industry"] = None
    rows.loc[1, "industry"] = "Aerospace"
    rows.loc[2, ["company_size", "days_since_last_activity"]] = np.nan

    for name, version in summary.versions.items():
        model = mlflow.pyfunc.load_model(f"models:/{MODEL}/{version}")
        signature = model.metadata.signature
        assert signature.inputs.input_names() == list(ACTIVE_FEATURES)
        (output,) = signature.outputs.inputs
        assert isinstance(output, TensorSpec) and output.shape == (-1, 2)

        probabilities = model.predict(rows)

        assert probabilities.shape == (4, 2)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        np.testing.assert_allclose(probabilities, trained.models[name].pipeline.predict_proba(rows))


def test_baseline_check() -> None:
    baseline = {"average_precision": 0.04, "brier": 0.041}

    assert tracking.baseline_check({"average_precision": 0.1, "brier": 0.040}, baseline) == "passed"
    assert tracking.baseline_check({"average_precision": 0.1, "brier": 0.050}, baseline) == "failed"
    assert tracking.baseline_check({"average_precision": 0.03, "brier": 0.03}, baseline) == "failed"


# --- CLI behaviour ----------------------------------------------------------------------------


def test_track_fails_before_training_when_server_is_down(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def never(*_: object, **__: object) -> None:
        raise AssertionError("must not load or train when the tracking server is down")

    monkeypatch.setattr(train, "load_feature_table", never)

    code = train.main(
        ["--track", "--tracking-uri", "http://127.0.0.1:9", "--output-dir", str(tmp_path)]
    )

    assert code == 2
    assert "scripts/start_mlflow.sh" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_untracked_training_never_touches_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    trained: TrainingResult,
    training_table: FeatureTable,
    tmp_path: Path,
) -> None:
    features = tmp_path / "features.parquet"
    write_outputs(FeatureBuild(training_table.frame, training_table.metadata, {}), features)
    monkeypatch.setattr(train, "train_candidates", lambda _table: trained)

    def never(*_: object, **__: object) -> None:
        raise AssertionError("MLflow must not be used without --track")

    monkeypatch.setattr(tracking, "log_candidates", never)
    monkeypatch.setattr(train, "require_tracking_server", never)

    assert train.main(["--features", str(features), "--output-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "xgboost" / "model.joblib").exists()


# --- startup script ---------------------------------------------------------------------------


def _start(tmp_path: Path, uri: str | None) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "scripts" / "start_mlflow.sh"  # isolated copy: no repo .env
    script.parent.mkdir()
    shutil.copy(ROOT / "scripts" / "start_mlflow.sh", script)
    env = {"PATH": os.environ["PATH"], **({"MLFLOW_BACKEND_STORE_URI": uri} if uri else {})}
    return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        (None, "MLFLOW_BACKEND_STORE_URI is not set"),
        ("sqlite:///mlflow.db", "must be a PostgreSQL URI"),
        ("postgresql+psycopg://u:p@localhost:5432/revintel", "refusing backend database"),
        ("postgresql+psycopg://u:p@localhost:5432/revintel_ml", "refusing backend database"),
    ],
)
def test_start_script_refuses_unsafe_configuration(
    tmp_path: Path, uri: str | None, message: str
) -> None:
    result = _start(tmp_path, uri)

    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / "artifacts").exists()  # nothing created before the checks
