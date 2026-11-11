"""Steps of the weekly lead conversion training pipeline (run by the Airflow DAG).

Each step reuses the existing feature, training, evaluation and MLflow code and returns a
small JSON-safe dict (paths relative to the run directory, hashes, IDs, validation
metrics). Large artifacts live in a per-run directory under REVINTEL_PIPELINE_ROOT.

Idempotency within one pipeline run (`run_key`):
- file-producing steps publish their output directory atomically with a `_SUCCESS.json`
  marker; a retry that finds the marker returns the recorded result without recomputing;
- MLflow logging reuses a finished session already logged for the run key;
- registration returns the version already registered for the run key and algorithm;
- the `challenger` alias assignment is idempotent.

The `champion` alias is never touched: promotion stays manual (`ml.training.promote`).
"""

import json
import os
import re
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.features.build_features import (
    DEFAULT_MANIFEST,
    SourceManifest,
    build_from_sources,
    extract_sources,
)
from ml.features.build_features import write_outputs as write_features
from ml.features.source_snapshot import read_source_snapshot, write_source_snapshot
from ml.training.mlflow_config import (
    DEFAULT_EXPERIMENT,
    default_registered_model,
    default_tracking_uri,
    require_tracking_server,
)

PIPELINE = "lead_conversion_training"
SUCCESS_FILE = "_SUCCESS.json"
FEATURES_FILE = "lead_features.parquet"
CANDIDATES = ("logistic_regression", "xgboost")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_ROOT = ROOT / "artifacts" / "pipeline_runs"


@dataclass(frozen=True)
class PipelineConfig:
    pipeline_root: Path
    source_database_url: str | None
    manifest_path: Path
    tracking_uri: str
    registered_model: str
    experiment: str

    @classmethod
    def from_env(cls) -> PipelineConfig:
        return cls(
            pipeline_root=Path(os.environ.get("REVINTEL_PIPELINE_ROOT", DEFAULT_PIPELINE_ROOT)),
            source_database_url=os.environ.get("SYNTHETIC_DATABASE_URL"),
            manifest_path=Path(os.environ.get("REVINTEL_SOURCE_MANIFEST", DEFAULT_MANIFEST)),
            tracking_uri=default_tracking_uri(),
            registered_model=default_registered_model(),
            experiment=os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT),
        )


def run_key_for(run_id: str) -> str:
    return f"{PIPELINE}/{run_id}"


@dataclass(frozen=True)
class RunArtifacts:
    """The per-run artifact directory. Results refer to files by paths relative to it, so a
    remote store (e.g. S3) can later replace this local directory without changing them."""

    root: Path
    run_key: str

    @property
    def directory(self) -> Path:
        return self.root / re.sub(r"[^A-Za-z0-9._-]+", "_", self.run_key)

    def path(self, relative: str) -> Path:
        return self.directory / relative

    def step(self, name: str, compute: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
        """Run `compute(output_dir)` once per run; publish its directory atomically."""
        final = self.directory / name
        marker = final / SUCCESS_FILE
        if marker.exists():
            return {**json.loads(marker.read_text()), "reused": True}
        work = self.directory / f".{name}.tmp-{uuid.uuid4().hex}"
        work.mkdir(parents=True)
        try:
            result = compute(work)
            (work / SUCCESS_FILE).write_text(json.dumps(result, indent=2) + "\n")
            if final.exists():  # left by an attempt that failed before publishing
                shutil.rmtree(final)
            work.rename(final)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return {**result, "reused": False}


def _artifacts(run_key: str, config: PipelineConfig) -> RunArtifacts:
    return RunArtifacts(config.pipeline_root, run_key)


# --- steps --------------------------------------------------------------------------------------


def extract_crm_snapshot(run_key: str, config: PipelineConfig | None = None) -> dict[str, Any]:
    """Materialize the feature pipeline's source rows (read-only extraction)."""
    config = config or PipelineConfig.from_env()
    if not config.source_database_url:
        raise ValueError("SYNTHETIC_DATABASE_URL is not set (the source CRM dataset)")

    def compute(output: Path) -> dict[str, Any]:
        manifest = SourceManifest.read(config.manifest_path)
        extract = extract_sources(config.source_database_url, manifest)
        metadata = write_source_snapshot(output, extract, manifest, config.source_database_url)
        return {
            "snapshot_dir": "snapshot",
            "source_database": metadata["source_database"],
            "row_counts": metadata["row_counts"],
            "content_sha256": metadata["content_sha256"],
            "dataset_sha256": manifest.dataset_sha256,
        }

    return _artifacts(run_key, config).step("snapshot", compute)


def build_features(run_key: str, config: PipelineConfig | None = None) -> dict[str, Any]:
    """Snapshots, labels and features from the run's source snapshot."""
    config = config or PipelineConfig.from_env()
    artifacts = _artifacts(run_key, config)

    def compute(output: Path) -> dict[str, Any]:
        leads, activities, manifest = read_source_snapshot(artifacts.path("snapshot"))
        result = build_from_sources(leads, activities, manifest, progress=lambda _: None)
        write_features(result, output / FEATURES_FILE)
        return {
            "features_path": f"features/{FEATURES_FILE}",
            "features_sha256": result.metadata["features_sha256"],
            "rows": result.metadata["row_count"],
            "positive_rate": result.metadata["positive_rate"],
        }

    return artifacts.step("features", compute)


def train_candidate_models(run_key: str, config: PipelineConfig | None = None) -> dict[str, Any]:
    """Fit both candidates once (XGBoost depth/early stopping use validation only)."""
    from ml.training.dataset import load_feature_table
    from ml.training.train import fit_candidates, save_fitted

    config = config or PipelineConfig.from_env()
    artifacts = _artifacts(run_key, config)

    def compute(output: Path) -> dict[str, Any]:
        table = load_feature_table(artifacts.path(f"features/{FEATURES_FILE}"))
        fitted = fit_candidates(table)
        save_fitted(fitted, table, output)
        return {
            "models_dir": "models",
            "candidates": sorted(fitted.pipelines),
            "xgboost": {
                "max_depth": fitted.xgboost["max_depth"],
                "best_iteration": fitted.xgboost["best_iteration"],
            },
        }

    return artifacts.step("models", compute)


def evaluate_models(run_key: str, config: PipelineConfig | None = None) -> dict[str, Any]:
    """Thresholds on validation, one test evaluation, then log both candidates to MLflow."""
    from ml.training.dataset import load_feature_table
    from ml.training.train import evaluate_candidates, load_fitted
    from ml.training.train import write_outputs as write_models

    config = config or PipelineConfig.from_env()
    artifacts = _artifacts(run_key, config)
    require_tracking_server(config.tracking_uri)
    table = load_feature_table(artifacts.path(f"features/{FEATURES_FILE}"))
    holder: dict[str, Any] = {}

    def compute(output: Path) -> dict[str, Any]:
        result = evaluate_candidates(table, load_fitted(artifacts.path("models"), table))
        write_models(result, output)
        holder["result"] = result
        return {
            "evaluation_dir": "evaluation",
            "baseline_validation": _validation_summary(
                result.report["metrics"]["baseline"]["validation"]
            ),
            "candidates": {
                name: {"validation": _validation_summary(model.metadata["metrics"]["validation"])}
                for name, model in result.models.items()
            },
        }

    evaluation = artifacts.step("evaluation", compute)
    from ml.training.tracking import find_logged_session, log_candidate_runs

    session = find_logged_session(
        config.tracking_uri, config.experiment, run_key, list(evaluation["candidates"])
    )
    if session is None:
        result = holder.get("result") or _reload_result(table, artifacts)
        session = log_candidate_runs(
            result,
            table,
            tracking_uri=config.tracking_uri,
            experiment=config.experiment,
            run_key=run_key,
            progress=lambda _: None,
        )
    for name, logged in session.candidates.items():
        evaluation["candidates"][name] |= {
            "run_id": logged.run_id,
            "model_uri": logged.model_uri,
            "model_id": logged.model_id,
            "validation_baseline_check": logged.validation_baseline_check,
        }
    return {**evaluation, "session_run_id": session.session_run_id}


def _reload_result(table: Any, artifacts: RunArtifacts) -> Any:
    """Re-evaluate from the saved fitted models (deterministic) when the evaluation step was
    reused but MLflow logging still has to happen."""
    from ml.training.train import evaluate_candidates, load_fitted

    return evaluate_candidates(table, load_fitted(artifacts.path("models"), table))


def _validation_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("roc_auc", "average_precision", "brier", "precision", "recall", "f1", "threshold")
    return {key: metrics[key] for key in keys}


# --- selection and registration ------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    candidate: str | None
    reason: str


def select_best_candidate(
    validation: Mapping[str, Mapping[str, float]], baseline_checks: Mapping[str, str]
) -> Selection:
    """Choose the candidate to register from VALIDATION metrics only.

    Eligible: `validation_baseline_check == "passed"`. Best: highest validation PR-AUC
    (average precision); ties go to lower validation Brier, then higher validation ROC-AUC,
    then algorithm name. Test metrics are never an input.
    """
    eligible = sorted(name for name, check in baseline_checks.items() if check == "passed")
    if not eligible:
        return Selection(None, "no candidate passed the validation baseline check")
    best = min(
        eligible,
        key=lambda name: (
            -validation[name]["average_precision"],
            validation[name]["brier"],
            -validation[name]["roc_auc"],
            name,
        ),
    )
    return Selection(best, f"highest validation PR-AUC among {eligible}")


def register_best_candidate(
    run_key: str, evaluation: Mapping[str, Any], config: PipelineConfig | None = None
) -> dict[str, Any]:
    """Register the selected candidate as a new model version (once per run key)."""
    from mlflow import MlflowClient

    from ml.training.tracking import LoggedCandidate, register_candidate, version_tags

    config = config or PipelineConfig.from_env()
    candidates = evaluation["candidates"]
    selection = select_best_candidate(
        {name: c["validation"] for name, c in candidates.items()},
        {name: c["validation_baseline_check"] for name, c in candidates.items()},
    )
    if selection.candidate is None:
        return {"selected": None, "version": None, "reason": selection.reason}

    require_tracking_server(config.tracking_uri)
    name = selection.candidate
    chosen = candidates[name]
    artifacts = _artifacts(run_key, config)
    metadata = json.loads(artifacts.path(f"evaluation/{name}/metadata.json").read_text())
    client = MlflowClient(tracking_uri=config.tracking_uri, registry_uri=config.tracking_uri)
    logged = LoggedCandidate(
        run_id=chosen["run_id"],
        model_uri=chosen["model_uri"],
        model_id=chosen.get("model_id") or None,
        validation_baseline_check=chosen["validation_baseline_check"],
    )
    tags = version_tags(
        name, metadata, chosen["validation_baseline_check"], evaluation["session_run_id"]
    )
    version = register_candidate(
        client, config.registered_model, name, logged, tags, run_key=run_key
    )
    return {
        "selected": name,
        "version": version,
        "registered_model": config.registered_model,
        "reason": selection.reason,
        "validation_pr_auc": chosen["validation"]["average_precision"],
    }


def assign_challenger_alias(
    run_key: str, registration: Mapping[str, Any], config: PipelineConfig | None = None
) -> dict[str, Any]:
    """Point `challenger` at the newly registered version (idempotent; never `champion`)."""
    from mlflow import MlflowClient

    from ml.training.register import CHALLENGER, designate_challenger, registry_state

    config = config or PipelineConfig.from_env()
    version = registration.get("version")
    if version is None:
        return {"challenger": "unchanged", "reason": registration.get("reason")}
    require_tracking_server(config.tracking_uri)
    client = MlflowClient(tracking_uri=config.tracking_uri, registry_uri=config.tracking_uri)
    aliases, _ = registry_state(client, config.registered_model, str(version))
    lines = designate_challenger(client, config.registered_model, str(version))
    return {
        "registered_model": config.registered_model,
        "previous_challenger": aliases.get(CHALLENGER),
        "challenger": str(version),
        "output": lines,
    }
