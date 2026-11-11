"""Explicit MLflow tracking and Model Registry logging for trained lead conversion models.

Reads a finished `TrainingResult` (models, metadata, evaluation report); it never trains
or evaluates anything itself. Layout per training run:

- parent run `training-session`: dataset/feature/split parameters, baseline metrics and
  the evaluation report;
- one nested run per candidate: algorithm, hyperparameters, shared parameters, threshold,
  validation/test metrics, metadata and the full sklearn Pipeline, registered as a new
  version of the registered model.

No alias (champion/challenger) is ever assigned here: promotion is a manual decision.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ml.model_contract import PYFUNC_PREDICT_FN, model_signature, to_wire_frame
from ml.training.dataset import FeatureTable
from ml.training.mlflow_config import DEFAULT_EXPERIMENT, DEFAULT_REGISTERED_MODEL
from ml.training.train import BASELINE, TrainingResult

MODEL_ARTIFACT = "model"
INPUT_EXAMPLE_ROWS = 5
REGISTERED_MODEL_DESCRIPTION = (
    "Probability that a lead converts within 30 days after the prediction timestamp. "
    "Versions are candidates; champion/challenger aliases are assigned manually."
)
# MLflow metric name -> key in ml.evaluation.metrics.evaluate()
METRICS = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "brier": "brier",
}
# Models are stored with skops (no arbitrary code execution on load); these are the only
# non-standard types each pipeline contains.
SKOPS_TRUSTED_TYPES = {
    "logistic_regression": [
        "ml.training.pipelines.to_category",
        "ml.training.pipelines.to_float",
        "numpy.dtype",
    ],
    "xgboost": [
        "ml.training.pipelines.to_category",
        "ml.training.pipelines.to_float",
        "numpy.dtype",
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier",
    ],
}


@dataclass(frozen=True)
class TrackingSummary:
    experiment_id: str
    session_run_id: str
    versions: dict[str, str]  # candidate -> registered model version


# --- parameters -------------------------------------------------------------------------------


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def shared_params(metadata: dict[str, Any]) -> dict[str, str]:
    source, split = metadata["source"], metadata["split"]
    params = {
        "data.dataset_sha256": source["dataset_sha256"],
        "data.features_sha256": source["features_sha256"],
        "data.generator_version": source["generator_version"],
        "features.spec_version": metadata["feature_spec_version"],
        "features.spec_fingerprint": metadata["feature_spec_fingerprint"],
        "features.active": ",".join(metadata["active_features"]),
        "features.label": metadata["label"],
        "split.method": split["method"],
        "split.label_window_days": str(split["label_window_days"]),
        "split.validation_boundary": split["validation_start"],
        "split.test_boundary": split["test_start"],
        "split.purged_train_validation": str(split["purged"]["train_validation"]),
        "split.purged_validation_test": str(split["purged"]["validation_test"]),
    }
    for name, part in split["splits"].items():
        params |= {
            f"split.{name}_start": part["first_prediction_timestamp"],
            f"split.{name}_end": part["last_prediction_timestamp"],
            f"split.{name}_rows": str(part["rows"]),
            f"split.{name}_positive_rate": str(part["positive_rate"]),
            f"split.{name}_row_ids_sha256": part["row_ids_sha256"],
        }
    return params


def _metrics(by_split: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        f"{split}_{name}": float(values[key])
        for split, values in by_split.items()
        for name, key in METRICS.items()
        if values[key] is not None
    }


def baseline_check(candidate: dict[str, Any], baseline: dict[str, Any]) -> str:
    """'passed' when the candidate beats the naive baseline on validation PR-AUC and Brier.

    A sanity check against the constant baseline only; not a review or promotion decision.
    """
    beats = (
        candidate["average_precision"] > baseline["average_precision"]
        and candidate["brier"] < baseline["brier"]
    )
    return "passed" if beats else "failed"


def _ensure_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name, description=REGISTERED_MODEL_DESCRIPTION)


# --- logging and registration -----------------------------------------------------------------

SESSION_RUN_NAME = "training-session"
RUN_KEY_TAG = "pipeline_run_key"


@dataclass(frozen=True)
class LoggedCandidate:
    run_id: str
    model_uri: str
    model_id: str | None
    validation_baseline_check: str


@dataclass(frozen=True)
class LoggedSession:
    experiment_id: str
    session_run_id: str
    candidates: dict[str, LoggedCandidate]


def _find_logged_session(
    client: MlflowClient, experiment_id: str, run_key: str, names: list[str]
) -> LoggedSession | None:
    """A finished session already logged for `run_key` (a retry of the same pipeline run)."""
    sessions = client.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.{RUN_KEY_TAG} = '{run_key}' AND tags.role = 'training_session' "
            "AND tags.complete = 'true' AND attributes.status = 'FINISHED'"
        ),
    )
    for session in sessions:
        runs = client.search_runs(
            [experiment_id],
            filter_string=(
                f"tags.`mlflow.parentRunId` = '{session.info.run_id}' "
                "AND attributes.status = 'FINISHED'"
            ),
        )
        candidates = {
            run.data.tags["algorithm"]: LoggedCandidate(
                run_id=run.info.run_id,
                model_uri=run.data.tags["logged_model_uri"],
                model_id=run.data.tags.get("logged_model_id"),
                validation_baseline_check=run.data.tags["validation_baseline_check"],
            )
            for run in runs
            if "logged_model_uri" in run.data.tags
        }
        if sorted(candidates) == sorted(names):
            return LoggedSession(experiment_id, session.info.run_id, candidates)
    return None


def find_logged_session(
    tracking_uri: str, experiment: str, run_key: str, names: list[str]
) -> LoggedSession | None:
    """The finished session already logged for `run_key`, if any (no logging)."""
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    found = client.get_experiment_by_name(experiment)
    if found is None:
        return None
    return _find_logged_session(client, found.experiment_id, run_key, names)


def log_candidate_runs(
    result: TrainingResult,
    table: FeatureTable,
    *,
    tracking_uri: str,
    experiment: str = DEFAULT_EXPERIMENT,
    run_key: str | None = None,
    progress: Callable[[str], None] = print,
) -> LoggedSession:
    """Log one training session with a nested run (and model) per candidate; no registration.

    With `run_key`, a finished session already logged for that key is reused, so retries
    of the same pipeline run never log duplicates.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    experiment_id = mlflow.set_experiment(experiment).experiment_id
    if run_key is not None:
        existing = _find_logged_session(client, experiment_id, run_key, list(result.models))
        if existing is not None:
            progress(f"  reusing training session {existing.session_run_id} for {run_key}")
            return existing

    report = result.report
    any_metadata = next(iter(result.models.values())).metadata
    common = shared_params(any_metadata)
    git_commit = any_metadata.get("git_commit") or "unknown"
    baseline_metrics = report["metrics"][BASELINE]
    signature = model_signature()
    validation_rows = result.split.validation[:INPUT_EXAMPLE_ROWS]
    input_example = to_wire_frame(table.frame.iloc[validation_rows])
    requirements = [
        f"scikit-learn=={sklearn.__version__}",
        f"xgboost=={xgboost.__version__}",
        f"pandas=={pd.__version__}",
        f"numpy=={np.__version__}",
    ]
    key_tags = {RUN_KEY_TAG: run_key} if run_key is not None else {}

    candidates: dict[str, LoggedCandidate] = {}
    with mlflow.start_run(run_name=SESSION_RUN_NAME) as session:
        mlflow.set_tags({"role": "training_session", "git_commit": git_commit, **key_tags})
        mlflow.log_params(common)
        mlflow.log_param("threshold.rule", report["threshold_rule"])
        mlflow.log_metrics({f"baseline_{k}": v for k, v in _metrics(baseline_metrics).items()})
        mlflow.log_dict(report, "evaluation_report.json")

        for name, model in result.models.items():
            metadata = model.metadata
            check = baseline_check(
                metadata["metrics"]["validation"], baseline_metrics["validation"]
            )
            with mlflow.start_run(run_name=name, nested=True) as run:
                mlflow.set_tags(
                    {
                        "algorithm": name,
                        "role": "candidate",
                        "git_commit": git_commit,
                        "validation_baseline_check": check,
                        **key_tags,
                    }
                )
                mlflow.log_params(
                    {
                        **common,
                        "algorithm": name,
                        **{f"hp.{k}": _text(v) for k, v in metadata["hyperparameters"].items()},
                        "threshold.value": str(metadata["threshold"]["value"]),
                        "threshold.rule": metadata["threshold"]["rule"],
                    }
                )
                mlflow.log_metrics(
                    {
                        **_metrics(metadata["metrics"]),
                        "validation_threshold_f1": metadata["threshold"]["validation_f1"],
                    }
                )
                mlflow.log_dict(metadata, "metadata.json")
                info = mlflow.sklearn.log_model(
                    model.pipeline,
                    name=MODEL_ARTIFACT,
                    signature=signature,
                    input_example=input_example,
                    pyfunc_predict_fn=PYFUNC_PREDICT_FN,
                    pip_requirements=requirements,
                    skops_trusted_types=SKOPS_TRUSTED_TYPES[name],
                )
                mlflow.set_tags(
                    {"logged_model_uri": info.model_uri, "logged_model_id": info.model_id or ""}
                )
                candidates[name] = LoggedCandidate(
                    run_id=run.info.run_id,
                    model_uri=info.model_uri,
                    model_id=info.model_id,
                    validation_baseline_check=check,
                )
        mlflow.set_tag("complete", "true")
    return LoggedSession(experiment_id, session.info.run_id, candidates)


def version_tags(
    name: str, metadata: dict[str, Any], check: str, session_run_id: str
) -> dict[str, str]:
    validation = metadata["metrics"]["validation"]
    return {
        "algorithm": name,
        "role": "candidate",
        "validation_baseline_check": check,
        "threshold": str(metadata["threshold"]["value"]),
        "dataset_sha256": metadata["source"]["dataset_sha256"],
        "features_sha256": metadata["source"]["features_sha256"],
        "feature_spec_fingerprint": metadata["feature_spec_fingerprint"],
        "validation_pr_auc": str(validation["average_precision"]),
        "test_pr_auc": str(metadata["metrics"]["test"]["average_precision"]),
        "training_session_run_id": session_run_id,
        "git_commit": metadata.get("git_commit") or "unknown",
    }


def register_candidate(
    client: MlflowClient,
    registered_model: str,
    name: str,
    logged: LoggedCandidate,
    tags: dict[str, str],
    *,
    run_key: str | None = None,
) -> str:
    """Register a logged candidate as a new model version and return its number.

    The version is created together with its tags in one call. With `run_key`, an existing
    version for the same key and algorithm is returned instead (retry-safe).
    """
    _ensure_registered_model(client, registered_model)
    if run_key is not None:
        for version in client.search_model_versions(f"name='{registered_model}'"):
            if version.tags.get(RUN_KEY_TAG) == run_key and version.tags.get("algorithm") == name:
                return str(version.version)
        tags = {**tags, RUN_KEY_TAG: run_key}
    version = client.create_model_version(
        registered_model,
        source=logged.model_uri,
        run_id=logged.run_id,
        tags=tags,
        model_id=logged.model_id,
    )
    return str(version.version)


def log_candidates(
    result: TrainingResult,
    table: FeatureTable,
    *,
    tracking_uri: str,
    experiment: str = DEFAULT_EXPERIMENT,
    registered_model: str = DEFAULT_REGISTERED_MODEL,
    progress: Callable[[str], None] = print,
) -> TrackingSummary:
    """Log one training session and register every candidate as a new model version."""
    session = log_candidate_runs(
        result, table, tracking_uri=tracking_uri, experiment=experiment, progress=progress
    )
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    versions: dict[str, str] = {}
    for name, logged in session.candidates.items():
        tags = version_tags(
            name,
            result.models[name].metadata,
            logged.validation_baseline_check,
            session.session_run_id,
        )
        versions[name] = register_candidate(client, registered_model, name, logged, tags)
        progress(
            f"  {name}: run {logged.run_id}, {registered_model} version {versions[name]} "
            f"(validation_baseline_check={logged.validation_baseline_check})"
        )
    return TrackingSummary(session.experiment_id, session.session_run_id, versions)
