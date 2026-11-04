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
import urllib.error
import urllib.request
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
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema, TensorSpec

from ml.features.feature_spec import ACTIVE_FEATURES, FEATURES_BY_NAME
from ml.training.dataset import FeatureTable
from ml.training.train import BASELINE, TrainingResult

DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
DEFAULT_EXPERIMENT = "lead-conversion"
DEFAULT_REGISTERED_MODEL = "revintel-lead-conversion"
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
_WIRE_TYPES = {"categorical": "string", "count": "long", "numeric": "double"}
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


class TrackingUnavailableError(RuntimeError):
    """The MLflow tracking server cannot be reached."""


@dataclass(frozen=True)
class TrackingSummary:
    experiment_id: str
    session_run_id: str
    versions: dict[str, str]  # candidate -> registered model version


def require_tracking_server(tracking_uri: str, timeout: float = 3.0) -> None:
    """Fail fast (before training) when an HTTP tracking server is not reachable."""
    if not tracking_uri.startswith(("http://", "https://")):
        return
    try:
        with urllib.request.urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=timeout):
            pass
    except (urllib.error.URLError, OSError) as exc:
        raise TrackingUnavailableError(
            f"MLflow tracking server at {tracking_uri} is not reachable ({exc}). "
            "Start it with scripts/start_mlflow.sh, or train without --track."
        ) from exc


# --- model signature --------------------------------------------------------------------------


def model_signature() -> ModelSignature:
    """Input: the active features in spec order (JSON-friendly types, nullable values allowed).
    Output: `predict_proba` probabilities, shape (rows, 2) = [P(not converted), P(converted)]."""
    inputs = Schema(
        [ColSpec(_WIRE_TYPES[FEATURES_BY_NAME[name].kind], name) for name in ACTIVE_FEATURES]
    )
    outputs = Schema([TensorSpec(np.dtype("float64"), (-1, 2), "probabilities")])
    return ModelSignature(inputs=inputs, outputs=outputs)


def to_wire_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Active features as callers send them: strings, int64 counts, float64 (NaN = missing)."""
    columns = {}
    for name in ACTIVE_FEATURES:
        kind = FEATURES_BY_NAME[name].kind
        column = features[name].astype(object).where(features[name].notna(), None)
        if kind == "categorical":
            columns[name] = column.map(lambda v: None if v is None else str(v))
        elif kind == "count":
            columns[name] = column.astype("int64")
        else:
            columns[name] = column.astype("float64")
    return pd.DataFrame(columns).reset_index(drop=True)


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


# --- logging ----------------------------------------------------------------------------------


def log_candidates(
    result: TrainingResult,
    table: FeatureTable,
    *,
    tracking_uri: str,
    experiment: str = DEFAULT_EXPERIMENT,
    registered_model: str = DEFAULT_REGISTERED_MODEL,
    progress: Callable[[str], None] = print,
) -> TrackingSummary:
    """Log one training session and register each candidate as a new model version."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    experiment_id = mlflow.set_experiment(experiment).experiment_id
    _ensure_registered_model(client, registered_model)

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

    versions: dict[str, str] = {}
    with mlflow.start_run(run_name="training-session") as session:
        mlflow.set_tags({"role": "training_session", "git_commit": git_commit})
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
                    pyfunc_predict_fn="predict_proba",
                    pip_requirements=requirements,
                    skops_trusted_types=SKOPS_TRUSTED_TYPES[name],
                    registered_model_name=registered_model,
                )
                version = str(info.registered_model_version)
                validation = metadata["metrics"]["validation"]
                for key, value in {
                    "algorithm": name,
                    "role": "candidate",
                    "validation_baseline_check": check,
                    "threshold": str(metadata["threshold"]["value"]),
                    "dataset_sha256": metadata["source"]["dataset_sha256"],
                    "features_sha256": metadata["source"]["features_sha256"],
                    "feature_spec_fingerprint": metadata["feature_spec_fingerprint"],
                    "validation_pr_auc": str(validation["average_precision"]),
                    "test_pr_auc": str(metadata["metrics"]["test"]["average_precision"]),
                    "training_session_run_id": session.info.run_id,
                    "git_commit": git_commit,
                }.items():
                    client.set_model_version_tag(registered_model, version, key, value)
                versions[name] = version
                progress(
                    f"  {name}: run {run.info.run_id}, {registered_model} version {version} "
                    f"(validation_baseline_check={check})"
                )
    return TrackingSummary(experiment_id, session.info.run_id, versions)
