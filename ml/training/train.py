"""Train and evaluate the lead conversion candidates on one shared split.

Both candidates (Logistic Regression baseline, XGBoost challenger) and the naive baseline
use the same feature table, active features, labels, purged chronological split and
evaluation code. Order of use of the data:

1. fit on train;
2. on validation: XGBoost depth / early stopping, then each decision threshold, frozen;
3. on test: one final evaluation with the frozen thresholds.
"""

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.pipeline import Pipeline

from ml.evaluation.metrics import THRESHOLD_RULE, constant_scores, evaluate, select_threshold
from ml.features.build_features import DEFAULT_OUTPUT as DEFAULT_FEATURES
from ml.features.build_features import git_commit
from ml.features.feature_spec import ACTIVE_FEATURES, LABEL
from ml.training.artifacts import LeadConversionModel, save_model
from ml.training.dataset import (
    FeatureSchemaError,
    FeatureTable,
    features_and_label,
    load_feature_table,
)
from ml.training.mlflow_config import (
    DEFAULT_EXPERIMENT,
    DEFAULT_TRACKING_URI,
    TrackingUnavailableError,
    default_registered_model,
    default_tracking_uri,
    require_tracking_server,
)
from ml.training.pipelines import (
    LOGISTIC_REGRESSION_PARAMS,
    XGBOOST_DEPTHS,
    XGBOOST_EARLY_STOPPING_ROUNDS,
    XGBOOST_MAX_ESTIMATORS,
    XGBOOST_PARAMS,
    build_logistic_regression,
    fit_xgboost,
)
from ml.training.splits import SPLITS, ChronologicalSplit, purged_chronological_split

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "models" / "lead_conversion"
REPORT_FILE = "evaluation_report.json"
FIT_FILE = "fit.json"
BASELINE = "baseline"


@dataclass(frozen=True)
class TrainingResult:
    split: ChronologicalSplit
    models: dict[str, LeadConversionModel]
    report: dict[str, Any]


@dataclass(frozen=True)
class FittedCandidates:
    """Fitted pipelines and the split they were fitted on (before thresholds and metrics)."""

    split: ChronologicalSplit
    pipelines: dict[str, Pipeline]
    training_rate: float
    xgboost: dict[str, Any]  # chosen depth, best iteration and the search settings


def _parts(
    table: FeatureTable, split: ChronologicalSplit
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    X, y = features_and_label(table.frame.reset_index(drop=True))
    return {name: (X.iloc[split.rows(name)], y[split.rows(name)]) for name in SPLITS}


def fit_candidates(
    table: FeatureTable,
    *,
    xgboost_depths: tuple[int, ...] = XGBOOST_DEPTHS,
    xgboost_max_estimators: int = XGBOOST_MAX_ESTIMATORS,
    xgboost_early_stopping_rounds: int = XGBOOST_EARLY_STOPPING_ROUNDS,
) -> FittedCandidates:
    """Fit both candidates on train (XGBoost uses validation for depth and early stopping)."""
    split = purged_chronological_split(table.frame.reset_index(drop=True)["prediction_timestamp"])
    parts = _parts(table, split)
    X_train, y_train = parts["train"]
    X_validation, y_validation = parts["validation"]
    logistic = build_logistic_regression().fit(X_train, y_train)
    xgb = fit_xgboost(
        X_train,
        y_train,
        X_validation,
        y_validation,
        depths=xgboost_depths,
        max_estimators=xgboost_max_estimators,
        early_stopping_rounds=xgboost_early_stopping_rounds,
    )
    return FittedCandidates(
        split=split,
        pipelines={"logistic_regression": logistic, "xgboost": xgb.pipeline},
        training_rate=float(y_train.mean()),
        xgboost={
            "max_depth": xgb.max_depth,
            "best_iteration": xgb.best_iteration,
            "max_estimators": xgboost_max_estimators,
            "early_stopping_rounds": xgboost_early_stopping_rounds,
            "depths_tried": list(xgboost_depths),
            "validation_average_precision_by_depth": {
                str(k): v for k, v in xgb.validation_average_precision_by_depth.items()
            },
        },
    )


def save_fitted(fitted: FittedCandidates, table: FeatureTable, directory: Path) -> None:
    """Persist fitted pipelines (joblib: load only this project's own artifacts)."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, pipeline in fitted.pipelines.items():
        joblib.dump(pipeline, directory / f"{name}.joblib")
    split = fitted.split.metadata(table.frame.reset_index(drop=True))
    (directory / FIT_FILE).write_text(
        json.dumps(
            {
                "features_sha256": table.metadata["features_sha256"],
                "training_rate": fitted.training_rate,
                "xgboost": fitted.xgboost,
                "split_row_ids_sha256": {
                    name: part["row_ids_sha256"] for name, part in split["splits"].items()
                },
            },
            indent=2,
        )
        + "\n"
    )


def load_fitted(directory: Path, table: FeatureTable) -> FittedCandidates:
    """Reload fitted pipelines; the split is recomputed and must match the one fitted on."""
    saved = json.loads((directory / FIT_FILE).read_text())
    if saved["features_sha256"] != table.metadata["features_sha256"]:
        raise FeatureSchemaError("fitted models were trained on a different feature table")
    frame = table.frame.reset_index(drop=True)
    split = purged_chronological_split(frame["prediction_timestamp"])
    hashes = {
        name: part["row_ids_sha256"] for name, part in split.metadata(frame)["splits"].items()
    }
    if hashes != saved["split_row_ids_sha256"]:
        raise FeatureSchemaError("fitted models were trained on a different split")
    return FittedCandidates(
        split=split,
        pipelines={
            name: joblib.load(directory / f"{name}.joblib")
            for name in ("logistic_regression", "xgboost")
        },
        training_rate=saved["training_rate"],
        xgboost=saved["xgboost"],
    )


def train_candidates(
    table: FeatureTable,
    *,
    xgboost_depths: tuple[int, ...] = XGBOOST_DEPTHS,
    xgboost_max_estimators: int = XGBOOST_MAX_ESTIMATORS,
    xgboost_early_stopping_rounds: int = XGBOOST_EARLY_STOPPING_ROUNDS,
) -> TrainingResult:
    fitted = fit_candidates(
        table,
        xgboost_depths=xgboost_depths,
        xgboost_max_estimators=xgboost_max_estimators,
        xgboost_early_stopping_rounds=xgboost_early_stopping_rounds,
    )
    return evaluate_candidates(table, fitted)


def evaluate_candidates(table: FeatureTable, fitted: FittedCandidates) -> TrainingResult:
    """Thresholds from validation (frozen), then one final evaluation on test."""
    split, pipelines = fitted.split, fitted.pipelines
    split_metadata = split.metadata(table.frame.reset_index(drop=True))
    parts = _parts(table, split)
    X_validation, y_validation = parts["validation"]

    def scores(name: str, features: pd.DataFrame) -> np.ndarray:
        if name == BASELINE:
            return constant_scores(fitted.training_rate, len(features))
        return pipelines[name].predict_proba(features)[:, 1]

    names = (BASELINE, *pipelines)
    # Thresholds from validation only; frozen before the test split is touched.
    thresholds = {
        name: select_threshold(y_validation, scores(name, X_validation)) for name in names
    }
    validation = {
        name: evaluate(y_validation, scores(name, X_validation), thresholds[name].value)
        for name in names
    }
    # Final evaluation on test with the frozen thresholds.
    X_test, y_test = parts["test"]
    test = {name: evaluate(y_test, scores(name, X_test), thresholds[name].value) for name in names}

    hyperparameters = {
        "logistic_regression": {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "numeric_preprocessing": "float -> median imputation + missing indicator -> scaling",
            **LOGISTIC_REGRESSION_PARAMS,
        },
        "xgboost": {
            "estimator": "xgboost.XGBClassifier",
            "numeric_preprocessing": "float -> native NaN handling (no imputation, no scaling)",
            **XGBOOST_PARAMS,
            **fitted.xgboost,
        },
    }
    common = {
        "label": LABEL,
        "active_features": list(ACTIVE_FEATURES),
        "feature_spec_version": table.metadata["feature_spec_version"],
        "feature_spec_fingerprint": table.metadata["feature_spec_fingerprint"],
        "source": {
            "features_path": str(table.path),
            "features_sha256": table.metadata["features_sha256"],
            "dataset_sha256": table.metadata["source"]["dataset_sha256"],
            "generator_version": table.metadata["source"]["generator_version"],
        },
        "split": split_metadata,
        "categorical_semantics": "missing -> '__missing__'; one-hot; unknown categories ignored",
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "git_commit": git_commit(),
    }
    models = {
        name: LeadConversionModel(
            pipeline=pipeline,
            metadata={
                "model_name": name,
                **common,
                "hyperparameters": hyperparameters[name],
                "threshold": {
                    "value": thresholds[name].value,
                    "rule": THRESHOLD_RULE,
                    "validation_f1": thresholds[name].validation_f1,
                },
                "metrics": {"validation": validation[name], "test": test[name]},
            },
        )
        for name, pipeline in pipelines.items()
    }
    report = {
        **{key: common[key] for key in ("source", "feature_spec_version", "split")},
        "feature_spec_fingerprint": common["feature_spec_fingerprint"],
        "baseline": f"constant probability = training positive rate ({fitted.training_rate:.6f})",
        "threshold_rule": THRESHOLD_RULE,
        "thresholds": {name: thresholds[name].value for name in names},
        "xgboost": {
            "max_depth": fitted.xgboost["max_depth"],
            "best_iteration": fitted.xgboost["best_iteration"],
        },
        "metrics": {name: {"validation": validation[name], "test": test[name]} for name in names},
    }
    return TrainingResult(split=split, models=models, report=report)


def write_outputs(result: TrainingResult, output_dir: Path) -> None:
    for name, model in result.models.items():
        save_model(model, output_dir / name)
    (output_dir / REPORT_FILE).write_text(json.dumps(result.report, indent=2) + "\n")


def format_report(report: dict[str, Any]) -> str:
    split = report["split"]
    lines = [
        f"Split (validation starts {split['validation_start']}, test starts "
        f"{split['test_start']}; purged train/validation "
        f"{split['purged']['train_validation']:,}, validation/test "
        f"{split['purged']['validation_test']:,}):",
    ]
    for name, s in split["splits"].items():
        lines.append(
            f"  {name:<10} {s['rows']:>8,} rows  positive rate {s['positive_rate']:.4f}  "
            f"T {s['first_prediction_timestamp'][:10]} .. {s['last_prediction_timestamp'][:10]}"
        )
    header = "  {:<20} {:<10} {:>8} {:>8} {:>8} {:>10} {:>9} {:>8} {:>8}"
    lines.append(
        header.format(
            "model", "split", "ROC-AUC", "PR-AUC", "Brier", "threshold", "precision", "recall", "F1"
        )
    )
    for name, by_split in report["metrics"].items():
        for split_name, m in by_split.items():
            lines.append(
                header.format(
                    name,
                    split_name,
                    *(
                        "-" if m[k] is None else f"{m[k]:.4f}"
                        for k in ("roc_auc", "average_precision", "brier", "threshold")
                    ),
                    *(f"{m[k]:.4f}" for k in ("precision", "recall", "f1")),
                )
            )
    xgb = report["xgboost"]
    lines.append(f"XGBoost: max_depth {xgb['max_depth']}, best iteration {xgb['best_iteration']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate lead conversion models.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--track",
        action="store_true",
        help="log runs to MLflow and register both candidates (off by default)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=default_tracking_uri(),
        help=f"MLflow tracking server (default: $MLFLOW_TRACKING_URI or {DEFAULT_TRACKING_URI})",
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--registered-model", default=default_registered_model())
    args = parser.parse_args(argv)

    started = time.perf_counter()
    if args.track:
        # Checked before any training; MLflow is imported only when tracking is requested.
        try:
            require_tracking_server(args.tracking_uri)
        except TrackingUnavailableError as exc:
            print(f"Error: {exc} To train without MLflow, omit --track.", file=sys.stderr)
            return 2
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        from ml.training import tracking

    try:
        table = load_feature_table(args.features)
    except (FileNotFoundError, FeatureSchemaError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Training lead conversion models on {args.features} ({len(table.frame):,} rows)")
    result = train_candidates(table)
    write_outputs(result, args.output_dir)
    print(format_report(result.report))
    print(f"Wrote models and {REPORT_FILE} to {args.output_dir}")
    if args.track:
        print(f"Logging to MLflow at {args.tracking_uri} (experiment {args.experiment!r})")
        try:
            summary = tracking.log_candidates(
                result,
                table,
                tracking_uri=args.tracking_uri,
                experiment=args.experiment,
                registered_model=args.registered_model,
            )
        except (tracking.MlflowException, OSError) as exc:
            print(
                f"Error: MLflow logging failed ({exc}); local artifacts are in {args.output_dir}.",
                file=sys.stderr,
            )
            return 1
        print(
            f"Registered {args.registered_model} versions {summary.versions} "
            f"(training session run {summary.session_run_id}); no aliases assigned."
        )
    print(f"Total time: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
