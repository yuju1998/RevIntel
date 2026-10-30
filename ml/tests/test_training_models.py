"""Candidate training, saved pipelines, feature contract and threshold isolation."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from ml.evaluation.metrics import Threshold, constant_scores, evaluate, select_threshold
from ml.features.build_features import FeatureBuild, write_outputs
from ml.features.feature_spec import ACTIVE_FEATURES, LABEL
from ml.features.online_features import online_features
from ml.tests.conftest import FAST_XGBOOST, FeatureInputs
from ml.training.artifacts import METADATA_FILE, load_model, save_model
from ml.training.dataset import FeatureSchemaError, FeatureTable, load_feature_table
from ml.training.train import TrainingResult, train_candidates

MODELS = ("logistic_regression", "xgboost")


def _test_features(trained: TrainingResult, table: FeatureTable) -> pd.DataFrame:
    return table.frame.iloc[trained.split.test].loc[:, list(ACTIVE_FEATURES)]


# --- training ---------------------------------------------------------------------------------


def test_both_models_share_split_rows_and_feature_contract(trained: TrainingResult) -> None:
    logistic, xgb = (trained.models[name].metadata for name in MODELS)

    assert logistic["split"] == xgb["split"]  # includes per-split row-ID hashes
    assert logistic["source"] == xgb["source"]
    assert logistic["active_features"] == xgb["active_features"] == list(ACTIVE_FEATURES)
    for name in MODELS:
        pipeline = trained.models[name].pipeline
        assert list(pipeline.feature_names_in_) == list(ACTIVE_FEATURES)  # no lead_id or T
        assert isinstance(pipeline.named_steps["preprocess"], ColumnTransformer)


def test_probabilities_are_valid(trained: TrainingResult, training_table: FeatureTable) -> None:
    features = _test_features(trained, training_table)
    for name in MODELS:
        p = trained.models[name].predict_proba(features)
        assert p.shape == (len(features),)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_report_covers_every_model_and_baseline(trained: TrainingResult) -> None:
    metrics = trained.report["metrics"]

    assert set(metrics) == {"baseline", *MODELS}
    baseline = metrics["baseline"]["test"]
    assert baseline["roc_auc"] == 0.5
    assert baseline["average_precision"] == pytest.approx(baseline["positive_rate"], abs=1e-6)
    assert trained.report["xgboost"]["best_iteration"] >= 0


def test_test_labels_never_influence_thresholds(training_table: FeatureTable) -> None:
    frame = training_table.frame.copy()
    first = train_candidates(training_table, **FAST_XGBOOST)
    test_rows = first.split.test
    frame.loc[test_rows, LABEL] = ~frame.loc[test_rows, LABEL]
    flipped = train_candidates(dataclasses.replace(training_table, frame=frame), **FAST_XGBOOST)

    assert flipped.report["thresholds"] == first.report["thresholds"]
    for name in ("baseline", *MODELS):
        before, after = first.report["metrics"][name], flipped.report["metrics"][name]
        assert after["validation"] == before["validation"]
        assert after["test"] != before["test"]


def test_threshold_maximises_validation_f1_and_prefers_higher_on_ties() -> None:
    y = np.array([True, False, False, True, False])
    p = np.array([0.9, 0.7, 0.6, 0.5, 0.1])

    # F1 is 2/3 both at 0.9 (P=1, R=0.5) and at 0.5 (P=0.5, R=1); the higher one wins.
    assert select_threshold(y, p) == Threshold(value=0.9, validation_f1=round(2 / 3, 6))
    constant = constant_scores(0.4, 5)
    assert select_threshold(y, constant).value == 0.4  # baseline: predict all positive
    assert evaluate(y, constant, 0.4)["recall"] == 1.0


# --- saved pipeline ---------------------------------------------------------------------------


def test_saved_pipeline_predicts_raw_feature_rows(
    trained: TrainingResult, training_table: FeatureTable, tmp_path: Path
) -> None:
    features = _test_features(trained, training_table)
    for name in MODELS:
        save_model(trained.models[name], tmp_path / name)
        loaded = load_model(tmp_path / name)

        assert loaded.threshold == trained.models[name].threshold
        np.testing.assert_array_equal(
            loaded.predict_proba(features), trained.models[name].predict_proba(features)
        )
        metadata = json.loads((tmp_path / name / METADATA_FILE).read_text())
        assert {"dataset_sha256", "features_sha256"} <= set(metadata["source"])
        assert metadata["threshold"]["value"] == loaded.threshold


def test_unknown_categories_and_missing_values(
    trained: TrainingResult, training_table: FeatureTable
) -> None:
    rows = _test_features(trained, training_table).head(4).copy()
    rows["industry"] = pd.array(["Aerospace", None, "Software", "Aerospace"], dtype="string")
    rows["days_since_last_activity"] = np.nan
    rows["company_size"] = pd.array([None, 5, None, 7], dtype="Int64")
    rows["estimated_value"] = [None, *rows["estimated_value"].iloc[1:]]

    for name in MODELS:
        p = trained.models[name].predict_proba(rows)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_online_feature_dict_scores_like_the_offline_row(
    trained: TrainingResult, feature_inputs: FeatureInputs
) -> None:
    snapshot = feature_inputs.snapshots.iloc[trained.split.test[0]]
    lead = feature_inputs.leads.set_index("lead_id", drop=False).loc[snapshot["lead_id"]]
    online = online_features(
        lead.to_dict(), feature_inputs.activities, snapshot["prediction_timestamp"]
    )
    offline = feature_inputs.features.iloc[[trained.split.test[0]]]

    for name in MODELS:
        model = trained.models[name]
        assert model.predict_proba(pd.DataFrame([online]))[0] == pytest.approx(
            model.predict_proba(offline)[0], abs=1e-12
        )


# --- feature contract -------------------------------------------------------------------------


def test_missing_feature_is_rejected(trained: TrainingResult, training_table: FeatureTable) -> None:
    features = _test_features(trained, training_table).drop(columns="days_since_last_activity")

    with pytest.raises(FeatureSchemaError, match="days_since_last_activity"):
        trained.models["xgboost"].predict_proba(features)


def test_model_from_another_feature_spec_is_rejected(
    trained: TrainingResult, tmp_path: Path
) -> None:
    save_model(trained.models["logistic_regression"], tmp_path)
    metadata_path = tmp_path / METADATA_FILE
    metadata = json.loads(metadata_path.read_text())
    metadata_path.write_text(json.dumps({**metadata, "feature_spec_fingerprint": "f" * 64}))

    with pytest.raises(FeatureSchemaError, match="feature_spec_fingerprint"):
        load_model(tmp_path)


def _write_table(path: Path, frame: pd.DataFrame, metadata: dict) -> Path:
    write_outputs(FeatureBuild(frame, metadata, {}), path)
    return path


def test_feature_table_contract(training_table: FeatureTable, tmp_path: Path) -> None:
    frame, metadata = training_table.frame, training_table.metadata
    good = _write_table(tmp_path / "good.parquet", frame, metadata)
    assert len(load_feature_table(good).frame) == len(frame)

    cases = {
        "other_spec": (frame, {**metadata, "feature_spec_fingerprint": "f" * 64}),
        "missing_column": (frame.drop(columns="industry"), metadata),
        "extra_column": (frame.assign(owner_id="x"), metadata),
        "modified_rows": (frame.assign(activity_count_7d=frame["activity_count_7d"] + 1), metadata),
    }
    messages = {
        "other_spec": "feature specification",
        "missing_column": "columns",
        "extra_column": "columns",
        "modified_rows": "features_sha256",
    }
    for name, (bad_frame, bad_metadata) in cases.items():
        path = _write_table(tmp_path / f"{name}.parquet", bad_frame, bad_metadata)
        with pytest.raises(FeatureSchemaError, match=messages[name]):
            load_feature_table(path)
