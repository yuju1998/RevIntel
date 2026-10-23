"""Feature specification: active/deferred features, input categories, fingerprint, guards."""

import dataclasses

import pandas as pd
import pytest

from ml.features.feature_spec import (
    ACTIVE_FEATURES,
    AVAILABILITY_INPUTS,
    FEATURES,
    FEATURES_BY_NAME,
    FORBIDDEN_FEATURE_INPUTS,
    MODEL_FEATURE_SOURCES,
    OUTPUT_COLUMNS,
    SNAPSHOT_CONTROL_INPUTS,
    compute_features,
    spec_fingerprint,
    spec_metadata,
)
from ml.features.sources import (
    UnsafeSourceError,
    activities_frame,
    leads_frame,
    require_source_database,
)
from ml.tests.conftest import FeatureInputs
from ml.tests.feature_helpers import T, lead_row


def test_active_feature_list_is_exact() -> None:
    assert ACTIVE_FEATURES == (
        "lead_source",
        "industry",
        "company_size",
        "estimated_value",
        "lead_age_days",
        "activity_count_7d",
        "activity_count_30d",
        "email_count_30d",
        "meeting_count_30d",
        "inbound_response_count_30d",
        "active_days_30d",
        "days_since_last_activity",
    )


def test_notes_count_is_defined_but_deferred() -> None:
    notes = FEATURES_BY_NAME["notes_count_30d"]

    assert not notes.enabled and notes.disabled_reason
    assert "notes_count_30d" not in ACTIVE_FEATURES
    assert spec_metadata()["deferred_features"] == [
        {"name": "notes_count_30d", "reason": notes.disabled_reason}
    ]


def test_input_categories_keep_control_columns_out_of_feature_values() -> None:
    assert SNAPSHOT_CONTROL_INPUTS <= FORBIDDEN_FEATURE_INPUTS
    assert AVAILABILITY_INPUTS <= FORBIDDEN_FEATURE_INPUTS
    for spec in FEATURES:
        assert spec.feature_sources <= MODEL_FEATURE_SOURCES, spec.name
        assert not spec.feature_sources & FORBIDDEN_FEATURE_INPUTS, spec.name
        assert spec.availability_inputs <= AVAILABILITY_INPUTS, spec.name


def test_fingerprint_is_stable_and_tracks_definition_changes() -> None:
    enabled_notes = tuple(dataclasses.replace(spec, enabled=True) for spec in FEATURES)
    wider_window = tuple(
        dataclasses.replace(spec, window=spec.window * 2) if spec.window else spec
        for spec in FEATURES
    )

    assert spec_fingerprint() == spec_fingerprint()
    assert spec_fingerprint(enabled_notes) != spec_fingerprint()
    assert spec_fingerprint(wider_window) != spec_fingerprint()


def test_output_columns_and_dtypes_follow_the_spec(feature_inputs: FeatureInputs) -> None:
    features = feature_inputs.features

    assert tuple(features.columns) == OUTPUT_COLUMNS
    assert isinstance(features["lead_source"].dtype, pd.CategoricalDtype)
    assert str(features["estimated_value"].dtype) == "decimal128(14, 2)[pyarrow]"
    assert str(features["company_size"].dtype) == "Int64"
    for name in ("activity_count_7d", "active_days_30d"):
        assert str(features[name].dtype) == "int32"
    assert str(features["days_since_last_activity"].dtype) == "float64"


def test_naive_timestamps_and_unknown_lead_source_are_rejected() -> None:
    lead = leads_frame([lead_row()])

    naive = lead.assign(prediction_timestamp=pd.Timestamp(T.replace(tzinfo=None)))
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_features(naive, activities_frame([]))
    unknown = lead.assign(prediction_timestamp=pd.Timestamp(T), lead_source="FAX")
    with pytest.raises(ValueError, match="unknown lead_source"):
        compute_features(unknown, activities_frame([]))


@pytest.mark.parametrize("database", ["revintel", "postgres", "revintel_ml_prod"])
def test_source_database_guard(database: str) -> None:
    with pytest.raises(UnsafeSourceError):
        require_source_database(f"postgresql+psycopg://u:p@localhost:5432/{database}")
    assert require_source_database("postgresql+psycopg://u:p@localhost/revintel_ml")


def test_build_cli_never_falls_back_to_database_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ml.features.build_features import main

    monkeypatch.delenv("SYNTHETIC_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/revintel")

    assert main([]) == 2
    assert "DATABASE_URL is deliberately ignored" in capsys.readouterr().err
