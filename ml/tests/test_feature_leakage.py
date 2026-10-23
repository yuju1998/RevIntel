"""No feature may depend on information unavailable at prediction time T."""

from datetime import timedelta

import numpy as np
import pandas as pd

from app.models.enums import ActivityDirection, ActivityType, LeadStatus
from ml.features.feature_spec import (
    ACTIVE_FEATURES,
    LABEL,
    NUMERIC_FEATURES,
    OUTPUT_COLUMNS,
    compute_features,
)
from ml.features.validation import roc_auc
from ml.tests.conftest import FeatureInputs
from ml.tests.feature_helpers import DAY, T, activity_row, features_at

HOUR = timedelta(hours=1)


def test_output_has_no_control_or_deferred_columns(feature_inputs: FeatureInputs) -> None:
    columns = set(feature_inputs.features.columns)

    assert tuple(feature_inputs.features.columns) == OUTPUT_COLUMNS
    for forbidden in (
        "status",
        "converted_at",
        "updated_at",
        "owner_id",
        "created_at",
        "notes_count_30d",
        LABEL,
    ):
        assert forbidden not in columns


def test_future_activities_never_change_features(feature_inputs: FeatureInputs) -> None:
    leads = feature_inputs.snapshots["lead_id"].unique()
    after_every_t = feature_inputs.policy.as_of  # every T <= as_of - 30d
    future = pd.DataFrame(
        {
            "lead_id": np.repeat(leads, 3),
            "occurred_at": after_every_t,
            "created_at": after_every_t,
            "activity_type": [ActivityType.MEETING, ActivityType.DEMO, ActivityType.EMAIL]
            * len(leads),
            "direction": ActivityDirection.INBOUND,
        }
    )
    with_future = pd.concat([feature_inputs.activities, future], ignore_index=True)

    result = compute_features(feature_inputs.snapshots, with_future)

    pd.testing.assert_frame_equal(result, feature_inputs.features)


def test_created_at_is_only_an_availability_switch() -> None:
    occurred = T - 2 * DAY
    without = features_at([])
    not_yet_logged = features_at([activity_row(occurred, created_at=T + HOUR)])
    logged = features_at([activity_row(occurred, created_at=T - HOUR)])
    logged_earlier = features_at([activity_row(occurred, created_at=occurred)])

    assert not_yet_logged == without
    assert logged != without
    assert logged["activity_count_7d"] == 1
    assert logged["days_since_last_activity"] == 2  # from occurred_at, not created_at
    assert logged_earlier == logged  # the created_at value itself carries no signal
    assert {k: logged[k] for k in ("lead_source", "lead_age_days", "estimated_value")} == {
        k: without[k] for k in ("lead_source", "lead_age_days", "estimated_value")
    }


def test_snapshot_control_columns_never_change_features(feature_inputs: FeatureInputs) -> None:
    snapshots = feature_inputs.snapshots
    mutated = snapshots.assign(
        status=LeadStatus.CONVERTED.value,
        converted_at=snapshots["prediction_timestamp"] + DAY,
        updated_at=snapshots["prediction_timestamp"] + DAY,
        owner_id="00000000-0000-4000-8000-0000000000ff",
        **{LABEL: ~snapshots[LABEL]},
    )

    pd.testing.assert_frame_equal(
        compute_features(mutated, feature_inputs.activities), feature_inputs.features
    )


def test_no_single_feature_separates_the_label(feature_inputs: FeatureInputs) -> None:
    # Broad leakage canary on numeric/count features only (no target encoding of
    # categoricals); a sanity band, not a tuning target.
    labels = feature_inputs.snapshots[LABEL]
    for name in NUMERIC_FEATURES:
        auc = roc_auc(feature_inputs.features[name].astype("float64"), labels)
        assert auc is None or max(auc, 1 - auc) < 0.9, name
    assert set(NUMERIC_FEATURES) < set(ACTIVE_FEATURES)
