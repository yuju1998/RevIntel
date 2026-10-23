"""Online features for (lead, T) equal the offline training row for the same snapshot."""

from datetime import datetime

import pytest

from ml.features.feature_spec import ACTIVE_FEATURES, to_records
from ml.features.online_features import online_features
from ml.tests.conftest import FeatureInputs
from ml.tests.feature_helpers import T, lead_row


def test_online_features_equal_offline_rows(feature_inputs: FeatureInputs) -> None:
    leads = feature_inputs.leads.set_index("lead_id", drop=False)
    activities = feature_inputs.activities
    offline = to_records(feature_inputs.features)

    for i in range(0, len(offline), 20):  # deterministic sample
        row = offline[i]
        t = row["prediction_timestamp"].to_pydatetime()
        # What the online loader would return: this lead's activities known at T.
        known = activities[
            (activities["lead_id"] == row["lead_id"])
            & (activities["occurred_at"] <= t)
            & (activities["created_at"] <= t)
        ]

        online = online_features(leads.loc[row["lead_id"]].to_dict(), known, t)

        assert list(online) == list(ACTIVE_FEATURES)
        assert online == {name: row[name] for name in ACTIVE_FEATURES}


def test_online_features_require_an_aware_timestamp(feature_inputs: FeatureInputs) -> None:
    lead = {**lead_row(), "lead_id": lead_row()["id"]}

    with pytest.raises(ValueError, match="timezone-aware"):
        online_features(lead, feature_inputs.activities.iloc[:0], datetime(2026, 3, 1))
    assert online_features(lead, feature_inputs.activities.iloc[:0], T)["activity_count_30d"] == 0
