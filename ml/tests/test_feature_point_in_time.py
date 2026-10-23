"""Exact window, availability, label and snapshot-policy boundaries."""

from datetime import timedelta

import pandas as pd
import pytest

from app.models.enums import ActivityDirection, ActivityType, LeadStatus
from ml.features.feature_spec import LABEL, to_records
from ml.features.snapshots import SnapshotPolicy, build_snapshots
from ml.features.sources import leads_frame
from ml.features.validation import FeatureValidationError, validate_features
from ml.tests.conftest import FeatureInputs
from ml.tests.feature_helpers import (
    CREATED,
    DAY,
    T,
    activity_row,
    features_at,
    lead_row,
    snapshot_labels,
)

US = timedelta(microseconds=1)

# --- feature windows --------------------------------------------------------------------------


def test_window_bounds_are_left_exclusive_and_right_inclusive() -> None:
    features = features_at(
        [
            activity_row(T),  # in 7d and 30d
            activity_row(T - 7 * DAY),  # 30d only
            activity_row(T - 30 * DAY),  # neither window
            activity_row(T + US),  # after T: never
        ]
    )

    assert features["activity_count_7d"] == 1
    assert features["activity_count_30d"] == 2
    assert features["days_since_last_activity"] == 0


def test_typed_counts() -> None:
    features = features_at(
        [
            activity_row(T - DAY, activity_type=ActivityType.EMAIL),
            activity_row(T - DAY, activity_type=ActivityType.MEETING),
            activity_row(T - DAY, activity_type=ActivityType.DEMO),
            activity_row(T - DAY, direction=ActivityDirection.INBOUND),
        ]
    )

    assert features["email_count_30d"] == 2
    assert features["meeting_count_30d"] == 1  # DEMO is not a meeting
    assert features["inbound_response_count_30d"] == 1


def test_active_days_use_utc_calendar_dates() -> None:
    midnight = T.replace(hour=0, minute=0, second=0) - 2 * DAY
    same_day = features_at([activity_row(midnight + timedelta(hours=1)), activity_row(midnight)])
    across = features_at([activity_row(midnight), activity_row(midnight - US)])

    assert same_day["active_days_30d"] == 1
    assert across["active_days_30d"] == 2


def test_lead_without_history_or_optional_attributes() -> None:
    lead = lead_row(industry=None, company_size=None, estimated_value=None)
    features = features_at([], lead=lead)

    assert features["activity_count_30d"] == features["active_days_30d"] == 0
    assert features["days_since_last_activity"] is None  # never, not "just now"
    assert features["industry"] is features["company_size"] is None
    assert features["estimated_value"] is None
    assert features["lead_age_days"] == 56


# --- availability (knowledge time) ------------------------------------------------------------


def test_activity_must_be_logged_by_t() -> None:
    logged_at_t = features_at([activity_row(T - DAY, created_at=T)])
    logged_after_t = features_at([activity_row(T - DAY, created_at=T + US)])

    assert logged_at_t["activity_count_30d"] == 1
    assert logged_after_t == features_at([])  # treated exactly like no activity


# --- label and eligibility --------------------------------------------------------------------

POLICY = SnapshotPolicy(as_of=CREATED + 400 * DAY, max_lead_age_days=177)


def test_label_window_is_left_exclusive_and_right_inclusive() -> None:
    def converted(at: timedelta) -> dict[int, bool]:
        lead = lead_row(
            status=LeadStatus.CONVERTED, converted_at=CREATED + at, updated_at=CREATED + at
        )
        return snapshot_labels(lead, POLICY)

    assert converted(37 * DAY)[1] is True  # converted_at == T + 30d
    assert converted(37 * DAY + timedelta(seconds=1))[1] is False  # eligible negative
    assert 2 not in converted(14 * DAY)  # converted_at == T: not active at T
    assert set(converted(14 * DAY)) == {1}


def test_disqualified_lead_is_never_a_negative_after_disqualification() -> None:
    lead = lead_row(status=LeadStatus.DISQUALIFIED, updated_at=CREATED + 21 * DAY)

    assert snapshot_labels(lead, POLICY) == {1: False, 2: False}  # T = 21d is excluded


def test_label_window_must_end_by_as_of() -> None:
    at_as_of = SnapshotPolicy(as_of=CREATED + 37 * DAY, max_lead_age_days=177)
    one_second_short = SnapshotPolicy(
        as_of=CREATED + 37 * DAY - timedelta(seconds=1), max_lead_age_days=177
    )

    assert snapshot_labels(lead_row(), at_as_of) == {1: False}
    assert snapshot_labels(lead_row(), one_second_short) == {}


# --- observation horizon (observation end = created_at + 177d here) ---------------------------


def test_open_lead_window_must_end_within_the_observation_horizon() -> None:
    exact = SnapshotPolicy(POLICY.as_of, 177, cadence=147 * DAY)  # T + 30d == observation end
    past = SnapshotPolicy(POLICY.as_of, 177, cadence=147 * DAY + timedelta(seconds=1))

    assert snapshot_labels(lead_row(), exact) == {1: False}
    assert snapshot_labels(lead_row(), past) == {}
    assert max(snapshot_labels(lead_row(), POLICY)) == 21  # k = 22 would be censored


@pytest.mark.parametrize(
    ("status", "terminal_day", "expected_k22"),
    [
        (LeadStatus.CONVERTED, 170, True),  # resolved before the horizon: keep, positive
        (LeadStatus.CONVERTED, 177, True),  # resolved exactly at the horizon: keep
        (LeadStatus.DISQUALIFIED, 170, False),  # resolved negative before the horizon: keep
        (LeadStatus.DISQUALIFIED, 177, False),
        (LeadStatus.CONVERTED, 180, None),  # after the horizon: never observable
        (LeadStatus.DISQUALIFIED, 180, None),
        (LeadStatus.CONVERTED, 190, None),  # after T + 30d: unresolved
        (LeadStatus.DISQUALIFIED, 154, None),  # terminal exactly at T: not active
    ],
)
def test_terminal_outcome_resolves_a_window_past_the_horizon(
    status: LeadStatus, terminal_day: int, expected_k22: bool | None
) -> None:
    # k = 22: T = created + 154d, window ends at created + 184d, past the 177d horizon.
    at = CREATED + terminal_day * DAY
    lead = lead_row(
        status=status,
        converted_at=at if status == LeadStatus.CONVERTED else None,
        updated_at=at,
    )

    assert snapshot_labels(lead, POLICY).get(22) is expected_k22


def test_terminal_after_the_horizon_is_counted_as_excluded() -> None:
    at = CREATED + 180 * DAY
    lead = lead_row(status=LeadStatus.CONVERTED, converted_at=at, updated_at=at)

    snapshots = build_snapshots(leads_frame([lead]), POLICY)

    # k = 22..25 (T = 154d..175d): conversion inside the window but after the 177d horizon.
    assert snapshots.terminal_after_horizon_excluded == 4
    assert not snapshots.frame["resolved_past_horizon"].any()


# --- generated data ---------------------------------------------------------------------------


def _reference(lead_activities: list[dict], t: pd.Timestamp) -> dict:
    known = [a for a in lead_activities if a["created_at"] <= t and a["occurred_at"] <= t]
    in_30d = [a for a in known if a["occurred_at"] > t - 30 * DAY]
    return {
        "activity_count_7d": sum(a["occurred_at"] > t - 7 * DAY for a in known),
        "activity_count_30d": len(in_30d),
        "email_count_30d": sum(a["activity_type"] == ActivityType.EMAIL for a in in_30d),
        "meeting_count_30d": sum(a["activity_type"] == ActivityType.MEETING for a in in_30d),
        "inbound_response_count_30d": sum(
            a["direction"] == ActivityDirection.INBOUND for a in in_30d
        ),
        "active_days_30d": len({a["occurred_at"].date() for a in in_30d}),
        "days_since_last_activity": (
            (t - max(a["occurred_at"] for a in known)) / DAY if known else None
        ),
    }


def test_vectorised_features_match_a_brute_force_reference(
    feature_inputs: FeatureInputs,
) -> None:
    by_lead: dict[str, list[dict]] = {}
    for row in feature_inputs.activities.to_dict("records"):
        by_lead.setdefault(row["lead_id"], []).append(row)
    not_yet_logged = 0

    for snapshot, features in zip(
        feature_inputs.snapshots.to_dict("records"),
        to_records(feature_inputs.features),
        strict=True,
    ):
        t = snapshot["prediction_timestamp"]
        lead_activities = by_lead.get(snapshot["lead_id"], [])
        not_yet_logged += sum(a["occurred_at"] <= t < a["created_at"] for a in lead_activities)
        expected = _reference(lead_activities, t)
        assert {name: features[name] for name in expected} == pytest.approx(expected)
        assert features["lead_age_days"] == (t - snapshot["created_at"]) / DAY

    assert not_yet_logged > 0  # the availability rule was exercised


def test_generated_snapshots_satisfy_the_policy(feature_inputs: FeatureInputs) -> None:
    table = feature_inputs.features.assign(**{LABEL: feature_inputs.snapshots[LABEL]})

    validate_features(table, feature_inputs.leads, feature_inputs.policy)
    assert (feature_inputs.snapshots["k"] >= 1).all()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("activity_count_7d", 99, "activity_count_7d > activity_count_30d"),
        ("days_since_last_activity", -1.0, "negative days_since_last_activity"),
        (LABEL, None, "label mismatch"),
    ],
)
def test_validation_rejects_violated_invariants(
    feature_inputs: FeatureInputs, column: str, value: object, message: str
) -> None:
    table = feature_inputs.features.assign(**{LABEL: feature_inputs.snapshots[LABEL]})
    first = table.index[0]
    table.loc[first, column] = (not table.loc[first, LABEL]) if column == LABEL else value

    with pytest.raises(FeatureValidationError, match=message):
        validate_features(table, feature_inputs.leads, feature_inputs.policy)
