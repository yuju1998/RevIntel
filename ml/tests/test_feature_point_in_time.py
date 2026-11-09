"""Exact window, availability, label and snapshot-policy boundaries."""

from datetime import timedelta
from typing import Any

import pandas as pd
import pytest

from app.models.enums import ActivityDirection, ActivityType, LeadStatus
from ml.features.feature_spec import LABEL, compute_features, to_records
from ml.features.snapshots import SnapshotPolicy, build_snapshots, lead_timeline
from ml.features.sources import activities_frame, leads_frame
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

TERMINAL_STATUSES = (LeadStatus.CONVERTED, LeadStatus.DISQUALIFIED)


def _terminal_lead(status: LeadStatus, at: timedelta) -> dict[str, Any]:
    return lead_row(
        status=status,
        converted_at=CREATED + at if status == LeadStatus.CONVERTED else None,
        updated_at=CREATED + at,
    )


def test_open_lead_window_must_end_within_the_observation_horizon() -> None:
    exact = SnapshotPolicy(POLICY.as_of, 177, cadence=147 * DAY)  # T + 30d == observation end
    past = SnapshotPolicy(POLICY.as_of, 177, cadence=147 * DAY + timedelta(seconds=1))

    assert snapshot_labels(lead_row(), exact) == {1: False}
    assert snapshot_labels(lead_row(), past) == {}
    assert max(snapshot_labels(lead_row(), POLICY)) == 21  # k = 22 would be censored


@pytest.mark.parametrize(
    ("status", "expected_label"),
    [(LeadStatus.CONVERTED, True), (LeadStatus.DISQUALIFIED, False)],
)
def test_terminal_lead_window_must_end_within_the_observation_horizon(
    status: LeadStatus, expected_label: bool
) -> None:
    # The terminal event is inside the label window under both policies.
    lead = _terminal_lead(status, 148 * DAY)
    exact = SnapshotPolicy(POLICY.as_of, 177, cadence=147 * DAY)  # T + 30d == observation end
    past = SnapshotPolicy(POLICY.as_of, 177, cadence=147 * DAY + timedelta(seconds=1))

    assert snapshot_labels(lead, exact) == {1: expected_label}
    assert snapshot_labels(lead, past) == {}


@pytest.mark.parametrize("status", TERMINAL_STATUSES)
@pytest.mark.parametrize("terminal_day", [155, 170, 177, 180, 184])
def test_future_terminal_event_cannot_keep_a_censored_snapshot(
    status: LeadStatus, terminal_day: int
) -> None:
    # k = 22: T = created + 154d, window ends at created + 184d, past the 177d horizon.
    labels = snapshot_labels(_terminal_lead(status, terminal_day * DAY), POLICY)

    assert max(labels) == 21


@pytest.mark.parametrize("status", TERMINAL_STATUSES)
@pytest.mark.parametrize("terminal_day", [1, 7, 8, 147, 148, 154, 155, 170, 177, 180, 300])
def test_inclusion_depends_only_on_being_active_at_t(status: LeadStatus, terminal_day: int) -> None:
    open_lead = set(snapshot_labels(lead_row(), POLICY))

    kept = set(snapshot_labels(_terminal_lead(status, terminal_day * DAY), POLICY))

    assert kept == {k for k in open_lead if 7 * k < terminal_day}


@pytest.mark.parametrize(
    "lead",
    [
        lead_row(),
        _terminal_lead(LeadStatus.CONVERTED, 179 * DAY),
        _terminal_lead(LeadStatus.DISQUALIFIED, 179 * DAY),
    ],
    ids=["open", "converted", "disqualified"],
)
def test_oldest_snapshot_on_the_180_day_horizon_is_147_days(lead: dict[str, Any]) -> None:
    policy = SnapshotPolicy(POLICY.as_of, 180)

    assert max(snapshot_labels(lead, policy)) * 7 == 147


def test_label_boundaries_hold_at_the_last_eligible_snapshot() -> None:
    def converted(at: timedelta) -> dict[int, bool]:
        return snapshot_labels(_terminal_lead(LeadStatus.CONVERTED, at), POLICY)

    assert converted(177 * DAY)[21] is True  # converted_at == T + 30d == observation end
    assert converted(177 * DAY + timedelta(seconds=1))[21] is False  # after T + 30d
    assert 21 not in converted(147 * DAY)  # converted_at == T: not active at T


def test_policy_metadata_has_no_terminal_outcome_exception() -> None:
    assert POLICY.metadata()["rule"] == (
        "T + 30d <= as_of AND T < terminal time (if any) AND T + 30d <= observation end"
    )


def test_validation_rejects_a_snapshot_kept_for_its_future_outcome() -> None:
    leads = leads_frame([_terminal_lead(LeadStatus.CONVERTED, 170 * DAY)])
    snapshot = leads.assign(prediction_timestamp=pd.Timestamp(CREATED + 154 * DAY))
    table = compute_features(snapshot, activities_frame([])).assign(**{LABEL: True})

    with pytest.raises(FeatureValidationError, match="unobservable label window"):
        validate_features(table, leads, POLICY)


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


def test_generated_snapshots_are_no_older_than_the_horizon_allows(
    feature_inputs: FeatureInputs,
) -> None:
    assert feature_inputs.policy.max_lead_age_days == 180
    assert (feature_inputs.features["lead_age_days"] <= 147).all()


def test_generated_inclusion_matches_the_same_leads_without_outcomes(
    feature_inputs: FeatureInputs,
) -> None:
    leads, policy = feature_inputs.leads, feature_inputs.policy
    terminal_by_lead = lead_timeline(leads, policy)["terminal_at"].set_axis(leads["lead_id"])
    no_outcomes = leads.assign(
        status=LeadStatus.WORKING,
        converted_at=leads["converted_at"].where(leads["converted_at"].isna()),
        updated_at=leads["created_at"],
    )

    candidates = build_snapshots(no_outcomes, policy)
    terminal = candidates["lead_id"].map(terminal_by_lead)
    active = terminal.isna() | (candidates["prediction_timestamp"] < terminal)

    keys = ["lead_id", "prediction_timestamp"]
    pd.testing.assert_frame_equal(
        feature_inputs.snapshots[keys], candidates.loc[active, keys].reset_index(drop=True)
    )


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
