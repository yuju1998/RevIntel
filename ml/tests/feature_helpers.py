"""Hand-built leads and activities with exact timestamps for feature tests."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from app.models.enums import ActivityDirection, ActivityType, LeadSource, LeadStatus
from ml.features.feature_spec import LABEL, compute_features, to_records
from ml.features.snapshots import SnapshotPolicy, build_snapshots
from ml.features.sources import activities_frame, leads_frame

LEAD_ID = "00000000-0000-4000-8000-000000000001"
CREATED = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
T = CREATED + timedelta(days=56)
DAY = timedelta(days=1)


def lead_row(**overrides: Any) -> dict[str, Any]:
    return {
        "id": LEAD_ID,
        "created_at": CREATED,
        "lead_source": LeadSource.WEB,
        "industry": "Software",
        "company_size": 50,
        "estimated_value": Decimal("1000.00"),
        "status": LeadStatus.WORKING,
        "converted_at": None,
        "updated_at": CREATED,
        **overrides,
    }


def activity_row(
    occurred_at: datetime,
    created_at: datetime | None = None,
    activity_type: ActivityType = ActivityType.EMAIL,
    direction: ActivityDirection = ActivityDirection.OUTBOUND,
) -> dict[str, Any]:
    return {
        "lead_id": LEAD_ID,
        "occurred_at": occurred_at,
        "created_at": occurred_at if created_at is None else created_at,
        "activity_type": activity_type,
        "direction": direction,
    }


def features_at(
    activities: list[dict[str, Any]], t: datetime = T, lead: dict[str, Any] | None = None
) -> dict[str, Any]:
    snapshot = leads_frame([lead or lead_row()]).assign(prediction_timestamp=pd.Timestamp(t))
    return to_records(compute_features(snapshot, activities_frame(activities)))[0]


def snapshot_labels(lead: dict[str, Any], policy: SnapshotPolicy) -> dict[int, bool]:
    """k -> label for every kept snapshot of one lead."""
    frame = build_snapshots(leads_frame([lead]), policy).frame
    return dict(zip(frame["k"].tolist(), frame[LABEL].tolist(), strict=True))
