"""Online feature computation for scoring one lead at time `at`.

Uses the same extraction queries and the same `compute_features` as the offline table, so
a lead scored at T gets exactly the features its training snapshot at T would have.
No snapshot eligibility and no label here: the caller decides which leads to score.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import Connection

from ml.features.feature_spec import (
    ACTIVE_FEATURES,
    LEAD_INPUT_COLUMNS,
    compute_features,
    to_records,
)
from ml.features.sources import load_lead_activities, load_leads


def online_features(
    lead: Mapping[str, Any], activities: pd.DataFrame, at: datetime
) -> dict[str, Any]:
    """Features (in `ACTIVE_FEATURES` order) for `lead` at `at`."""
    if at.tzinfo is None:
        raise ValueError("at must be timezone-aware")
    snapshot = pd.DataFrame(
        [
            {
                "lead_id": str(lead["lead_id"]),
                "prediction_timestamp": at,
                **{name: lead[name] for name in LEAD_INPUT_COLUMNS},
            }
        ]
    )
    snapshot["prediction_timestamp"] = pd.to_datetime(snapshot["prediction_timestamp"], utc=True)
    snapshot["created_at"] = pd.to_datetime(snapshot["created_at"], utc=True)
    record = to_records(compute_features(snapshot, activities))[0]
    return {name: record[name] for name in ACTIVE_FEATURES}


def fetch_online_features(
    connection: Connection, lead_id: UUID | str, at: datetime
) -> dict[str, Any]:
    """Load one lead and its activities known at `at`, then compute its features."""
    if at.tzinfo is None:
        raise ValueError("at must be timezone-aware")
    leads = load_leads(connection, lead_ids=[lead_id])
    if leads.empty:
        raise LookupError(f"lead {lead_id} not found")
    activities = load_lead_activities(connection, lead_ids=[lead_id], until=at)
    return online_features(leads.iloc[0].to_dict(), activities, at)
