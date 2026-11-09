"""Snapshot policy and label for offline training data.

This is the only module that reads `status`, `converted_at` and `updated_at`; they decide
which (lead, T) snapshots exist and their label, and never reach feature computation.

For each lead, T = created_at + k * cadence (k = 1, 2, ...). A snapshot is kept when:
1. its label window is observed: T + 30d <= as_of;
2. the lead is active at T: T < terminal time (converted_at, or the disqualification
   time from the synthetic terminal-time convention); open leads are always active;
3. its whole label window is within the lead's observation horizon:
   T + 30d <= observation end. Past that the snapshot is censored, whatever happens to the
   lead later: inclusion never depends on a conversion or disqualification after T.
   Keeping a censored snapshot only when the lead later converts or is disqualified would
   select rows on their outcome and inflate the positive rate at those ages.

Label: converted_at in (T, T + 30d].
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ml.features.feature_spec import LABEL, LABEL_WINDOW
from ml.synthetic.conventions import lead_observation_end, lead_terminal_time

SNAPSHOT_CADENCE = timedelta(days=7)


@dataclass(frozen=True)
class SnapshotPolicy:
    as_of: datetime
    max_lead_age_days: int
    cadence: timedelta = SNAPSHOT_CADENCE
    label_window: timedelta = LABEL_WINDOW

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")

    def metadata(self) -> dict[str, object]:
        return {
            "cadence_days": self.cadence.days,
            "first_snapshot": "created_at + 1 * cadence",
            "as_of": self.as_of.isoformat(),
            "max_lead_age_days": self.max_lead_age_days,
            "rule": (
                "T + 30d <= as_of AND T < terminal time (if any) AND T + 30d <= observation end"
            ),
        }


def lead_timeline(leads: pd.DataFrame, policy: SnapshotPolicy) -> pd.DataFrame:
    """Per lead (same index as `leads`): terminal time (NaT while open) and observation end,
    both from the synthetic-dataset conventions."""
    terminal = [
        lead_terminal_time(status, None if pd.isna(converted) else converted, updated)
        for status, converted, updated in zip(
            leads["status"], leads["converted_at"], leads["updated_at"], strict=True
        )
    ]
    observation_end = [
        lead_observation_end(created, policy.max_lead_age_days) for created in leads["created_at"]
    ]
    return pd.DataFrame(
        {
            "terminal_at": pd.to_datetime(pd.Series(terminal, dtype=object), utc=True),
            "observation_end": pd.to_datetime(pd.Series(observation_end, dtype=object), utc=True),
        },
        index=leads.index,
    ).apply(lambda column: column.dt.as_unit("us"))


def build_snapshots(leads: pd.DataFrame, policy: SnapshotPolicy) -> pd.DataFrame:
    """Lead rows repeated once per eligible T, with `prediction_timestamp`, `k` and the label.

    Rows are sorted by (prediction_timestamp, lead_id).
    """
    leads = leads.reset_index(drop=True)
    timeline = lead_timeline(leads, policy)
    # Every kept snapshot has T < observation end, which bounds k.
    max_k = int(timedelta(days=policy.max_lead_age_days) / policy.cadence)
    k = np.tile(np.arange(1, max_k + 1), len(leads))
    row = np.repeat(np.arange(len(leads)), max_k)

    snaps = leads.iloc[row].reset_index(drop=True)
    t = snaps["created_at"] + pd.to_timedelta(k * policy.cadence)
    window_end = t + policy.label_window
    terminal = timeline["terminal_at"].iloc[row].reset_index(drop=True)
    observation_end = timeline["observation_end"].iloc[row].reset_index(drop=True)

    observed_window = window_end <= policy.as_of
    active = terminal.isna() | (t < terminal)
    in_horizon = window_end <= observation_end
    keep = observed_window & active & in_horizon

    converted = snaps["converted_at"]
    snaps["prediction_timestamp"] = t
    snaps["k"] = k
    snaps[LABEL] = (converted.notna() & (t < converted) & (converted <= window_end)).astype(bool)
    frame = snaps.loc[keep].sort_values(["prediction_timestamp", "lead_id"], kind="stable")
    return frame.reset_index(drop=True)
