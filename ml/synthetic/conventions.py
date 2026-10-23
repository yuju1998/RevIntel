"""Conventions of the synthetic dataset that downstream ML code relies on.

The CRM schema has no lead status history. In the synthetic dataset, a lead's
`updated_at` is the timestamp of its most recent simulated state-changing update, so a
terminal lead's exit time can be recovered from (status, converted_at, updated_at).

This is a SYNTHETIC-DATASET CONVENTION, not a general production assumption: in real
data `updated_at` changes for unrelated edits. Keep every use of it behind
`lead_terminal_time` so it can be replaced by a real status-history source without
changing feature logic. Never use `updated_at` (or status/converted_at) as a feature.
"""

from datetime import datetime, timedelta

from app.models.enums import LeadStatus

TERMINAL_TIME_CONVENTION = (
    "Synthetic-dataset convention: CONVERTED leads left the pipeline at converted_at; "
    "DISQUALIFIED leads at updated_at (their disqualification timestamp); other leads are "
    "still open. Not valid for production data. Use only for snapshot eligibility, never "
    "as a model feature."
)

OBSERVATION_END_CONVENTION = (
    "Synthetic-dataset convention: the generator simulates each lead for at most "
    "max_lead_age_days after created_at, so nothing about a lead is observable after "
    "created_at + max_lead_age_days. A still-open lead is censored there, not a negative."
)


def lead_terminal_time(
    status: LeadStatus | str, converted_at: datetime | None, updated_at: datetime
) -> datetime | None:
    """When the lead stopped being an active lead, or None if it is still open.

    A prediction snapshot at time T is valid only while T < this value.
    """
    if status == LeadStatus.CONVERTED:
        return converted_at
    if status == LeadStatus.DISQUALIFIED:
        return updated_at
    return None


def lead_observation_end(created_at: datetime, max_lead_age_days: int) -> datetime:
    """The last moment the generator simulated this lead (see OBSERVATION_END_CONVENTION).

    A label window ending later is unobservable unless a terminal event at or before this
    time already resolved the outcome.
    """
    return created_at + timedelta(days=max_lead_age_days)
