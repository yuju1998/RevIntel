"""Conventions of the synthetic dataset that downstream ML code relies on.

The CRM schema has no lead status history. In the synthetic dataset, a lead's
`updated_at` is the timestamp of its most recent simulated state-changing update, so a
terminal lead's exit time can be recovered from (status, converted_at, updated_at).

This is a SYNTHETIC-DATASET CONVENTION, not a general production assumption: in real
data `updated_at` changes for unrelated edits. Keep every use of it behind
`lead_terminal_time` so it can be replaced by a real status-history source without
changing feature logic. Never use `updated_at` (or status/converted_at) as a feature.
"""

from datetime import datetime

from app.models.enums import LeadStatus

TERMINAL_TIME_CONVENTION = (
    "Synthetic-dataset convention: CONVERTED leads left the pipeline at converted_at; "
    "DISQUALIFIED leads at updated_at (their disqualification timestamp); other leads are "
    "still open. Not valid for production data. Use only for snapshot eligibility, never "
    "as a model feature."
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
