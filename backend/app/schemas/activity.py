"""Activity API schemas."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Self

from pydantic import AwareDatetime, field_validator, model_validator

from app.models.enums import ActivityDirection, ActivityType
from app.schemas.common import NonBlankText, RequestModel, ResponseModel, Text255

PARENT_FIELDS = ("account_id", "lead_id", "opportunity_id", "contact_id")

# occurred_at is when the activity actually happened, so it cannot be in the future.
# A small allowance absorbs client/server clock skew.
MAX_FUTURE_SKEW = timedelta(minutes=5)


class ActivityCreate(RequestModel):
    account_id: uuid.UUID | None = None
    lead_id: uuid.UUID | None = None
    opportunity_id: uuid.UUID | None = None
    contact_id: uuid.UUID | None = None
    # created_by_id is not accepted: it is always the authenticated user.
    activity_type: ActivityType
    direction: ActivityDirection
    subject: Text255 | None = None
    # If supplied, must contain non-whitespace text; stored as submitted.
    body: NonBlankText | None = None
    # Timezone-aware only: naive datetimes are rejected.
    occurred_at: AwareDatetime

    @field_validator("occurred_at")
    @classmethod
    def _not_in_the_future(cls, value: datetime) -> datetime:
        if value > datetime.now(UTC) + MAX_FUTURE_SKEW:
            raise ValueError("must not be more than 5 minutes in the future")
        return value

    @model_validator(mode="after")
    def _require_a_parent(self) -> Self:
        # At least one parent; several may be given at once.
        if all(getattr(self, field) is None for field in PARENT_FIELDS):
            raise ValueError(f"at least one of {', '.join(PARENT_FIELDS)} is required")
        return self


class ActivityRead(ResponseModel):
    id: uuid.UUID
    account_id: uuid.UUID | None
    lead_id: uuid.UUID | None
    opportunity_id: uuid.UUID | None
    contact_id: uuid.UUID | None
    created_by_id: uuid.UUID | None
    activity_type: ActivityType
    direction: ActivityDirection
    subject: str | None
    body: str | None
    occurred_at: datetime
    created_at: datetime
