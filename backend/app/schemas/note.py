"""Note API schemas."""

import uuid
from datetime import datetime
from typing import Self

from pydantic import model_validator

from app.schemas.common import NonBlankText, RequestModel, ResponseModel

PARENT_FIELDS = ("account_id", "lead_id", "opportunity_id")


class NoteCreate(RequestModel):
    account_id: uuid.UUID | None = None
    lead_id: uuid.UUID | None = None
    opportunity_id: uuid.UUID | None = None
    # author_id is not accepted: it is always the authenticated user.
    # Must contain non-whitespace text; stored as submitted.
    body: NonBlankText

    @model_validator(mode="after")
    def _require_a_parent(self) -> Self:
        if all(getattr(self, field) is None for field in PARENT_FIELDS):
            raise ValueError(f"at least one of {', '.join(PARENT_FIELDS)} is required")
        return self


class NoteRead(ResponseModel):
    id: uuid.UUID
    account_id: uuid.UUID | None
    lead_id: uuid.UUID | None
    opportunity_id: uuid.UUID | None
    author_id: uuid.UUID | None
    body: str
    created_at: datetime
    updated_at: datetime
