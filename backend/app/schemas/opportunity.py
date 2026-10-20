"""Opportunity API schemas."""

import uuid
from datetime import date, datetime
from decimal import Decimal

from pydantic import field_validator

from app.models.enums import OpportunityStage
from app.schemas.common import Money14, RequestModel, ResponseModel, Text255


class OpportunityCreate(RequestModel):
    account_id: uuid.UUID
    name: Text255
    stage: OpportunityStage
    amount: Money14 | None = None
    expected_close_date: date | None = None
    description: str | None = None
    owner_id: uuid.UUID | None = None


class OpportunityUpdate(RequestModel):
    """Partial update: omitted fields are left unchanged; explicit null clears a nullable field.

    account_id is not accepted: opportunities cannot move between accounts.
    """

    name: Text255 | None = None
    stage: OpportunityStage | None = None
    amount: Money14 | None = None
    expected_close_date: date | None = None
    description: str | None = None
    owner_id: uuid.UUID | None = None

    # Defaults are not validated, so this only fires for an explicit null.
    @field_validator("name", "stage")
    @classmethod
    def _reject_explicit_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("may be omitted but cannot be null")
        return value


class OpportunityRead(ResponseModel):
    id: uuid.UUID
    account_id: uuid.UUID
    name: str
    stage: OpportunityStage
    amount: Decimal | None
    expected_close_date: date | None
    description: str | None
    owner_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime
