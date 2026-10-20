"""Account API schemas."""

import uuid
from datetime import datetime
from decimal import Decimal

from app.schemas.common import (
    Money16,
    NonNegativeInt32,
    RequestModel,
    ResponseModel,
    Text255,
    Website,
)


class AccountCreate(RequestModel):
    name: Text255
    industry: Text255 | None = None
    employee_count: NonNegativeInt32 | None = None
    annual_revenue: Money16 | None = None
    website: Website | None = None
    owner_id: uuid.UUID | None = None


class AccountRead(ResponseModel):
    id: uuid.UUID
    name: str
    industry: str | None
    employee_count: int | None
    annual_revenue: Decimal | None
    website: str | None
    owner_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime
