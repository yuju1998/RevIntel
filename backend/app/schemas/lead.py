"""Lead API schemas."""

import uuid
from datetime import datetime
from decimal import Decimal

from app.models.enums import LeadSource, LeadStatus
from app.schemas.common import (
    CrmEmail,
    Money14,
    NonNegativeInt32,
    RequestModel,
    ResponseModel,
    Text255,
)


class LeadCreate(RequestModel):
    company_name: Text255
    contact_name: Text255
    email: CrmEmail | None = None
    lead_source: LeadSource
    industry: Text255 | None = None
    company_size: NonNegativeInt32 | None = None
    estimated_value: Money14 | None = None
    # CONVERTED is rejected by the service: conversion is a separate workflow.
    # converted_at is intentionally not accepted.
    status: LeadStatus = LeadStatus.NEW
    owner_id: uuid.UUID | None = None


class LeadRead(ResponseModel):
    id: uuid.UUID
    company_name: str
    contact_name: str
    email: str | None
    lead_source: LeadSource
    industry: str | None
    company_size: int | None
    estimated_value: Decimal | None
    status: LeadStatus
    owner_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime
    converted_at: datetime | None
