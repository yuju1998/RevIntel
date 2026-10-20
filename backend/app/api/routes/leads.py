"""Lead endpoints."""

import uuid

from fastapi import APIRouter, status

from app.api.deps import LeadServiceDep
from app.api.pagination import PageParamsDep
from app.models.enums import LeadSource, LeadStatus
from app.models.lead import Lead
from app.schemas.common import Page
from app.schemas.lead import LeadCreate, LeadRead
from app.services.pagination import PageResult

router = APIRouter(prefix="/leads", tags=["leads"])


@router.post(
    "",
    response_model=LeadRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        422: {
            "description": "Validation error, status CONVERTED, or owner_id unknown/inactive"
        }
    },
)
def create_lead(payload: LeadCreate, service: LeadServiceDep) -> Lead:
    return service.create_lead(payload)


@router.get("", response_model=Page[LeadRead])
def list_leads(
    service: LeadServiceDep,
    page: PageParamsDep,
    status: LeadStatus | None = None,
    lead_source: LeadSource | None = None,
) -> PageResult[Lead]:
    return service.list_leads(
        status=status, lead_source=lead_source, limit=page.limit, offset=page.offset
    )


@router.get(
    "/{lead_id}",
    response_model=LeadRead,
    responses={404: {"description": "Lead not found"}},
)
def get_lead(lead_id: uuid.UUID, service: LeadServiceDep) -> Lead:
    return service.get_lead(lead_id)
