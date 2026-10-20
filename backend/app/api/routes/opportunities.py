"""Opportunity endpoints."""

import uuid

from fastapi import APIRouter, status

from app.api.deps import OpportunityServiceDep
from app.models.opportunity import Opportunity
from app.schemas.opportunity import OpportunityCreate, OpportunityRead, OpportunityUpdate

router = APIRouter(prefix="/opportunities", tags=["opportunities"])


@router.post(
    "",
    response_model=OpportunityRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        422: {"description": "Validation error, unknown account_id, or owner_id unknown/inactive"}
    },
)
def create_opportunity(payload: OpportunityCreate, service: OpportunityServiceDep) -> Opportunity:
    return service.create_opportunity(payload)


@router.patch(
    "/{opportunity_id}",
    response_model=OpportunityRead,
    responses={
        404: {"description": "Opportunity not found"},
        422: {"description": "Validation error, or owner_id unknown/inactive"},
    },
)
def update_opportunity(
    opportunity_id: uuid.UUID, payload: OpportunityUpdate, service: OpportunityServiceDep
) -> Opportunity:
    """Partial update: omitted fields are unchanged; null clears nullable fields."""
    return service.update_opportunity(opportunity_id, payload)
