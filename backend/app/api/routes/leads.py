"""Lead endpoints."""

import uuid

from fastapi import APIRouter, status

from app.api.auth import UNAUTHORIZED_RESPONSE, RequireUser
from app.api.deps import ActivityServiceDep, LeadScoringServiceDep, LeadServiceDep
from app.api.pagination import PageParamsDep
from app.models.activity import Activity
from app.models.enums import LeadSource, LeadStatus
from app.models.lead import Lead
from app.models.lead_prediction import LeadPrediction
from app.schemas.activity import ActivityRead
from app.schemas.common import Page
from app.schemas.lead import LeadCreate, LeadRead
from app.schemas.lead_score import LeadScoreRead
from app.services.pagination import PageResult

router = APIRouter(prefix="/leads", tags=["leads"])


@router.post(
    "",
    response_model=LeadRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[RequireUser],
    responses={
        **UNAUTHORIZED_RESPONSE,
        422: {
            "description": "Validation error, status CONVERTED, or owner_id unknown/inactive"
        },
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


@router.get(
    "/{lead_id}/activities",
    response_model=Page[ActivityRead],
    responses={404: {"description": "Lead not found"}},
)
def list_lead_activities(
    lead_id: uuid.UUID, service: ActivityServiceDep, page: PageParamsDep
) -> PageResult[Activity]:
    """Activities linked to the lead, newest first by occurred_at (ties by id)."""
    return service.list_lead_activities(lead_id, limit=page.limit, offset=page.offset)


SCORING_UNAVAILABLE_RESPONSE = {
    503: {"description": "Lead scoring is currently unavailable (model or MLflow problem)"}
}


@router.post(
    "/{lead_id}/scores",
    response_model=LeadScoreRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[RequireUser],
    responses={
        **UNAUTHORIZED_RESPONSE,
        404: {"description": "Lead not found"},
        **SCORING_UNAVAILABLE_RESPONSE,
    },
)
def score_lead(lead_id: uuid.UUID, service: LeadScoringServiceDep) -> LeadPrediction:
    """Score the lead now with the champion model and store the score (append-only)."""
    return service.score(lead_id)


@router.get(
    "/{lead_id}/scores/latest",
    response_model=LeadScoreRead,
    responses={404: {"description": "Lead not found, or the lead has no score yet"}},
)
def get_latest_lead_score(lead_id: uuid.UUID, service: LeadScoringServiceDep) -> LeadPrediction:
    """The most recent stored score, exactly as stored (the model is not called)."""
    return service.latest(lead_id)
