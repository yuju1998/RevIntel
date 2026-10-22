"""Activity endpoints."""

from fastapi import APIRouter, status

from app.api.auth import UNAUTHORIZED_RESPONSE, CurrentUserDep
from app.api.deps import ActivityServiceDep
from app.models.activity import Activity
from app.schemas.activity import ActivityCreate, ActivityRead

router = APIRouter(prefix="/activities", tags=["activities"])


@router.post(
    "",
    response_model=ActivityRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        **UNAUTHORIZED_RESPONSE,
        422: {
            "description": (
                "Validation error (no parent, naive or future occurred_at, created_by_id "
                "supplied, ...), a parent that does not exist, or references to different "
                "accounts"
            )
        },
    },
)
def create_activity(
    payload: ActivityCreate, service: ActivityServiceDep, current_user: CurrentUserDep
) -> Activity:
    """created_by_id is the authenticated user. account_id is derived from
    opportunity_id/contact_id when omitted."""
    return service.create_activity(payload, created_by_id=current_user.id)
