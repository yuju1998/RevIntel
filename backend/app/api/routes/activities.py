"""Activity endpoints."""

from fastapi import APIRouter, status

from app.api.deps import ActivityServiceDep
from app.models.activity import Activity
from app.schemas.activity import ActivityCreate, ActivityRead

router = APIRouter(prefix="/activities", tags=["activities"])


@router.post(
    "",
    response_model=ActivityRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        422: {
            "description": (
                "Validation error (no parent, naive or future occurred_at, ...), a parent or "
                "created_by_id that does not exist, an inactive created_by_id user, or "
                "references to different accounts"
            )
        }
    },
)
def create_activity(payload: ActivityCreate, service: ActivityServiceDep) -> Activity:
    """account_id is derived from opportunity_id/contact_id when omitted."""
    return service.create_activity(payload)
