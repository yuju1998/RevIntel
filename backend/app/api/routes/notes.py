"""Note endpoints."""

from fastapi import APIRouter, status

from app.api.auth import UNAUTHORIZED_RESPONSE, CurrentUserDep
from app.api.deps import NoteServiceDep
from app.models.note import Note
from app.schemas.note import NoteCreate, NoteRead

router = APIRouter(prefix="/notes", tags=["notes"])


@router.post(
    "",
    response_model=NoteRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        **UNAUTHORIZED_RESPONSE,
        422: {
            "description": (
                "Validation error (no parent, blank body, author_id supplied, ...), a parent "
                "that does not exist, or references to different accounts"
            )
        },
    },
)
def create_note(payload: NoteCreate, service: NoteServiceDep, current_user: CurrentUserDep) -> Note:
    """author_id is the authenticated user. account_id is derived from opportunity_id when
    omitted."""
    return service.create_note(payload, author_id=current_user.id)
