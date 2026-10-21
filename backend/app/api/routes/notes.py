"""Note endpoints."""

from fastapi import APIRouter, status

from app.api.deps import NoteServiceDep
from app.models.note import Note
from app.schemas.note import NoteCreate, NoteRead

router = APIRouter(prefix="/notes", tags=["notes"])


@router.post(
    "",
    response_model=NoteRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        422: {
            "description": (
                "Validation error (no parent, blank body, ...), a parent or author_id that "
                "does not exist, an inactive author, or references to different accounts"
            )
        }
    },
)
def create_note(payload: NoteCreate, service: NoteServiceDep) -> Note:
    """account_id is derived from opportunity_id when omitted."""
    return service.create_note(payload)
