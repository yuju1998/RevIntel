"""Contact endpoints."""

from fastapi import APIRouter, status

from app.api.deps import ContactServiceDep
from app.models.contact import Contact
from app.schemas.contact import ContactCreate, ContactRead

router = APIRouter(prefix="/contacts", tags=["contacts"])


@router.post(
    "",
    response_model=ContactRead,
    status_code=status.HTTP_201_CREATED,
    responses={422: {"description": "Validation error, or account_id does not exist"}},
)
def create_contact(payload: ContactCreate, service: ContactServiceDep) -> Contact:
    return service.create_contact(payload)
