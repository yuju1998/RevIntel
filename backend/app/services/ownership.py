"""Validation of optional record owners (users) shared by CRM services."""

import uuid

from app.core.exceptions import BusinessRuleError, RelatedResourceNotFoundError
from app.repositories.user import UserRepository


def ensure_active_owner(users: UserRepository, owner_id: uuid.UUID | None) -> None:
    """An owner is optional, but if given it must be an existing, active user."""
    if owner_id is None:
        return
    user = users.get(owner_id)
    if user is None:
        raise owner_not_found()
    if not user.is_active:
        raise BusinessRuleError("owner_id refers to an inactive user")


def owner_not_found() -> RelatedResourceNotFoundError:
    return RelatedResourceNotFoundError("owner_id does not refer to an existing user")
