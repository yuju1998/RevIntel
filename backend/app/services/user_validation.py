"""Validation of optional user references (owner_id, created_by_id, author_id)."""

import uuid
from collections.abc import Callable
from functools import partial

from app.core.exceptions import BusinessRuleError, RelatedResourceNotFoundError
from app.repositories.user import UserRepository


def ensure_active_user(users: UserRepository, user_id: uuid.UUID | None, *, field: str) -> None:
    """A user reference is optional, but if given it must be an existing, active user."""
    if user_id is None:
        return
    user = users.get(user_id)
    if user is None:
        raise user_not_found(field)
    if not user.is_active:
        raise BusinessRuleError(f"{field} refers to an inactive user")


def user_not_found(field: str) -> RelatedResourceNotFoundError:
    return RelatedResourceNotFoundError(f"{field} does not refer to an existing user")


def user_not_found_for(field: str) -> Callable[[], RelatedResourceNotFoundError]:
    """Error factory for a user foreign-key violation, for `write_transaction` mappings."""
    return partial(user_not_found, field)
