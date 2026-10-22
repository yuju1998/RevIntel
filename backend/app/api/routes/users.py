"""User endpoints."""

from fastapi import APIRouter

from app.api.auth import CurrentUserDep
from app.models.user import User
from app.schemas.user import UserRead

router = APIRouter(prefix="/users", tags=["users"])


@router.get(
    "/me",
    response_model=UserRead,
    responses={401: {"description": "Missing, invalid or expired token, or inactive user"}},
)
def read_current_user(current_user: CurrentUserDep) -> User:
    return current_user
