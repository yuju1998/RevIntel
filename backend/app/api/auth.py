"""Authentication and role dependencies for protected endpoints."""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.deps import SessionDep
from app.core.config import Settings
from app.core.exceptions import PermissionDeniedError
from app.core.security import invalid_token
from app.models.enums import UserRole
from app.models.user import User
from app.repositories.user import UserRepository
from app.services.auth_service import AuthService

# auto_error=False so a missing/non-Bearer header goes through our own 401 response.
bearer_scheme = HTTPBearer(
    auto_error=False, description="Access token from POST /api/v1/auth/login"
)


def get_auth_service(request: Request, session: SessionDep) -> AuthService:
    settings: Settings = request.app.state.settings
    return AuthService(
        session,
        UserRepository(session),
        jwt_secret=settings.jwt_secret,
        access_token_expire_minutes=settings.access_token_expire_minutes,
    )


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


def get_current_user(
    request: Request,
    service: AuthServiceDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> User:
    """The authenticated, active user; 401 otherwise. Reloaded from the DB on every request."""
    if credentials is None:
        raise invalid_token()
    user = service.user_for_token(credentials.credentials)
    # For the access log (request.state is shared with the request middleware).
    request.state.user_id = str(user.id)
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]

# For routes that only need "must be logged in" and don't use the user object.
RequireUser = Depends(get_current_user)


def require_roles(*roles: UserRole) -> Callable[[User], User]:
    """Dependency factory: the current user must have one of `roles`, otherwise 403."""
    allowed = frozenset(roles)

    def dependency(user: CurrentUserDep) -> User:
        if user.role not in allowed:
            raise PermissionDeniedError("Insufficient permissions")
        return user

    return dependency


AdminUserDep = Annotated[User, Depends(require_roles(UserRole.ADMIN))]

# OpenAPI documentation for protected routes.
UNAUTHORIZED_RESPONSE = {
    401: {"description": "Missing, invalid or expired token, or inactive user"}
}
