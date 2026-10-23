"""Authentication endpoints."""

from fastapi import APIRouter

from app.api.deps import AuthServiceDep
from app.api.sensitive_route import SensitiveBodyRoute
from app.schemas.auth import LoginRequest, TokenResponse

# Validation errors on these routes never echo request input (the password).
router = APIRouter(prefix="/auth", tags=["auth"], route_class=SensitiveBodyRoute)


@router.post(
    "/login",
    response_model=TokenResponse,
    responses={401: {"description": "Incorrect email or password"}},
)
def login(payload: LoginRequest, service: AuthServiceDep) -> TokenResponse:
    """Exchange email and password for a bearer access token.

    Unknown email, wrong password and inactive user all return the same 401.
    """
    token = service.login(payload.email, payload.password)
    return TokenResponse(access_token=token.token, expires_in=token.expires_in)
