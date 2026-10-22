"""Authentication API schemas."""

from typing import Annotated, Literal

from pydantic import Field, SecretStr, StringConstraints

from app.schemas.common import RequestModel, ResponseModel

MAX_PASSWORD_LENGTH = 1024


class LoginRequest(RequestModel):
    # Not syntax-checked: an unknown or malformed email simply fails authentication.
    email: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=320)]
    # SecretStr keeps the password out of reprs; the length cap bounds hashing work.
    password: Annotated[SecretStr, Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)]


class TokenResponse(ResponseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Seconds until the access token expires.")
