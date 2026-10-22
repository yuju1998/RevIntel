"""User schemas. There is no public user-creation API; UserCreate is used by the CLI."""

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator, Field, SecretStr, StringConstraints

from app.models.enums import UserRole
from app.schemas.auth import MAX_PASSWORD_LENGTH
from app.schemas.common import RequestModel, ResponseModel, Text255, validate_email_syntax

MIN_PASSWORD_LENGTH = 12


class UserCreate(RequestModel):
    # Syntax-checked here; the user service stores the normalized (lowercase) form.
    email: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=320),
        AfterValidator(validate_email_syntax),
    ]
    full_name: Text255
    role: UserRole
    password: Annotated[
        SecretStr, Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    ]


class UserRead(ResponseModel):
    """Never includes password_hash."""

    id: uuid.UUID
    email: str
    full_name: str
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: datetime
