"""Shared request/response building blocks for CRM schemas."""

from decimal import Decimal
from typing import Annotated

from email_validator import EmailNotValidError, validate_email
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

INT32_MAX = 2_147_483_647


class RequestModel(BaseModel):
    # Unknown fields are rejected rather than silently ignored.
    model_config = ConfigDict(extra="forbid")


class ResponseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


def validate_email_syntax(value: str) -> str:
    """Validate email syntax but keep the caller's spelling and casing.

    Contact and lead emails are stored as entered (docs/DATABASE.md), so the
    normalized form returned by email-validator is deliberately discarded.
    """
    try:
        validate_email(value, check_deliverability=False)
    except EmailNotValidError as exc:
        raise ValueError("value is not a valid email address") from exc
    return value


# Short text is trimmed; if provided it must be non-blank. Lengths match the DB columns.
Text255 = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
Phone = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)]
Website = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]

# Contact/lead email: trimmed, syntax-checked, otherwise stored as entered.
# Not for users.email, which UserService normalizes to lowercase.
CrmEmail = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=320),
    AfterValidator(validate_email_syntax),
]



def _require_non_whitespace(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace text")
    return value


# Long free-form text (TEXT columns): must contain non-whitespace, but is stored exactly as
# submitted so formatting (leading indentation, line breaks) is preserved.
NonBlankText = Annotated[str, AfterValidator(_require_non_whitespace)]

NonNegativeInt32 = Annotated[int, Field(ge=0, le=INT32_MAX)]

CENTS = Decimal("0.01")


def _to_cents(value: Decimal) -> Decimal:
    # Never rounds (more than 2 decimal places is rejected first); it only fixes the
    # scale so responses match what NUMERIC(p,2) returns, e.g. 25000 -> "25000.00".
    return value.quantize(CENTS)


# USD amounts. Requests accept JSON numbers or strings; responses serialize as strings.
Money14 = Annotated[
    Decimal, Field(ge=0, max_digits=14, decimal_places=2), AfterValidator(_to_cents)
]
Money16 = Annotated[
    Decimal, Field(ge=0, max_digits=16, decimal_places=2), AfterValidator(_to_cents)
]


class Page[T](ResponseModel):
    items: list[T]
    total: int = Field(description="Number of rows matching the filters.")
    limit: int
    offset: int
