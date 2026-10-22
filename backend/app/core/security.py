"""Password hashing (Argon2 via pwdlib) and JWT access tokens (PyJWT, HS256).

Nothing here logs. Errors never contain passwords, hashes or token contents.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import cache

import jwt
from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError
from pwdlib.hashers.argon2 import Argon2Hasher
from pydantic import SecretStr

from app.core.exceptions import AuthenticationError

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "revintel"
JWT_AUDIENCE = "revintel-api"
ACCESS_TOKEN_TYPE = "access"
REQUIRED_CLAIMS = ["sub", "iat", "exp", "iss", "aud", "type"]

_password_hash = PasswordHash((Argon2Hasher(),))


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str) -> tuple[bool, str | None]:
    """Return (valid, new_hash). new_hash is set when the stored hash should be upgraded."""
    try:
        return _password_hash.verify_and_update(password, password_hash)
    except UnknownHashError:
        # Not an Argon2 hash (e.g. a placeholder): the password can never match.
        return False, None


def verify_dummy_password(password: str) -> None:
    """Spend the same time as a real verification when no user matched (anti-enumeration)."""
    _password_hash.verify(password, _dummy_hash())


@cache
def _dummy_hash() -> str:
    return _password_hash.hash("revintel-timing-equalizer")


@dataclass(frozen=True, slots=True)
class AccessToken:
    token: str = field(repr=False)
    expires_in: int


def create_access_token(
    user_id: uuid.UUID,
    *,
    secret: SecretStr,
    expires_minutes: int,
    now: datetime | None = None,
) -> AccessToken:
    """Signed HS256 access token identifying the user by `sub` only.

    Role, email and is_active are deliberately absent: the user is reloaded from the
    database on every protected request.
    """
    issued_at = now or datetime.now(UTC)
    lifetime = timedelta(minutes=expires_minutes)
    claims = {
        "sub": str(user_id),
        "iat": issued_at,
        "exp": issued_at + lifetime,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "type": ACCESS_TOKEN_TYPE,
    }
    token = jwt.encode(claims, secret.get_secret_value(), algorithm=JWT_ALGORITHM)
    return AccessToken(token=token, expires_in=int(lifetime.total_seconds()))


def decode_access_token(token: str, *, secret: SecretStr) -> uuid.UUID:
    """Verify an access token and return its user id, or raise a generic AuthenticationError."""
    try:
        claims = jwt.decode(
            token,
            secret.get_secret_value(),
            # Only HS256: rejects "none" and algorithm-substitution tokens.
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={"require": REQUIRED_CLAIMS},
        )
        if claims["type"] != ACCESS_TOKEN_TYPE:
            raise ValueError("not an access token")
        return uuid.UUID(claims["sub"])
    except (jwt.PyJWTError, ValueError, TypeError):
        # `from None`: the original error is not chained into tracebacks.
        raise invalid_token() from None


def invalid_token() -> AuthenticationError:
    return AuthenticationError("Could not validate credentials")
