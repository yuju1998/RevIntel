"""Helpers shared by unit tests: only things that were duplicated across test modules."""

import io
import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from pydantic import SecretStr
from sqlalchemy.exc import IntegrityError

from app.core.security import create_access_token
from app.models import User
from app.models.enums import UserRole

# Unit tests point at closed ports so they can never depend on Docker by accident.
UNREACHABLE_DATABASE_URL = "postgresql+psycopg://revintel:s3cret@127.0.0.1:1/revintel_test"
UNREACHABLE_REDIS_URL = "redis://:s3cret@127.0.0.1:1/0"
# Test-only signing key (>= 32 characters).
TEST_JWT_SECRET = "unit-test-jwt-secret-0123456789abcdef"

# Not an Argon2 hash: users with it can never log in.
FAKE_PASSWORD_HASH = "not-a-real-hash"


def make_user(
    role: UserRole = UserRole.SALES_REP,
    *,
    is_active: bool = True,
    password_hash: str = FAKE_PASSWORD_HASH,
    email: str = "rep@example.com",
) -> User:
    """Unsaved User with all response fields populated."""
    now = datetime.now(UTC)
    return User(
        id=uuid.uuid4(),
        email=email,
        full_name="Test User",
        role=role,
        is_active=is_active,
        password_hash=password_hash,
        created_at=now,
        updated_at=now,
    )


def bearer_headers(user: User, **token_options: object) -> dict[str, str]:
    """Authorization header with an access token for `user`, signed with TEST_JWT_SECRET."""
    token = create_access_token(
        user.id, secret=SecretStr(TEST_JWT_SECRET), expires_minutes=5, **token_options
    )
    return {"Authorization": f"Bearer {token.token}"}


def integrity_error(constraint: str | None) -> IntegrityError:
    """IntegrityError whose psycopg diagnostics name the violated constraint."""
    orig = Exception("violation")
    orig.diag = SimpleNamespace(constraint_name=constraint)  # type: ignore[attr-defined]
    return IntegrityError("INSERT ...", {}, orig)


def mock_repo(cls: type) -> MagicMock:
    """Spec'd repository mock whose add() returns the object it was given."""
    repo = MagicMock(spec=cls)
    repo.add.side_effect = lambda obj: obj
    return repo


def json_log_records(stream: io.StringIO) -> list[dict]:
    """Parse JSON log output (one object per line)."""
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
