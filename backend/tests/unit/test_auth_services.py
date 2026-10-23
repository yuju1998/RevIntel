"""AuthService and UserService with fake repositories (no database)."""

import uuid
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr
from sqlalchemy.orm import Session

from app.core.exceptions import AuthenticationError, BusinessRuleError
from app.core.security import create_access_token, decode_access_token, hash_password
from app.models import User
from app.models.enums import UserRole
from app.repositories.user import UserRepository
from app.schemas.user import UserCreate
from app.services.auth_service import AuthService
from app.services.user_service import UserService
from tests.unit.helpers import integrity_error, make_user

SECRET = SecretStr("unit-test-jwt-secret-0123456789abcdef")
PASSWORD = "correct horse battery staple"
PASSWORD_HASH = hash_password(PASSWORD)


def _user(*, is_active: bool = True, password_hash: str = PASSWORD_HASH) -> User:
    # Users here have a real hash of PASSWORD unless a test overrides it.
    return make_user(is_active=is_active, password_hash=password_hash)




@pytest.fixture
def session() -> MagicMock:
    return MagicMock(spec=Session)


@pytest.fixture
def users() -> MagicMock:
    return MagicMock(spec=UserRepository)


@pytest.fixture
def auth(session: MagicMock, users: MagicMock) -> AuthService:
    return AuthService(session, users, jwt_secret=SECRET, access_token_expire_minutes=60)


def test_login_normalizes_email_and_issues_token(auth: AuthService, users: MagicMock) -> None:
    user = _user()
    users.get_by_email.return_value = user

    token = auth.login("  Ada@Example.COM ", SecretStr(PASSWORD))

    users.get_by_email.assert_called_once_with("ada@example.com", include_password_hash=True)
    assert decode_access_token(token.token, secret=SECRET) == user.id
    assert token.expires_in == 3600


@pytest.mark.parametrize(
    "case", ["unknown-email", "wrong-password", "inactive-user", "placeholder-hash"]
)
def test_login_failures_are_indistinguishable(
    auth: AuthService, users: MagicMock, session: MagicMock, case: str
) -> None:
    users.get_by_email.return_value = {
        "unknown-email": None,
        "wrong-password": _user(),
        "inactive-user": _user(is_active=False),
        "placeholder-hash": _user(password_hash="not-a-real-hash"),
    }[case]
    password = "wrong password" if case == "wrong-password" else PASSWORD

    with pytest.raises(AuthenticationError) as exc_info:
        auth.login("ada@example.com", SecretStr(password))

    assert exc_info.value.message == "Incorrect email or password"
    session.commit.assert_not_called()


def test_login_upgrades_outdated_hash(
    auth: AuthService, users: MagicMock, session: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _user()
    users.get_by_email.return_value = user
    monkeypatch.setattr(
        "app.services.auth_service.verify_password", lambda *_: (True, "$argon2id$upgraded")
    )

    auth.login("ada@example.com", SecretStr(PASSWORD))

    assert user.password_hash == "$argon2id$upgraded"
    session.commit.assert_called_once()


def test_user_for_token_returns_active_user(auth: AuthService, users: MagicMock) -> None:
    user = _user()
    users.get.return_value = user
    token = create_access_token(user.id, secret=SECRET, expires_minutes=5)

    assert auth.user_for_token(token.token) is user
    users.get.assert_called_once_with(user.id)


@pytest.mark.parametrize("stored", [None, _user(is_active=False)], ids=["deleted", "inactive"])
def test_user_for_token_rejects_missing_or_inactive_user(
    auth: AuthService, users: MagicMock, stored: User | None
) -> None:
    users.get.return_value = stored
    token = create_access_token(uuid.uuid4(), secret=SECRET, expires_minutes=5)

    with pytest.raises(AuthenticationError, match="Could not validate credentials"):
        auth.user_for_token(token.token)


def test_user_for_token_does_not_query_for_invalid_tokens(
    auth: AuthService, users: MagicMock
) -> None:
    with pytest.raises(AuthenticationError):
        auth.user_for_token("garbage")
    users.get.assert_not_called()


# --- UserService -------------------------------------------------------------------------


def _user_create(**fields: object) -> UserCreate:
    values: dict[str, object] = {
        "email": "  Grace.Hopper@Example.COM ",
        "full_name": "Grace Hopper",
        "role": "ADMIN",
        "password": PASSWORD,
    }
    return UserCreate(**{**values, **fields})


def test_create_user_normalizes_email_and_hashes_password(
    session: MagicMock, users: MagicMock
) -> None:
    users.get_by_email.return_value = None
    users.add.side_effect = lambda user: user

    user = UserService(session, users).create_user(_user_create())

    assert user.email == "grace.hopper@example.com"
    assert user.role == UserRole.ADMIN and user.is_active is True
    assert user.password_hash.startswith("$argon2id$")
    assert PASSWORD not in user.password_hash
    session.commit.assert_called_once()


def test_create_user_rejects_existing_email_case_insensitively(
    session: MagicMock, users: MagicMock
) -> None:
    users.get_by_email.return_value = _user()

    with pytest.raises(BusinessRuleError, match="already exists"):
        UserService(session, users).create_user(_user_create(email="ADA@example.com"))

    users.get_by_email.assert_called_once_with("ada@example.com")
    users.add.assert_not_called()


def test_create_user_maps_unique_violation_race(session: MagicMock, users: MagicMock) -> None:
    users.get_by_email.return_value = None
    users.add.side_effect = integrity_error("uq_users_email")

    with pytest.raises(BusinessRuleError, match="already exists"):
        UserService(session, users).create_user(_user_create())


@pytest.mark.parametrize(
    "fields",
    [
        {"email": "not-an-email"},
        {"password": "short"},
        {"password": "x" * 1025},
        {"role": "SUPERUSER"},
        {"full_name": "  "},
    ],
)
def test_user_create_validation(fields: dict) -> None:
    with pytest.raises(ValueError):
        _user_create(**fields)


def test_user_create_hides_password_in_repr() -> None:
    assert PASSWORD not in repr(_user_create())
