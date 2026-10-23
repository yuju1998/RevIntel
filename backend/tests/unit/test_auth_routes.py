"""Auth routes, current-user dependency and role helper over HTTP (fake repository)."""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Annotated
from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
from httpx2 import Response
from pydantic import SecretStr

from app.api.auth import AdminUserDep, require_roles
from app.api.deps import get_auth_service
from app.core.config import Settings
from app.core.security import hash_password
from app.main import create_app
from app.models import User
from app.models.enums import UserRole
from app.repositories.user import UserRepository
from app.services.auth_service import AuthService
from tests.unit.helpers import TEST_JWT_SECRET, bearer_headers, make_user

PASSWORD = "correct horse battery staple"
PASSWORD_HASH = hash_password(PASSWORD)
SECRET = SecretStr(TEST_JWT_SECRET)

AnyRoleDep = Annotated[User, Depends(require_roles(UserRole.ADMIN, UserRole.SALES_REP))]

PROTECTED_MUTATIONS = [
    ("POST", "/api/v1/accounts"),
    ("POST", "/api/v1/contacts"),
    ("POST", "/api/v1/leads"),
    ("POST", "/api/v1/opportunities"),
    ("PATCH", f"/api/v1/opportunities/{uuid.uuid4()}"),
    ("POST", "/api/v1/activities"),
    ("POST", "/api/v1/notes"),
]







def _user(role: UserRole = UserRole.SALES_REP, *, is_active: bool = True) -> User:
    return make_user(role, is_active=is_active, password_hash=PASSWORD_HASH)


@pytest.fixture
def users() -> MagicMock:
    return MagicMock(spec=UserRepository)


@pytest.fixture
def app(settings: Settings, users: MagicMock) -> FastAPI:
    application = create_app(settings)
    application.dependency_overrides[get_auth_service] = lambda: AuthService(
        MagicMock(), users, jwt_secret=SECRET, access_token_expire_minutes=60
    )

    # Test-only routes exercising the role helper.
    probe = APIRouter()

    @probe.get("/_test/admin-only")
    def admin_only(user: AdminUserDep) -> dict[str, str]:
        return {"role": user.role}

    @probe.get("/_test/any-role")
    def any_role(user: AnyRoleDep) -> dict[str, str]:
        return {"role": user.role}

    application.include_router(probe)
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def _assert_401(response: Response) -> None:
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


# --- /auth/login --------------------------------------------------------------------------


def test_login_returns_bearer_token(client: TestClient, users: MagicMock) -> None:
    users.get_by_email.return_value = _user()

    response = client.post(
        "/api/v1/auth/login", json={"email": "Rep@Example.com", "password": PASSWORD}
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"access_token", "token_type", "expires_in"}
    assert body["token_type"] == "bearer" and body["expires_in"] == 3600


@pytest.mark.parametrize("case", ["unknown", "wrong-password", "inactive"])
def test_login_failures_return_the_same_401(
    client: TestClient, users: MagicMock, case: str
) -> None:
    users.get_by_email.return_value = {
        "unknown": None,
        "wrong-password": _user(),
        "inactive": _user(is_active=False),
    }[case]
    password = "nope" if case == "wrong-password" else PASSWORD

    response = client.post(
        "/api/v1/auth/login", json={"email": "rep@example.com", "password": password}
    )

    _assert_401(response)
    assert response.json() == {"detail": "Incorrect email or password"}


@pytest.mark.parametrize(
    "body",
    [
        {"password": "Sup3rSecretPassw0rd!"},
        {"email": "rep@example.com", "password": "Sup3rSecretPassw0rd!" * 60},
        {"email": "rep@example.com", "pw": "Sup3rSecretPassw0rd!"},
        {"email": "rep@example.com", "password": "Sup3rSecretPassw0rd!", "extra": 1},
        {"email": ["Sup3rSecretPassw0rd!"], "password": "x"},
    ],
    ids=["missing-email", "too-long", "misnamed-field", "extra-field", "wrong-type"],
)
def test_malformed_login_never_echoes_the_password(client: TestClient, body: dict) -> None:
    response = client.post("/api/v1/auth/login", json=body)

    assert response.status_code == 422
    assert "Sup3rSecretPassw0rd!" not in response.text
    for error in response.json()["detail"]:
        assert "input" not in error
        assert {"type", "loc", "msg"} <= set(error)


def test_invalid_json_login_returns_422_without_body(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login",
        content=b'{"email": "a@b.c", "password": "Sup3rSecretPassw0rd!"',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert "Sup3rSecretPassw0rd!" not in response.text


def test_other_routes_keep_standard_validation_errors(client: TestClient) -> None:
    response = client.get("/api/v1/leads", params={"limit": 0})

    assert response.status_code == 422
    assert "input" in response.json()["detail"][0]


# --- current user -------------------------------------------------------------------------


def test_users_me_returns_current_user_without_hash(client: TestClient, users: MagicMock) -> None:
    user = _user(UserRole.ADMIN)
    users.get.return_value = user

    response = client.get("/api/v1/users/me", headers=bearer_headers(user))

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(user.id)
    assert body["role"] == "ADMIN"
    assert set(body) == {
        "id",
        "email",
        "full_name",
        "role",
        "is_active",
        "created_at",
        "updated_at",
    }
    assert "password" not in response.text and "$argon2" not in response.text


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer"},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "Bearer not-a-jwt"},
        {"Authorization": "Token abc"},
    ],
    ids=["missing", "empty-bearer", "basic", "garbage", "wrong-scheme"],
)
def test_users_me_rejects_missing_or_invalid_credentials(
    client: TestClient, headers: dict
) -> None:
    response = client.get("/api/v1/users/me", headers=headers)

    _assert_401(response)
    assert response.json() == {"detail": "Could not validate credentials"}


@pytest.mark.parametrize("stored", ["deleted", "inactive"])
def test_token_of_deleted_or_inactive_user_is_rejected(
    client: TestClient, users: MagicMock, stored: str
) -> None:
    user = _user()
    users.get.return_value = None if stored == "deleted" else _user(is_active=False)

    _assert_401(client.get("/api/v1/users/me", headers=bearer_headers(user)))


def test_expired_token_is_rejected(client: TestClient, users: MagicMock) -> None:
    user = _user()
    users.get.return_value = user
    headers = bearer_headers(user, now=datetime.now(UTC) - timedelta(minutes=10))

    _assert_401(client.get("/api/v1/users/me", headers=headers))
    users.get.assert_not_called()


# --- protection of CRM endpoints ----------------------------------------------------------


@pytest.mark.parametrize(("method", "path"), PROTECTED_MUTATIONS)
def test_mutations_require_authentication_before_body_validation(
    client: TestClient, method: str, path: str
) -> None:
    # Invalid body and no token: authentication is checked first.
    _assert_401(client.request(method, path, json={"definitely": "invalid"}))


@pytest.mark.parametrize(("method", "path"), PROTECTED_MUTATIONS)
def test_mutations_reject_invalid_tokens(client: TestClient, method: str, path: str) -> None:
    response = client.request(
        method, path, json={}, headers={"Authorization": "Bearer not-a-jwt"}
    )

    _assert_401(response)


def test_openapi_marks_exactly_the_mutations_and_me_as_protected(app: FastAPI) -> None:
    schema = app.openapi()
    protected = {
        (method.upper(), path)
        for path, operations in schema["paths"].items()
        for method, operation in operations.items()
        if operation.get("security")
    }

    assert protected == {
        ("POST", "/api/v1/accounts"),
        ("POST", "/api/v1/contacts"),
        ("POST", "/api/v1/leads"),
        ("POST", "/api/v1/opportunities"),
        ("PATCH", "/api/v1/opportunities/{opportunity_id}"),
        ("POST", "/api/v1/activities"),
        ("POST", "/api/v1/notes"),
        ("GET", "/api/v1/users/me"),
        ("GET", "/_test/admin-only"),
        ("GET", "/_test/any-role"),
    }


# --- role helper --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "expected"), [(UserRole.ADMIN, 200), (UserRole.SALES_REP, 403)]
)
def test_admin_only_dependency(
    client: TestClient, users: MagicMock, role: UserRole, expected: int
) -> None:
    user = _user(role)
    users.get.return_value = user

    response = client.get("/_test/admin-only", headers=bearer_headers(user))

    assert response.status_code == expected
    if expected == 403:
        assert response.json() == {"detail": "Insufficient permissions"}


@pytest.mark.parametrize("role", list(UserRole))
def test_require_roles_allows_listed_roles(
    client: TestClient, users: MagicMock, role: UserRole
) -> None:
    user = _user(role)
    users.get.return_value = user

    assert client.get("/_test/any-role", headers=bearer_headers(user)).status_code == 200


def test_role_check_requires_authentication_first(client: TestClient) -> None:
    _assert_401(client.get("/_test/admin-only"))
