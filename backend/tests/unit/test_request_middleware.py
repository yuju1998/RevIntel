"""Request IDs, access logging and safe 500s over HTTP (fake repositories, no database)."""

import io
import json
import logging
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated
from unittest.mock import MagicMock

import pytest
import structlog
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.auth import get_auth_service, require_roles
from app.core.config import Settings
from app.core.logging import configure_logging
from app.core.security import create_access_token, hash_password
from app.main import create_app
from app.models import User
from app.models.enums import UserRole
from app.repositories.user import UserRepository
from app.services.auth_service import AuthService
from tests.unit.conftest import TEST_JWT_SECRET

PASSWORD = "correct horse battery staple"
SECRET = SecretStr(TEST_JWT_SECRET)
ACCESS_LOG_FIELDS = {
    "event",
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "level",
    "logger",
    "timestamp",
}

AdminDep = Annotated[User, Depends(require_roles(UserRole.ADMIN))]


def _user(role: UserRole = UserRole.SALES_REP) -> User:
    now = datetime.now(UTC)
    return User(
        id=uuid.uuid4(),
        email="rep@example.com",
        full_name="Rep",
        role=role,
        is_active=True,
        password_hash=hash_password(PASSWORD),
        created_at=now,
        updated_at=now,
    )


def _service_level_work() -> None:
    """Stands in for a service/repository call made deep inside a request."""
    structlog.get_logger("app.services.fake").info("service_work")
    logging.getLogger("app.repositories.fake").info("repository_work")


@pytest.fixture
def users() -> MagicMock:
    return MagicMock(spec=UserRepository)


@pytest.fixture
def app(settings: Settings, users: MagicMock) -> FastAPI:
    application = create_app(settings)
    application.dependency_overrides[get_auth_service] = lambda: AuthService(
        MagicMock(), users, jwt_secret=SECRET, access_token_expire_minutes=60
    )
    probe = APIRouter()

    @probe.get("/_test/work")
    def sync_work() -> dict[str, str]:
        _service_level_work()
        return {"ok": "yes"}

    @probe.get("/_test/boom")
    def boom() -> None:
        password = "local-variable-password"  # noqa: F841  # must never be logged
        raise RuntimeError("boom with password=hunter2")

    @probe.get("/_test/admin")
    def admin(user: AdminDep) -> dict[str, str]:
        return {"ok": "yes"}

    application.include_router(probe)
    return application


@pytest.fixture
def logs() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def client(app: FastAPI, logs: io.StringIO) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        # The lifespan configured logging for stdout; capture it instead.
        configure_logging(environment="test", log_level="INFO", stream=logs)
        yield test_client
    structlog.contextvars.clear_contextvars()


def _records(logs: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in logs.getvalue().splitlines() if line.strip()]


def _access_logs(logs: io.StringIO) -> list[dict]:
    return [r for r in _records(logs) if r["event"] == "http_request"]


def _bearer(user: User) -> dict[str, str]:
    token = create_access_token(user.id, secret=SECRET, expires_minutes=5)
    return {"Authorization": f"Bearer {token.token}"}


# --- X-Request-ID -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "kwargs", "status"),
    [
        ("GET", "/_test/work", {}, 200),
        ("GET", "/api/v1/users/me", {}, 401),
        ("POST", "/api/v1/accounts", {"json": {}}, 401),
        ("GET", "/no/such/route", {}, 404),
        ("GET", "/api/v1/leads", {"params": {"limit": 0}}, 422),
        ("POST", "/api/v1/auth/login", {"json": {"password": "x"}}, 422),
        ("GET", "/_test/boom", {}, 500),
    ],
)
def test_every_response_has_a_request_id_matching_the_access_log(
    client: TestClient, logs: io.StringIO, method: str, path: str, kwargs: dict, status: int
) -> None:
    response = client.request(method, path, **kwargs)

    assert response.status_code == status
    request_id = response.headers["X-Request-ID"]
    assert uuid.UUID(request_id).version == 4
    [access] = _access_logs(logs)
    assert access["request_id"] == request_id
    assert access["status_code"] == status


def test_403_response_has_request_id(
    client: TestClient, users: MagicMock, logs: io.StringIO
) -> None:
    user = _user(UserRole.SALES_REP)
    users.get.return_value = user

    response = client.get("/_test/admin", headers=_bearer(user))

    assert response.status_code == 403
    assert _access_logs(logs)[0]["request_id"] == response.headers["X-Request-ID"]


def test_request_ids_are_unique(client: TestClient) -> None:
    ids = {client.get("/_test/work").headers["X-Request-ID"] for _ in range(5)}

    assert len(ids) == 5


def test_incoming_request_id_is_ignored(client: TestClient, logs: io.StringIO) -> None:
    response = client.get("/_test/work", headers={"X-Request-ID": "attacker\nforged-line"})

    assert response.headers["X-Request-ID"] != "attacker\nforged-line"
    assert "forged-line" not in logs.getvalue()


# --- access log ---------------------------------------------------------------------------


def test_access_log_fields(client: TestClient, logs: io.StringIO) -> None:
    lead_id = uuid.uuid4()

    client.get(f"/api/v1/leads/{lead_id}/activities?limit=5&secret=hunter2")

    [access] = _access_logs(logs)
    assert set(access) == ACCESS_LOG_FIELDS | {"route"}
    assert access["method"] == "GET"
    assert access["path"] == f"/api/v1/leads/{lead_id}/activities"
    assert access["route"] == "/api/v1/leads/{lead_id}/activities"
    assert access["level"] == "error"  # unreachable DB in unit tests -> 500
    assert isinstance(access["duration_ms"], float) and access["duration_ms"] >= 0
    assert "hunter2" not in logs.getvalue() and "limit=5" not in logs.getvalue()


@pytest.mark.parametrize(
    ("path", "template"),
    [
        ("/api/v1/leads/leads/activities", "/api/v1/leads/{lead_id}/activities"),
        ("/api/v1/accounts/api", "/api/v1/accounts/{account_id}"),
    ],
)
def test_route_is_canonical_when_a_parameter_matches_a_fixed_segment(
    client: TestClient, logs: io.StringIO, path: str, template: str
) -> None:
    # Regression: the template must not be rebuilt from concrete parameter values.
    client.get(path)

    [access] = _access_logs(logs)
    assert access["path"] == path
    assert access["route"] == template


def test_unmatched_paths_have_no_route(client: TestClient, logs: io.StringIO) -> None:
    client.get("/no/such/route")

    [access] = _access_logs(logs)
    assert set(access) == ACCESS_LOG_FIELDS
    assert access["level"] == "info"


def test_authenticated_requests_log_user_id(
    client: TestClient, users: MagicMock, logs: io.StringIO
) -> None:
    user = _user()
    users.get.return_value = user

    client.get("/api/v1/users/me", headers=_bearer(user))
    client.get("/api/v1/users/me")

    authenticated, anonymous = _access_logs(logs)
    assert authenticated["user_id"] == str(user.id)
    assert "user_id" not in anonymous


def test_health_checks_are_logged_at_debug(
    client: TestClient, logs: io.StringIO
) -> None:
    client.get("/health")
    assert _access_logs(logs) == []

    configure_logging(environment="test", log_level="DEBUG", stream=logs)
    client.get("/health")
    [access] = _access_logs(logs)
    assert access["level"] == "debug" and access["path"] == "/health"


# --- request context propagation ---------------------------------------------------------


def test_request_id_reaches_service_and_stdlib_logs_in_sync_routes(
    client: TestClient, logs: io.StringIO
) -> None:
    response = client.get("/_test/work")

    request_id = response.headers["X-Request-ID"]
    by_event = {record["event"]: record for record in _records(logs)}
    assert by_event["service_work"]["request_id"] == request_id
    assert by_event["repository_work"]["request_id"] == request_id
    assert by_event["http_request"]["request_id"] == request_id


def test_context_is_cleared_between_and_after_requests(
    client: TestClient, logs: io.StringIO
) -> None:
    first = client.get("/_test/work").headers["X-Request-ID"]
    second = client.get("/_test/work").headers["X-Request-ID"]
    structlog.get_logger().info("outside_any_request")

    records = _records(logs)
    service_ids = [r["request_id"] for r in records if r["event"] == "service_work"]
    assert service_ids == [first, second]
    outside = next(r for r in records if r["event"] == "outside_any_request")
    assert "request_id" not in outside


# --- unhandled exceptions ---------------------------------------------------------------


def test_unhandled_exception_returns_generic_500(client: TestClient, logs: io.StringIO) -> None:
    response = client.get("/_test/boom")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error"}
    assert "boom" not in response.text and "hunter2" not in response.text

    request_id = response.headers["X-Request-ID"]
    [error] = [r for r in _records(logs) if r["event"] == "unhandled_exception"]
    assert error["request_id"] == request_id
    assert error["exception_type"] == "builtins.RuntimeError"
    assert "in boom" in error["exception"]  # frames are kept
    output = logs.getvalue()
    for leak in ("hunter2", "boom with", "local-variable-password"):
        assert leak not in output


def test_unhandled_exception_is_logged_once(client: TestClient, logs: io.StringIO) -> None:
    client.get("/_test/boom")

    events = [r["event"] for r in _records(logs)]
    assert events.count("unhandled_exception") == 1
    assert events.count("http_request") == 1


# --- auth protections are unchanged -------------------------------------------------------


def test_login_traffic_never_logs_passwords_or_tokens(
    client: TestClient, users: MagicMock, logs: io.StringIO
) -> None:
    user = _user()
    users.get_by_email.return_value = user
    users.get.return_value = user

    token = client.post(
        "/api/v1/auth/login", json={"email": "rep@example.com", "password": PASSWORD}
    ).json()["access_token"]
    client.post("/api/v1/auth/login", json={"email": "rep@example.com", "password": "wrong-pw"})
    malformed = client.post("/api/v1/auth/login", json={"password": PASSWORD})
    client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {token}"})

    # Sensitive-login protection still strips input from validation errors.
    assert PASSWORD not in malformed.text
    output = logs.getvalue()
    for secret in (PASSWORD, "wrong-pw", token, TEST_JWT_SECRET, user.password_hash):
        assert secret not in output
    assert [r["status_code"] for r in _access_logs(logs)] == [200, 401, 422, 200]


def test_401_still_has_www_authenticate(client: TestClient) -> None:
    response = client.get("/api/v1/users/me")

    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert "X-Request-ID" in response.headers
