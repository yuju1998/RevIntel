"""Request logging against PostgreSQL (revintel_test): correlation and no secrets."""

import io
import json
from collections.abc import Iterator

import pytest
import structlog
from fastapi.testclient import TestClient
from sqlalchemy import make_url, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import configure_logging
from app.main import create_app
from app.models import User
from tests.integration.conftest import AUTH_USER_PASSWORD

pytestmark = pytest.mark.integration


@pytest.fixture
def logs() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def client(
    integration_settings: Settings, migrated_database: str, auth_user: User, logs: io.StringIO
) -> Iterator[TestClient]:
    with TestClient(create_app(integration_settings)) as test_client:
        configure_logging(environment="test", log_level="INFO", stream=logs)
        yield test_client
    structlog.contextvars.clear_contextvars()


def _records(logs: io.StringIO) -> list[dict]:
    # Every line must be a JSON object outside the local environment.
    return [json.loads(line) for line in logs.getvalue().splitlines() if line.strip()]


def test_request_flow_is_correlated_and_free_of_secrets(
    client: TestClient,
    logs: io.StringIO,
    auth_user: User,
    db_session: Session,
    integration_settings: Settings,
) -> None:
    login = client.post(
        "/api/v1/auth/login", json={"email": "API-User@example.com", "password": AUTH_USER_PASSWORD}
    )
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    created = client.post("/api/v1/accounts", json={"name": "Logged Co"}, headers=headers)
    timeline = client.get(f"/api/v1/accounts/{created.json()['id']}/timeline")
    failed = client.post(
        "/api/v1/auth/login", json={"email": "api-user@example.com", "password": "wrong pw"}
    )

    responses = [login, created, timeline, failed]
    assert [r.status_code for r in responses] == [200, 201, 200, 401]
    access = [r for r in _records(logs) if r["event"] == "http_request"]
    assert [a["request_id"] for a in access] == [r.headers["X-Request-ID"] for r in responses]
    assert [a["status_code"] for a in access] == [200, 201, 200, 401]
    assert access[1]["user_id"] == str(auth_user.id)
    assert access[1]["route"] == "/api/v1/accounts"
    assert access[2]["route"] == "/api/v1/accounts/{account_id}/timeline"
    assert "user_id" not in access[0] and "user_id" not in access[3]

    stored_hash = db_session.scalar(select(User.password_hash).where(User.id == auth_user.id))
    db_password = make_url(str(integration_settings.database_url)).password
    output = logs.getvalue()
    for secret in (
        AUTH_USER_PASSWORD,
        "wrong pw",
        token,
        stored_hash,
        integration_settings.jwt_secret.get_secret_value(),
        f":{db_password}@",
    ):
        assert secret not in output
    assert "Authorization" not in output and "Bearer" not in output
