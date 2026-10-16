from unittest.mock import MagicMock

import pytest
import redis.exceptions
import sqlalchemy.exc
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

SECRET_MESSAGE = 'password authentication failed for user "admin" password=s3cret host=db.internal'


def _postgres_error() -> Exception:
    return sqlalchemy.exc.OperationalError("SELECT 1", {}, Exception(SECRET_MESSAGE))


def _redis_error() -> Exception:
    return redis.exceptions.ConnectionError(SECRET_MESSAGE)


def test_ready_returns_200_when_all_dependencies_are_healthy(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert set(body["checks"]) == {"postgres", "redis"}
    for check in body["checks"].values():
        assert check["status"] == "ok"
        assert check["latency_ms"] >= 0


def test_ready_runs_select_1_and_ping(
    client: TestClient, fake_engine: MagicMock, fake_redis: MagicMock
) -> None:
    client.get("/ready")

    connection = fake_engine.connect.return_value.__enter__.return_value
    (statement,), _ = connection.execute.call_args
    assert str(statement) == "SELECT 1"
    fake_redis.ping.assert_called_once_with()


@pytest.mark.parametrize(
    ("postgres_fails", "redis_fails"),
    [(True, False), (False, True), (True, True)],
)
def test_ready_returns_503_when_any_dependency_fails(
    client: TestClient,
    fake_engine: MagicMock,
    fake_redis: MagicMock,
    postgres_fails: bool,
    redis_fails: bool,
) -> None:
    if postgres_fails:
        fake_engine.connect.side_effect = _postgres_error()
    if redis_fails:
        fake_redis.ping.side_effect = _redis_error()

    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["postgres"]["status"] == ("error" if postgres_fails else "ok")
    assert body["checks"]["redis"]["status"] == ("error" if redis_fails else "ok")


def test_ready_treats_timeouts_as_failures(
    client: TestClient, fake_engine: MagicMock, fake_redis: MagicMock
) -> None:
    fake_engine.connect.side_effect = sqlalchemy.exc.TimeoutError("pool timeout")
    fake_redis.ping.side_effect = redis.exceptions.TimeoutError("timed out")

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["postgres"]["status"] == "error"
    assert response.json()["checks"]["redis"]["status"] == "error"


def test_ready_does_not_expose_exception_details(
    client: TestClient, fake_engine: MagicMock, fake_redis: MagicMock
) -> None:
    fake_engine.connect.side_effect = _postgres_error()
    fake_redis.ping.side_effect = _redis_error()

    response = client.get("/ready")

    assert response.status_code == 503
    for leaked in ("s3cret", "admin", "db.internal", "OperationalError", "ConnectionError"):
        assert leaked not in response.text


def test_ready_returns_503_with_real_clients_when_dependencies_are_unreachable(
    settings: Settings,
) -> None:
    # Real engine and Redis client against closed local ports: no Docker needed.
    with TestClient(create_app(settings)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["postgres"]["status"] == "error"
    assert response.json()["checks"]["redis"]["status"] == "error"
    assert "s3cret" not in response.text
