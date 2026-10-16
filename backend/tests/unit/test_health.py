from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_does_not_touch_dependencies(
    client: TestClient, fake_engine: MagicMock, fake_redis: MagicMock
) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    fake_engine.connect.assert_not_called()
    fake_redis.ping.assert_not_called()


def test_health_returns_ok_when_dependencies_are_unreachable(settings: Settings) -> None:
    # Real (lazy) engine and Redis client pointed at closed ports, no overrides.
    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
