import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

pytestmark = pytest.mark.integration

CLOSED_PORT = 1


def test_ready_returns_200_with_real_postgres_and_redis(integration_settings: Settings) -> None:
    with TestClient(create_app(integration_settings)) as client:
        response = client.get("/ready")

    assert response.status_code == 200, response.json()
    assert response.json()["status"] == "ready"


def test_ready_returns_503_when_postgres_is_unreachable(integration_settings: Settings) -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=f"postgresql+psycopg://revintel:revintel@127.0.0.1:{CLOSED_PORT}/revintel",
        redis_url=str(integration_settings.redis_url),
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["postgres"]["status"] == "error"
    assert checks["redis"]["status"] == "ok"


def test_ready_returns_503_when_redis_is_unreachable(integration_settings: Settings) -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=str(integration_settings.database_url),
        redis_url=f"redis://127.0.0.1:{CLOSED_PORT}/0",
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["postgres"]["status"] == "ok"
    assert checks["redis"]["status"] == "error"
