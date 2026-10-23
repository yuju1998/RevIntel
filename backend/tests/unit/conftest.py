from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from redis import Redis
from sqlalchemy import Engine

from app.api.deps import get_engine, get_redis
from app.core.config import Settings
from app.main import create_app
from tests.unit.helpers import TEST_JWT_SECRET, UNREACHABLE_DATABASE_URL, UNREACHABLE_REDIS_URL

SETTINGS_ENV_VARS = (
    "APP_NAME",
    "ENVIRONMENT",
    "DATABASE_URL",
    "DB_POOL_SIZE",
    "DB_MAX_OVERFLOW",
    "DB_POOL_TIMEOUT_SECONDS",
    "DB_CONNECT_TIMEOUT_SECONDS",
    "REDIS_URL",
    "REDIS_SOCKET_TIMEOUT_SECONDS",
    "JWT_SECRET",
    "JWT_ALGORITHM",
    "ACCESS_TOKEN_EXPIRE_MINUTES",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Remove settings env vars and run from a directory without a `.env` file."""
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=UNREACHABLE_DATABASE_URL,
        redis_url=UNREACHABLE_REDIS_URL,
        jwt_secret=TEST_JWT_SECRET,
    )


@pytest.fixture
def fake_engine() -> MagicMock:
    return MagicMock(spec=Engine)


@pytest.fixture
def fake_redis() -> MagicMock:
    client = MagicMock(spec=Redis)
    client.ping.return_value = True
    return client


@pytest.fixture
def client(
    settings: Settings, fake_engine: MagicMock, fake_redis: MagicMock
) -> Iterator[TestClient]:
    app = create_app(settings)
    app.dependency_overrides[get_engine] = lambda: fake_engine
    app.dependency_overrides[get_redis] = lambda: fake_redis
    with TestClient(app) as test_client:
        yield test_client
