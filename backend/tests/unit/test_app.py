import importlib
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from redis import Redis
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

import app.main
from app.core.config import Settings
from app.main import create_app


@pytest.mark.usefixtures("clean_env")
def test_importing_main_reads_no_settings_and_opens_no_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_engine = MagicMock()
    redis_from_url = MagicMock()
    monkeypatch.setattr("app.db.session.create_engine", create_engine)
    monkeypatch.setattr("app.core.redis.Redis.from_url", redis_from_url)

    # No DATABASE_URL/REDIS_URL and no `.env`: this would raise if import touched settings.
    module = importlib.reload(app.main)

    assert module.app is not None
    create_engine.assert_not_called()
    redis_from_url.assert_not_called()


@pytest.mark.usefixtures("clean_env")
def test_startup_fails_when_settings_are_invalid() -> None:
    application = create_app()

    with pytest.raises(ValidationError), TestClient(application):
        pass


def test_lifespan_stores_resources_on_app_state(settings: Settings) -> None:
    application = create_app(settings)

    with TestClient(application):
        assert application.state.settings is settings
        assert isinstance(application.state.engine, Engine)
        assert isinstance(application.state.session_factory, sessionmaker)
        assert isinstance(application.state.redis, Redis)


def test_lifespan_releases_resources_on_shutdown(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = MagicMock(spec=Engine)
    redis_client = MagicMock(spec=Redis)
    monkeypatch.setattr("app.main.create_db_engine", lambda _: engine)
    monkeypatch.setattr("app.main.create_redis_client", lambda _: redis_client)

    with TestClient(create_app(settings)):
        engine.dispose.assert_not_called()
        redis_client.close.assert_not_called()

    engine.dispose.assert_called_once()
    redis_client.close.assert_called_once()


def test_api_v1_prefix_is_reserved_for_business_endpoints(client: TestClient) -> None:
    assert client.get("/api/v1/health").status_code == 404
