"""Account cache against real Redis (database 15) and revintel_test."""

import io
import json
import uuid
from collections.abc import Iterator

import pytest
import structlog
from fastapi.testclient import TestClient
from redis import Redis
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.cache import RedisCache
from app.core.config import Settings
from app.core.logging import configure_logging
from app.main import create_app
from app.models import Account
from app.services.account_cache import AccountCache
from tests.integration.conftest import require_test_redis_database
from tests.integration.factories import create_account

pytestmark = pytest.mark.integration

URL = "/api/v1/accounts"


@pytest.fixture
def test_redis(integration_settings: Settings) -> Iterator[Redis]:
    client = Redis.from_url(str(integration_settings.redis_url))
    require_test_redis_database(client)
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def account(db_session: Session) -> Account:
    return create_account(db_session, name="Cached Co", industry="Software")


def _settings(base: Settings, **overrides: object) -> Settings:
    return base.model_copy(update=overrides)


def test_miss_then_hit_and_invalidate(
    anon_client: TestClient, db_session: Session, test_redis: Redis, account: Account
) -> None:
    key = AccountCache.key(account.id)

    first = anon_client.get(f"{URL}/{account.id}")
    assert first.status_code == 200
    assert 0 < test_redis.ttl(key) <= 300
    assert json.loads(test_redis.get(key)) == first.json()

    # Change the row behind the cache's back: a hit keeps returning the cached value.
    db_session.execute(
        text("UPDATE accounts SET name = 'Renamed Co' WHERE id = :id"), {"id": account.id}
    )
    db_session.commit()
    second = anon_client.get(f"{URL}/{account.id}")
    assert second.status_code == 200
    assert second.json() == first.json()  # served from Redis, byte-identical body

    AccountCache(RedisCache(test_redis), ttl_seconds=300).invalidate(account.id)
    third = anon_client.get(f"{URL}/{account.id}")
    assert third.json()["name"] == "Renamed Co"
    assert json.loads(test_redis.get(key))["name"] == "Renamed Co"


def test_not_found_is_not_cached(
    anon_client: TestClient, test_redis: Redis, clean_tables: None
) -> None:
    missing = uuid.uuid4()

    assert anon_client.get(f"{URL}/{missing}").status_code == 404
    assert test_redis.exists(AccountCache.key(missing)) == 0


def test_corrupt_entry_falls_back_and_is_repaired(
    anon_client: TestClient, test_redis: Redis, account: Account
) -> None:
    key = AccountCache.key(account.id)
    test_redis.set(key, b"not json{", ex=300)

    response = anon_client.get(f"{URL}/{account.id}")

    assert response.status_code == 200
    assert response.json()["name"] == "Cached Co"
    assert json.loads(test_redis.get(key)) == response.json()


def test_unavailable_redis_falls_back_to_postgres(
    integration_settings: Settings,
    migrated_database: str,
    clean_tables: None,
    account: Account,
) -> None:
    settings = _settings(integration_settings, redis_url="redis://127.0.0.1:1/15")
    logs = io.StringIO()
    with TestClient(create_app(settings)) as client:
        configure_logging(environment="test", stream=logs)
        responses = [client.get(f"{URL}/{account.id}") for _ in range(2)]
    structlog.contextvars.clear_contextvars()

    assert [r.status_code for r in responses] == [200, 200]
    assert responses[0].json() == responses[1].json()
    assert responses[0].json()["name"] == "Cached Co"
    warnings = [
        json.loads(line)
        for line in logs.getvalue().splitlines()
        if '"cache_unavailable"' in line
    ]
    assert {w["operation"] for w in warnings} == {"get", "set"}
    assert all(w["exception_type"] == "ConnectionError" for w in warnings)
    assert all(w["request_id"] for w in warnings)


def test_zero_ttl_disables_caching(
    integration_settings: Settings,
    migrated_database: str,
    clean_tables: None,
    test_redis: Redis,
    account: Account,
) -> None:
    settings = _settings(integration_settings, account_cache_ttl_seconds=0)
    with TestClient(create_app(settings)) as client:
        assert client.get(f"{URL}/{account.id}").status_code == 200

    assert test_redis.exists(AccountCache.key(account.id)) == 0


def test_guard_refuses_redis_database_zero() -> None:
    client = Redis.from_url("redis://127.0.0.1:6379/0")
    with pytest.raises(pytest.fail.Exception, match="database 0"):
        require_test_redis_database(client)
    client.close()
