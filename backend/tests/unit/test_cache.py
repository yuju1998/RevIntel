"""Fail-open cache implementations (mocked Redis client, no server)."""

import io
import json
from unittest.mock import MagicMock

import pytest
import redis.exceptions
from redis import Redis

from app.core.cache import NullCache, RedisCache
from app.core.config import Settings
from app.core.logging import configure_logging
from app.core.redis import create_redis_client
from tests.unit.conftest import TEST_JWT_SECRET, UNREACHABLE_DATABASE_URL, UNREACHABLE_REDIS_URL


@pytest.fixture
def client() -> MagicMock:
    return MagicMock(spec=Redis)


@pytest.fixture
def logs() -> io.StringIO:
    buffer = io.StringIO()
    configure_logging(environment="test", stream=buffer)
    return buffer


def _records(logs: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in logs.getvalue().splitlines() if line.strip()]


def test_redis_cache_round_trip(client: MagicMock) -> None:
    client.get.return_value = b"payload"
    cache = RedisCache(client)

    cache.set("k", b"payload", 300)
    assert cache.get("k") == b"payload"
    cache.delete("k", "k2")

    client.set.assert_called_once_with("k", b"payload", ex=300)
    client.get.assert_called_once_with("k")
    client.delete.assert_called_once_with("k", "k2")


def test_missing_key_is_a_miss(client: MagicMock) -> None:
    client.get.return_value = None

    assert RedisCache(client).get("k") is None


def test_non_positive_ttl_is_not_written(client: MagicMock) -> None:
    RedisCache(client).set("k", b"v", 0)

    client.set.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        redis.exceptions.ConnectionError("Error 61 connecting to cache:6379 password=s3cret"),
        redis.exceptions.TimeoutError("Timeout reading from socket"),
        redis.exceptions.ResponseError("WRONGTYPE"),
        OSError("socket closed"),
    ],
)
def test_redis_errors_fail_open_and_log_type_only(
    client: MagicMock, logs: io.StringIO, error: Exception
) -> None:
    client.get.side_effect = error
    client.set.side_effect = error
    client.delete.side_effect = error
    cache = RedisCache(client)

    assert cache.get("revintel:v1:account:abc") is None
    cache.set("revintel:v1:account:abc", b"v", 300)
    cache.delete("revintel:v1:account:abc")

    records = _records(logs)
    assert [r["operation"] for r in records] == ["get", "set", "delete"]
    for record in records:
        assert record["event"] == "cache_unavailable"
        assert record["level"] == "warning"
        assert record["exception_type"] == type(error).__name__
        assert record["key"] == "revintel:v1:account:abc"
    assert "s3cret" not in logs.getvalue() and "socket closed" not in logs.getvalue()


def test_null_cache_never_stores() -> None:
    cache = NullCache()
    cache.set("k", b"v", 300)

    assert cache.get("k") is None
    cache.delete("k")


def test_cache_settings_defaults_and_bounds(settings: Settings) -> None:
    assert settings.account_cache_ttl_seconds == 300
    assert settings.cache_redis_timeout_seconds == 0.2
    for field, value in [
        ("account_cache_ttl_seconds", -1),
        ("account_cache_ttl_seconds", 86401),
        ("cache_redis_timeout_seconds", 0),
        ("cache_redis_timeout_seconds", 6),
    ]:
        with pytest.raises(ValueError):
            Settings(
                _env_file=None,
                database_url=UNREACHABLE_DATABASE_URL,
                redis_url=UNREACHABLE_REDIS_URL,
                jwt_secret=TEST_JWT_SECRET,
                **{field: value},
            )


def test_cache_client_uses_its_own_short_timeout(settings: Settings) -> None:
    default = create_redis_client(settings)
    cache_client = create_redis_client(settings, timeout_seconds=0.2)

    assert default.connection_pool.connection_kwargs["socket_timeout"] == 2.0
    assert cache_client.connection_pool.connection_kwargs["socket_timeout"] == 0.2
    assert cache_client.connection_pool.connection_kwargs["socket_connect_timeout"] == 0.2
