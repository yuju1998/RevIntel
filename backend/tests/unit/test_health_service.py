import logging
from unittest.mock import MagicMock

import pytest
import redis.exceptions
import sqlalchemy.exc

from app.services import health_service


def test_check_postgres_ok(fake_engine: MagicMock) -> None:
    result = health_service.check_postgres(fake_engine)

    assert result.status == "ok"
    assert result.latency_ms >= 0


def test_check_postgres_error(fake_engine: MagicMock) -> None:
    fake_engine.connect.side_effect = sqlalchemy.exc.OperationalError("SELECT 1", {}, Exception())

    assert health_service.check_postgres(fake_engine).status == "error"


def test_check_postgres_error_when_query_fails(fake_engine: MagicMock) -> None:
    connection = fake_engine.connect.return_value.__enter__.return_value
    connection.execute.side_effect = sqlalchemy.exc.DBAPIError("SELECT 1", {}, Exception())

    assert health_service.check_postgres(fake_engine).status == "error"


def test_check_redis_ok(fake_redis: MagicMock) -> None:
    assert health_service.check_redis(fake_redis).status == "ok"


def test_check_redis_error(fake_redis: MagicMock) -> None:
    fake_redis.ping.side_effect = redis.exceptions.ConnectionError()

    assert health_service.check_redis(fake_redis).status == "error"


@pytest.mark.parametrize(
    ("postgres_fails", "redis_fails", "expected"),
    [
        (False, False, "ready"),
        (True, False, "not_ready"),
        (False, True, "not_ready"),
        (True, True, "not_ready"),
    ],
)
def test_check_readiness_aggregates_results(
    fake_engine: MagicMock,
    fake_redis: MagicMock,
    postgres_fails: bool,
    redis_fails: bool,
    expected: str,
) -> None:
    if postgres_fails:
        fake_engine.connect.side_effect = RuntimeError()
    if redis_fails:
        fake_redis.ping.side_effect = RuntimeError()

    result = health_service.check_readiness(fake_engine, fake_redis)

    assert result.status == expected


def test_failed_check_logs_exception_type_without_message(
    fake_redis: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    fake_redis.ping.side_effect = redis.exceptions.AuthenticationError("password=s3cret")

    with caplog.at_level(logging.WARNING, logger=health_service.__name__):
        health_service.check_redis(fake_redis)

    assert "redis" in caplog.text
    assert "AuthenticationError" in caplog.text
    assert "s3cret" not in caplog.text
