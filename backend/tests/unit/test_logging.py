"""Logging configuration, redaction and exception sanitizing (no HTTP)."""

import io
import json
import logging
from collections.abc import Iterator

import pytest
import structlog

from app.core.logging import (
    HANDLER_NAME,
    QUIET_LIBRARY_LOGGERS,
    REDACTED,
    configure_logging,
    scrub,
)

JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.c2lnbmF0dXJl"
ARGON2 = "$argon2id$v=19$m=65536,t=3,p=4$c2FsdA$aGFzaA"


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    buffer = io.StringIO()
    configure_logging(environment="test", log_level="DEBUG", stream=buffer)
    yield buffer
    structlog.contextvars.clear_contextvars()


def _json_lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_json_output_is_one_object_per_line(stream: io.StringIO) -> None:
    structlog.get_logger("app.test").info("something_happened", count=3)

    [record] = _json_lines(stream)
    assert record["event"] == "something_happened"
    assert record["count"] == 3
    assert record["level"] == "info"
    assert record["logger"] == "app.test"
    assert record["timestamp"].endswith("Z")


def test_stdlib_loggers_share_the_format_and_context(stream: io.StringIO) -> None:
    structlog.contextvars.bind_contextvars(request_id="req-123")
    logging.getLogger("some.library").warning("library said %s", "hi")

    [record] = _json_lines(stream)
    assert record == {
        "event": "library said hi",
        "request_id": "req-123",
        "level": "warning",
        "logger": "some.library",
        "timestamp": record["timestamp"],
    }


def test_local_environment_uses_human_readable_output() -> None:
    buffer = io.StringIO()
    configure_logging(environment="local", stream=buffer)

    structlog.get_logger("app.test").info("hello", answer=42)

    line = buffer.getvalue().strip()
    assert "hello" in line and "answer=42" in line
    assert not line.startswith("{")


@pytest.mark.parametrize("environment", ["test", "staging", "production"])
def test_non_local_environments_log_json(environment: str) -> None:
    buffer = io.StringIO()
    configure_logging(environment=environment, stream=buffer)

    structlog.get_logger().info("hello")

    assert json.loads(buffer.getvalue())["event"] == "hello"


def test_log_level_is_respected() -> None:
    buffer = io.StringIO()
    configure_logging(environment="test", log_level="INFO", stream=buffer)

    structlog.get_logger().debug("hidden")
    logging.getLogger("lib").debug("hidden too")
    structlog.get_logger().info("shown")

    assert [r["event"] for r in _json_lines(buffer)] == ["shown"]


def test_reconfiguring_replaces_only_our_handler() -> None:
    other = logging.NullHandler()
    logging.getLogger().addHandler(other)
    try:
        configure_logging(environment="test", stream=io.StringIO())
        configure_logging(environment="test", stream=io.StringIO())

        handlers = logging.getLogger().handlers
        assert [h for h in handlers if h.get_name() == HANDLER_NAME].__len__() == 1
        assert other in handlers
    finally:
        logging.getLogger().removeHandler(other)


def test_library_loggers_are_routed_or_quieted() -> None:
    configure_logging(environment="test", log_level="DEBUG", stream=io.StringIO())

    assert logging.getLogger("uvicorn").propagate is True
    assert logging.getLogger("uvicorn").handlers == []
    assert logging.getLogger("uvicorn.access").disabled is True
    for name in QUIET_LIBRARY_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING


# --- redaction ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "password",
        "new_password",
        "password_hash",
        "Authorization",
        "access_token",
        "jwt_secret",
        "api_key",
        "cookie",
        "database_url",
        "client_secret",
    ],
)
def test_sensitive_keys_are_redacted(stream: io.StringIO, key: str) -> None:
    structlog.get_logger().info("event", **{key: "super-sensitive-value"})

    [record] = _json_lines(stream)
    assert record[key] == REDACTED
    assert "super-sensitive-value" not in stream.getvalue()


def test_nested_sensitive_keys_are_redacted(stream: io.StringIO) -> None:
    structlog.get_logger().info(
        "event", payload={"email": "a@b.c", "password": "pw", "items": [{"token": "t"}]}
    )

    [record] = _json_lines(stream)
    assert record["payload"] == {
        "email": "a@b.c",
        "password": REDACTED,
        "items": [{"token": REDACTED}],
    }


@pytest.mark.parametrize(
    ("raw", "leak"),
    [
        (f"token {JWT} here", JWT),
        ("header was Bearer abc.def-ghi", "abc.def-ghi"),
        (f"hash {ARGON2}", ARGON2),
        ("postgresql+psycopg://revintel:hunter2@db:5432/revintel", "hunter2"),
        ("redis://:redispass@cache:6379/0", "redispass"),
    ],
)
def test_sensitive_patterns_are_scrubbed_everywhere(
    stream: io.StringIO, raw: str, leak: str
) -> None:
    structlog.get_logger().info(raw, detail=raw)
    logging.getLogger("third.party").warning("got %s", raw)

    assert leak not in stream.getvalue()
    assert len(_json_lines(stream)) == 2


def test_url_scrubbing_keeps_the_username_and_host() -> None:
    assert (
        scrub("postgresql+psycopg://revintel:hunter2@db:5432/revintel")
        == "postgresql+psycopg://revintel:***@db:5432/revintel"
    )


def test_non_sensitive_values_are_untouched() -> None:
    value = {"request_id": "abc", "path": "/api/v1/accounts", "status_code": 200}

    assert scrub(value) == value


# --- exceptions ---------------------------------------------------------------------------


def _raise_with_secrets() -> None:
    password = "local-variable-password"  # noqa: F841  # must never be rendered
    raise ValueError(f"failed for {JWT} with password=hunter2")


@pytest.mark.parametrize("environment", ["test", "local"])
def test_exceptions_log_type_and_frames_but_not_message_or_locals(environment: str) -> None:
    buffer = io.StringIO()
    configure_logging(environment=environment, stream=buffer)

    try:
        _raise_with_secrets()
    except ValueError:
        structlog.get_logger().exception("it_failed")
        logging.getLogger("third.party").exception("library failure")

    output = buffer.getvalue()
    assert "builtins.ValueError" in output
    assert "_raise_with_secrets" in output  # frames are kept
    assert "raise ValueError" not in output  # but not their source lines
    for leak in ("hunter2", JWT, "failed for", "local-variable-password"):
        assert leak not in output


def test_exception_fields_in_json(stream: io.StringIO) -> None:
    try:
        _raise_with_secrets()
    except ValueError as exc:
        structlog.get_logger().error("it_failed", exc_info=exc)

    [record] = _json_lines(stream)
    assert record["exception_type"] == "builtins.ValueError"
    assert record["exception"].startswith("Traceback (most recent call last):")
    assert record["exception"].endswith("builtins.ValueError")
    assert "exc_info" not in record
