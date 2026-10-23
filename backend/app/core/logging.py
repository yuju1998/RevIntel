"""Application logging: structlog on top of stdlib logging.

- One root handler formats both structlog and stdlib loggers (uvicorn, SQLAlchemy, ...),
  so every line has the same shape and carries the request-scoped `request_id`.
- `local` environment: human-readable console lines. Everywhere else: one JSON object per
  line on stdout.
- Secrets are redacted by key and by value pattern, and exceptions are reduced to their
  type plus stack frames (no message, no local variables), since messages can contain
  sensitive values.
"""

import logging
import re
import sys
import traceback
from collections.abc import Mapping
from typing import Any, TextIO

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

HANDLER_NAME = "revintel"
REDACTED = "[REDACTED]"

# Keys whose values are always redacted (matched case-insensitively as substrings).
SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "authorization",
    "cookie",
    "api_key",
    "apikey",
    "jwt",
    "credential",
)
SENSITIVE_KEYS = frozenset({"database_url", "redis_url", "dsn"})

# Values scrubbed wherever they appear in a string (messages, tracebacks, third-party logs).
SENSITIVE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # JWTs (header and payload are base64url JSON, so they start with "eyJ").
    (re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*"), REDACTED),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), f"Bearer {REDACTED}"),
    (re.compile(r"\$argon2[a-z]*\$[^\s'\",)]+"), REDACTED),
    # Credentials in URLs: scheme://user:password@host -> scheme://user:***@host
    # (the username may be empty, e.g. redis://:password@host).
    (re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://[^:/@\s]*):[^@/\s]+@"), r"\1:***@"),
)

# Stdlib loggers capped at WARNING whatever LOG_LEVEL is:
# - sqlalchemy.engine: SQL statements and result rows (rows can contain password hashes).
# - httpx/httpcore: outbound request lines include full URLs with query strings.
QUIET_LIBRARY_LOGGERS = ("sqlalchemy.engine", "httpx", "httpx2", "httpcore", "httpcore2")


def _is_sensitive_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    lowered = key.lower()
    return lowered in SENSITIVE_KEYS or any(part in lowered for part in SENSITIVE_KEY_PARTS)


def scrub(value: Any) -> Any:
    """Redact sensitive patterns in strings and sensitive keys in nested containers."""
    if isinstance(value, str):
        for pattern, replacement in SENSITIVE_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, Mapping):
        return {
            key: REDACTED if _is_sensitive_key(key) else scrub(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple | set):
        return type(value)(scrub(item) for item in value)
    return value


def redact_sensitive(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    for key in list(event_dict):
        if key.startswith("_"):  # structlog internals (e.g. _record)
            continue
        event_dict[key] = REDACTED if _is_sensitive_key(key) else scrub(event_dict[key])
    return event_dict


def sanitize_exception(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    """Replace exc_info with the exception type and its stack frames only.

    The exception message is dropped (it may contain secrets, SQL or user input). Frames
    are file/line/function only: no source lines and no local variables.
    """
    exc_info = event_dict.pop("exc_info", None)
    event_dict.pop("exception", None)
    if not exc_info:
        return event_dict
    if exc_info is True:
        exc_info = sys.exc_info()
    elif isinstance(exc_info, BaseException):
        exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
    exc_type, _, exc_traceback = exc_info
    if exc_type is None:
        return event_dict

    type_name = f"{exc_type.__module__}.{exc_type.__qualname__}"
    frames = "".join(
        f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}\n'
        for frame in traceback.extract_tb(exc_traceback)
    )
    event_dict["exception_type"] = type_name
    event_dict["exception"] = f"Traceback (most recent call last):\n{frames}{type_name}"
    return event_dict


def _shared_processors() -> list[Processor]:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        sanitize_exception,
        redact_sensitive,
    ]


def _renderer(environment: str, stream: TextIO) -> Processor:
    if environment == "local":
        colors = hasattr(stream, "isatty") and stream.isatty()
        # exc_info is already reduced to a string by sanitize_exception, so the renderer
        # never formats tracebacks itself (rich would show local variables).
        return structlog.dev.ConsoleRenderer(
            colors=colors, exception_formatter=structlog.dev.plain_traceback
        )
    return structlog.processors.JSONRenderer()


def configure_logging(
    *, environment: str, log_level: str = "INFO", stream: TextIO | None = None
) -> None:
    """Configure structlog and the stdlib root logger. Safe to call more than once."""
    stream = stream or sys.stdout
    shared = _shared_processors()

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    handler = logging.StreamHandler(stream)
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                _renderer(environment, stream),
            ],
        )
    )

    root = logging.getLogger()
    # Replace only our own handler; leave others (e.g. pytest's capture) in place.
    for existing in list(root.handlers):
        if existing.get_name() == HANDLER_NAME:
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(log_level)

    # Uvicorn installs its own handlers before startup; route its logs through ours.
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    # Our access log replaces uvicorn's (which would include query strings).
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False
    access_logger.disabled = True

    # hide_parameters=True on the engine additionally keeps bound values out of errors.
    for name in QUIET_LIBRARY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
