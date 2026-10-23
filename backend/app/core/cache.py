"""Small cache abstraction.

Redis is an optimization, never a correctness dependency: implementations must not raise.
A failed read behaves as a miss and a failed write or delete is a no-op, so callers
always fall back to PostgreSQL.
"""

from typing import Protocol

import structlog
from redis import Redis
from redis.exceptions import RedisError

logger = structlog.get_logger(__name__)

# Connection, timeout and protocol errors from redis-py, plus low-level socket errors.
CACHE_ERRORS: tuple[type[Exception], ...] = (RedisError, OSError)


class Cache(Protocol):
    def get(self, key: str) -> bytes | None: ...

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None: ...

    def delete(self, *keys: str) -> None: ...


class NullCache:
    """Caching disabled: every read misses and writes are discarded."""

    def get(self, key: str) -> bytes | None:
        return None

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        return None

    def delete(self, *keys: str) -> None:
        return None


class RedisCache:
    """Fail-open Redis cache: errors are logged (type only) and swallowed."""

    def __init__(self, client: Redis) -> None:
        self._client = client

    def get(self, key: str) -> bytes | None:
        try:
            value = self._client.get(key)
        except CACHE_ERRORS as exc:
            _log_failure("get", key, exc)
            return None
        return value if isinstance(value, bytes) else None

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        try:
            self._client.set(key, value, ex=ttl_seconds)
        except CACHE_ERRORS as exc:
            _log_failure("set", key, exc)

    def delete(self, *keys: str) -> None:
        if not keys:
            return
        try:
            self._client.delete(*keys)
        except CACHE_ERRORS as exc:
            _log_failure("delete", ",".join(keys), exc)


def _log_failure(operation: str, key: str, exc: Exception) -> None:
    # Exception type only: messages can include hosts or other connection details.
    logger.warning(
        "cache_unavailable", operation=operation, key=key, exception_type=type(exc).__name__
    )
