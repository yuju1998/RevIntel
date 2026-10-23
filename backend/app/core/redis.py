"""Redis client construction."""

from redis import Redis

from app.core.config import Settings


def create_redis_client(settings: Settings, *, timeout_seconds: float | None = None) -> Redis:
    """Redis client for REDIS_URL. `timeout_seconds` overrides the default socket timeout."""
    timeout = (
        timeout_seconds if timeout_seconds is not None else settings.redis_socket_timeout_seconds
    )
    # from_url is lazy: the connection pool connects on the first command.
    return Redis.from_url(
        str(settings.redis_url),
        socket_connect_timeout=timeout,
        socket_timeout=timeout,
    )
