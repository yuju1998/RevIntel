"""Redis client construction."""

from redis import Redis

from app.core.config import Settings


def create_redis_client(settings: Settings) -> Redis:
    # from_url is lazy: the connection pool connects on the first command.
    return Redis.from_url(
        str(settings.redis_url),
        socket_connect_timeout=settings.redis_socket_timeout_seconds,
        socket_timeout=settings.redis_socket_timeout_seconds,
    )
