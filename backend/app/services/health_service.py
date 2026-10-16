"""Readiness checks for infrastructure dependencies."""

import logging
import time
from collections.abc import Callable

from redis import Redis
from sqlalchemy import Engine, text

from app.schemas.health import CheckStatus, DependencyCheck, ReadinessResponse

logger = logging.getLogger(__name__)


def check_postgres(engine: Engine) -> DependencyCheck:
    def probe() -> None:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    return _run_check("postgres", probe)


def check_redis(client: Redis) -> DependencyCheck:
    return _run_check("redis", client.ping)


def check_readiness(engine: Engine, redis_client: Redis) -> ReadinessResponse:
    checks = {
        "postgres": check_postgres(engine),
        "redis": check_redis(redis_client),
    }
    is_ready = all(check.status == "ok" for check in checks.values())
    return ReadinessResponse(status="ready" if is_ready else "not_ready", checks=checks)


def _run_check(name: str, probe: Callable[[], object]) -> DependencyCheck:
    started = time.perf_counter()
    status: CheckStatus
    try:
        probe()
    except Exception as exc:
        # Log only the exception type: driver messages can include hosts and usernames.
        logger.warning("Readiness check %s failed: %s", name, type(exc).__name__)
        status = "error"
    else:
        status = "ok"
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return DependencyCheck(status=status, latency_ms=latency_ms)
