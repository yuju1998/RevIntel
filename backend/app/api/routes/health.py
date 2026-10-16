"""Liveness and readiness probes."""

from fastapi import APIRouter, Response, status

from app.api.deps import EngineDep, RedisDep
from app.schemas.health import HealthResponse, ReadinessResponse
from app.services import health_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe. Never touches PostgreSQL or Redis."""
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
def ready(response: Response, engine: EngineDep, redis_client: RedisDep) -> ReadinessResponse:
    """Readiness probe. Returns 503 unless PostgreSQL and Redis are both reachable."""
    result = health_service.check_readiness(engine, redis_client)
    if result.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
