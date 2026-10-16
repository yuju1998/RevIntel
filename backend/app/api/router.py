"""Top-level API router."""

from fastapi import APIRouter

from app.api.routes import health

API_V1_PREFIX = "/api/v1"

api_router = APIRouter()

# Infrastructure probes live at the root so load balancers and ECS don't depend on API versioning.
api_router.include_router(health.router)

# Business endpoints are registered on this router.
v1_router = APIRouter(prefix=API_V1_PREFIX)
api_router.include_router(v1_router)
