"""Top-level API router."""

from fastapi import APIRouter

from app.api.routes import (
    accounts,
    activities,
    auth,
    contacts,
    health,
    leads,
    notes,
    opportunities,
    users,
)

API_V1_PREFIX = "/api/v1"

api_router = APIRouter()

# Infrastructure probes live at the root so load balancers and ECS don't depend on API versioning.
api_router.include_router(health.router)

# Business endpoints are registered on this router.
v1_router = APIRouter(prefix=API_V1_PREFIX)
v1_router.include_router(auth.router)
v1_router.include_router(users.router)
v1_router.include_router(accounts.router)
v1_router.include_router(contacts.router)
v1_router.include_router(leads.router)
v1_router.include_router(opportunities.router)
v1_router.include_router(activities.router)
v1_router.include_router(notes.router)
api_router.include_router(v1_router)
