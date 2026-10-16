"""Response models for liveness and readiness endpoints."""

from typing import Literal

from pydantic import BaseModel

CheckStatus = Literal["ok", "error"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class DependencyCheck(BaseModel):
    status: CheckStatus
    latency_ms: float


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, DependencyCheck]
