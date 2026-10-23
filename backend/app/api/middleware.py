"""Request context middleware: request IDs, access logging and safe 500 responses.

Pure ASGI (not BaseHTTPMiddleware) so contextvars bound here are inherited by the
request's handlers, including sync routes and dependencies run in the thread pool.

Position: inside Starlette's ServerErrorMiddleware and outside the ExceptionMiddleware
that runs our domain-error handlers, so every response (2xx, 401/403/404/422 and 500)
gets X-Request-ID and exactly one access-log line.
"""

import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
# Probes hit these every few seconds; log them at DEBUG so INFO stays readable.
QUIET_PATHS = frozenset({"/health", "/ready"})

logger = structlog.get_logger("app.request")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Always generated server-side; an incoming X-Request-ID is ignored.
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        scope.setdefault("state", {})["request_id"] = request_id

        started = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception as exc:
            # Type and stack frames only (see app.core.logging.sanitize_exception); the
            # exception message is never logged or returned.
            logger.error("unhandled_exception", exc_info=exc)
            if response_started:
                raise
            status_code = 500
            response = JSONResponse(
                {"detail": "Internal Server Error"},
                status_code=500,
                headers={REQUEST_ID_HEADER: request_id},
            )
            await response(scope, receive, send)
        finally:
            _log_request(scope, status_code, started)
            structlog.contextvars.clear_contextvars()


def _log_request(scope: Scope, status_code: int, started: float) -> None:
    fields: dict[str, object] = {
        "method": scope["method"],
        # Path only: query strings are never logged.
        "path": scope["path"],
        "status_code": status_code,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    route = _route_template(scope)
    if route is not None:
        fields["route"] = route
    user_id = scope.get("state", {}).get("user_id")
    if user_id is not None:
        fields["user_id"] = user_id

    if status_code >= 500:
        logger.error("http_request", **fields)
    elif scope["path"] in QUIET_PATHS:
        logger.debug("http_request", **fields)
    else:
        logger.info("http_request", **fields)


def _route_template(scope: Scope) -> str | None:
    """Canonical matched route, e.g. /api/v1/accounts/{account_id}, for grouping requests.

    FastAPI's nested routers only expose their relative path on scope["route"], so the
    full template comes from the app's public url_path_for with placeholder parameters.
    Concrete parameter values are never used. None when no route matched (e.g. a 404).
    """
    route = scope.get("route")
    name = getattr(route, "name", None)
    if name is None:
        return None
    placeholders = {param: f"{{{param}}}" for param in scope.get("path_params", {})}
    try:
        return scope["app"].url_path_for(name, **placeholders)
    except Exception:
        # e.g. typed converters ({n:int}) reject placeholders: use the relative template.
        return getattr(route, "path", None)
