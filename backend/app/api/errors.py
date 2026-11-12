"""HTTP mapping for known domain errors.

Deliberately narrow: only the domain errors below are handled here. Unhandled exceptions
become a generic 500 in app.api.middleware.RequestContextMiddleware.
"""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AuthenticationError,
    BusinessRuleError,
    DomainError,
    NotFoundError,
    PermissionDeniedError,
    RelatedResourceNotFoundError,
    ScoringUnavailableError,
    SearchUnavailableError,
)

STATUS_BY_ERROR: dict[type[DomainError], int] = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    RelatedResourceNotFoundError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    BusinessRuleError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    AuthenticationError: status.HTTP_401_UNAUTHORIZED,
    PermissionDeniedError: status.HTTP_403_FORBIDDEN,
    ScoringUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    SearchUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def register_exception_handlers(app: FastAPI) -> None:
    for error_type, status_code in STATUS_BY_ERROR.items():
        app.add_exception_handler(error_type, _handler_for(status_code))


def _handler_for(status_code: int) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        message = exc.message if isinstance(exc, DomainError) else "Request failed"
        # RFC 6750: a 401 tells the client which authentication scheme to use.
        headers = (
            {"WWW-Authenticate": "Bearer"}
            if status_code == status.HTTP_401_UNAUTHORIZED
            else None
        )
        return JSONResponse(status_code=status_code, content={"detail": message}, headers=headers)

    return handle
