"""HTTP mapping for known domain errors.

Deliberately narrow: only the domain errors below are handled. Generic 500 handling,
structured logging and request IDs belong to a later platform task.
"""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    BusinessRuleError,
    DomainError,
    NotFoundError,
    RelatedResourceNotFoundError,
)

STATUS_BY_ERROR: dict[type[DomainError], int] = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    RelatedResourceNotFoundError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    BusinessRuleError: status.HTTP_422_UNPROCESSABLE_CONTENT,
}


def register_exception_handlers(app: FastAPI) -> None:
    for error_type, status_code in STATUS_BY_ERROR.items():
        app.add_exception_handler(error_type, _handler_for(status_code))


def _handler_for(status_code: int) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        message = exc.message if isinstance(exc, DomainError) else "Request failed"
        return JSONResponse(status_code=status_code, content={"detail": message})

    return handle
