"""Route class for endpoints whose request bodies contain secrets (e.g. passwords).

FastAPI's 422 response echoes the offending input, which can be the password itself (an
over-long password, a misnamed field, or the whole body when another field is missing).
This route class removes the `input` from validation errors for the routes that use it,
so other endpoints keep FastAPI's standard error format.
"""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute


class SensitiveBodyRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handler = super().get_route_handler()

        async def handle(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                errors = [
                    {key: value for key, value in error.items() if key != "input"}
                    for error in exc.errors()
                ]
                raise RequestValidationError(errors) from None

        return handle
