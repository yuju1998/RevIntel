"""Domain errors raised by services and mapped to HTTP responses in `app.api.errors`."""


class DomainError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(DomainError):
    """The resource addressed by the request (e.g. a path ID) does not exist."""


class RelatedResourceNotFoundError(DomainError):
    """A related resource referenced in the request body does not exist."""


class BusinessRuleError(DomainError):
    """The request is well-formed but violates a business rule."""


class AuthenticationError(DomainError):
    """Credentials are missing, invalid or expired, or the user is unknown or inactive.

    Messages are deliberately generic and never include tokens or passwords.
    """


class PermissionDeniedError(DomainError):
    """The authenticated user's role is not allowed to perform the action."""
