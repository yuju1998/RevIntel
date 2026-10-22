"""Login and access-token resolution. Stateless: no sessions, refresh tokens or blocklists."""

from pydantic import SecretStr
from sqlalchemy.orm import Session

from app.core.emails import normalize_user_email
from app.core.exceptions import AuthenticationError
from app.core.security import (
    AccessToken,
    create_access_token,
    decode_access_token,
    invalid_token,
    verify_dummy_password,
    verify_password,
)
from app.models.user import User
from app.repositories.user import UserRepository
from app.services.transaction import write_transaction


def invalid_credentials() -> AuthenticationError:
    # One message for unknown email, wrong password and inactive user (no enumeration).
    return AuthenticationError("Incorrect email or password")


class AuthService:
    def __init__(
        self,
        session: Session,
        users: UserRepository,
        *,
        jwt_secret: SecretStr,
        access_token_expire_minutes: int,
    ) -> None:
        self._session = session
        self._users = users
        self._jwt_secret = jwt_secret
        self._expire_minutes = access_token_expire_minutes

    def login(self, email: str, password: SecretStr) -> AccessToken:
        user = self._users.get_by_email(normalize_user_email(email), include_password_hash=True)
        if user is None:
            verify_dummy_password(password.get_secret_value())
            raise invalid_credentials()

        valid, upgraded_hash = verify_password(password.get_secret_value(), user.password_hash)
        if not valid or not user.is_active:
            raise invalid_credentials()
        if upgraded_hash is not None:
            # Argon2 parameters changed since this hash was made: store the stronger hash.
            with write_transaction(self._session):
                user.password_hash = upgraded_hash

        return create_access_token(
            user.id, secret=self._jwt_secret, expires_minutes=self._expire_minutes
        )

    def user_for_token(self, token: str) -> User:
        """The active user a valid access token belongs to, loaded fresh from PostgreSQL."""
        user_id = decode_access_token(token, secret=self._jwt_secret)
        user = self._users.get(user_id)
        if user is None or not user.is_active:
            raise invalid_token()
        return user
