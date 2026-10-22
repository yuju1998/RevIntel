"""User management. No HTTP endpoint creates users; the CLI and tests call this service."""

from sqlalchemy.orm import Session

from app.core.emails import normalize_user_email
from app.core.exceptions import BusinessRuleError
from app.core.security import hash_password
from app.models.user import User
from app.repositories.user import UserRepository
from app.schemas.user import UserCreate
from app.services.transaction import write_transaction


def _email_taken() -> BusinessRuleError:
    return BusinessRuleError("A user with this email already exists")


CONSTRAINT_ERRORS = {"uq_users_email": _email_taken}


class UserService:
    def __init__(self, session: Session, users: UserRepository) -> None:
        self._session = session
        self._users = users

    def create_user(self, data: UserCreate) -> User:
        email = normalize_user_email(data.email)
        with write_transaction(self._session, CONSTRAINT_ERRORS):
            if self._users.get_by_email(email) is not None:
                raise _email_taken()
            user = self._users.add(
                User(
                    email=email,
                    full_name=data.full_name,
                    role=data.role,
                    password_hash=hash_password(data.password.get_secret_value()),
                    is_active=True,
                )
            )
        return user
