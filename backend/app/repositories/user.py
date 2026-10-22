"""User data access. Repositories never commit or roll back; services own transactions."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, undefer

from app.models.user import User


class UserRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, user_id: uuid.UUID) -> User | None:
        return self._session.get(User, user_id)

    def get_by_email(self, email: str, *, include_password_hash: bool = False) -> User | None:
        """Exact match; callers pass an already-normalized (lowercase) email.

        password_hash is deferred on the model; only login asks for it.
        """
        stmt = select(User).where(User.email == email)
        if include_password_hash:
            stmt = stmt.options(undefer(User.password_hash))
        return self._session.scalar(stmt)

    def add(self, user: User) -> User:
        self._session.add(user)
        self._session.flush()
        return user
