"""Direct database inserts for integration tests (bypassing services)."""

import uuid

from sqlalchemy.orm import Session

from app.models import Account, User
from app.models.enums import UserRole

# Clearly fake: Phase 1 has no password hashing.
FAKE_PASSWORD_HASH = "not-a-real-hash"


def create_user(
    session: Session,
    *,
    email: str | None = None,
    is_active: bool = True,
    role: UserRole = UserRole.SALES_REP,
) -> User:
    user = User(
        email=email or f"user-{uuid.uuid4().hex[:12]}@example.com",
        password_hash=FAKE_PASSWORD_HASH,
        full_name="Test User",
        role=role,
        is_active=is_active,
    )
    session.add(user)
    session.commit()
    return user


def create_account(session: Session, *, name: str = "Acme", **fields: object) -> Account:
    account = Account(name=name, **fields)
    session.add(account)
    session.commit()
    return account
