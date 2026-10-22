"""Direct database inserts for integration tests (bypassing services)."""

import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models import Account, Activity, Contact, Lead, Note, Opportunity, User
from app.models.enums import (
    ActivityDirection,
    ActivityType,
    LeadSource,
    LeadStatus,
    OpportunityStage,
    UserRole,
)

# Clearly fake: Phase 1 has no password hashing.
FAKE_PASSWORD_HASH = "not-a-real-hash"


def create_user(
    session: Session,
    *,
    email: str | None = None,
    is_active: bool = True,
    role: UserRole = UserRole.SALES_REP,
    password: str | None = None,
) -> User:
    """Direct insert. With `password`, stores a real Argon2 hash so the user can log in."""
    user = User(
        email=email or f"user-{uuid.uuid4().hex[:12]}@example.com",
        password_hash=hash_password(password) if password else FAKE_PASSWORD_HASH,
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


def create_lead(session: Session, **fields: object) -> Lead:
    values: dict[str, object] = {
        "company_name": "Initech",
        "contact_name": "Peter",
        "lead_source": LeadSource.WEB,
        "status": LeadStatus.NEW,
    }
    lead = Lead(**{**values, **fields})
    session.add(lead)
    session.commit()
    return lead


def create_contact(session: Session, account: Account, **fields: object) -> Contact:
    contact = Contact(account_id=account.id, first_name="Ada", last_name="Lovelace", **fields)
    session.add(contact)
    session.commit()
    return contact


def create_opportunity(session: Session, account: Account, **fields: object) -> Opportunity:
    values: dict[str, object] = {"name": "Deal", "stage": OpportunityStage.PROSPECTING}
    opportunity = Opportunity(account_id=account.id, **{**values, **fields})
    session.add(opportunity)
    session.commit()
    return opportunity


def create_activity(session: Session, *, occurred_at: datetime, **fields: object) -> Activity:
    values: dict[str, object] = {
        "activity_type": ActivityType.CALL,
        "direction": ActivityDirection.OUTBOUND,
    }
    activity = Activity(occurred_at=occurred_at, **{**values, **fields})
    session.add(activity)
    session.commit()
    return activity


def create_note(session: Session, *, created_at: datetime | None = None, **fields: object) -> Note:
    values: dict[str, object] = {"body": "A note"}
    if created_at is not None:
        # Explicit created_at (allowed, see docs/DATABASE.md) lets tests control ordering.
        values["created_at"] = created_at
    note = Note(**{**values, **fields})
    session.add(note)
    session.commit()
    return note
