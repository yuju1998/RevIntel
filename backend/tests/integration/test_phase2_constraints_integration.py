"""Activity/note database constraints, exercised directly (bypassing services and schemas)."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Account, Activity, Contact, Lead, Note, Opportunity
from app.models.enums import ActivityDirection, ActivityType
from app.services.transaction import violated_constraint
from tests.integration.factories import (
    create_account,
    create_activity,
    create_contact,
    create_lead,
    create_note,
    create_opportunity,
    create_user,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _violation(session: Session, statement: str, params: dict | None = None) -> str | None:
    with pytest.raises(IntegrityError) as exc_info:
        session.execute(text(statement), params or {})
    session.rollback()
    return violated_constraint(exc_info.value)


def _only_parent(session: Session, parent: str) -> Account | Lead | Opportunity | Contact:
    """Create just the parent under test, so no other child row blocks its deletion."""
    if parent == "lead":
        return create_lead(session)
    account = create_account(session)
    if parent == "opportunity":
        return create_opportunity(session, account)
    if parent == "contact":
        return create_contact(session, account)
    return account


def test_activity_requires_at_least_one_parent(db_session: Session) -> None:
    db_session.add(
        Activity(
            activity_type=ActivityType.CALL, direction=ActivityDirection.INBOUND, occurred_at=NOW
        )
    )
    with pytest.raises(IntegrityError) as exc_info:
        db_session.flush()
    db_session.rollback()

    assert violated_constraint(exc_info.value) == "ck_activities_at_least_one_parent"


def test_note_requires_at_least_one_parent(db_session: Session) -> None:
    db_session.add(Note(body="orphan"))
    with pytest.raises(IntegrityError) as exc_info:
        db_session.flush()
    db_session.rollback()

    assert violated_constraint(exc_info.value) == "ck_notes_at_least_one_parent"


def test_activity_may_reference_several_parents(db_session: Session) -> None:
    account = create_account(db_session)
    create_activity(
        db_session,
        occurred_at=NOW,
        account_id=account.id,
        lead_id=create_lead(db_session).id,
        opportunity_id=create_opportunity(db_session, account).id,
        contact_id=create_contact(db_session, account).id,
    )


@pytest.mark.parametrize(
    ("column", "constraint"),
    [
        ("activity_type", "ck_activities_activity_type"),
        ("direction", "ck_activities_activity_direction"),
    ],
)
def test_activity_enum_checks_reject_unknown_values(
    db_session: Session, column: str, constraint: str
) -> None:
    activity = create_activity(db_session, occurred_at=NOW, lead_id=create_lead(db_session).id)

    assert (
        _violation(
            db_session,
            f"UPDATE activities SET {column} = 'BOGUS' WHERE id = :id",
            {"id": activity.id},
        )
        == constraint
    )


@pytest.mark.parametrize(
    ("column", "constraint"),
    [
        ("account_id", "fk_activities_account_id_accounts"),
        ("lead_id", "fk_activities_lead_id_leads"),
        ("opportunity_id", "fk_activities_opportunity_id_opportunities"),
        ("contact_id", "fk_activities_contact_id_contacts"),
        ("created_by_id", "fk_activities_created_by_id_users"),
    ],
)
def test_activity_foreign_keys(db_session: Session, column: str, constraint: str) -> None:
    lead = create_lead(db_session)
    assert (
        _violation(
            db_session,
            "INSERT INTO activities (id, lead_id, activity_type, direction, occurred_at, "
            f"{column}) VALUES (:id, :lead, 'CALL', 'INBOUND', now(), :missing)"
            if column != "lead_id"
            else "INSERT INTO activities (id, activity_type, direction, occurred_at, lead_id) "
            "VALUES (:id, 'CALL', 'INBOUND', now(), :missing)",
            {"id": uuid.uuid4(), "lead": lead.id, "missing": uuid.uuid4()},
        )
        == constraint
    )


@pytest.mark.parametrize(
    "parent", ["account", "lead", "opportunity", "contact"], ids=lambda p: f"activity-{p}"
)
def test_deleting_a_parent_of_an_activity_is_restricted(db_session: Session, parent: str) -> None:
    target = _only_parent(db_session, parent)
    create_activity(db_session, occurred_at=NOW, **{f"{parent}_id": target.id})

    table = {"account": "accounts", "lead": "leads", "opportunity": "opportunities"}.get(
        parent, "contacts"
    )
    assert _violation(
        db_session, f"DELETE FROM {table} WHERE id = :id", {"id": target.id}
    ) == f"fk_activities_{parent}_id_{table}"


@pytest.mark.parametrize("parent", ["account", "lead", "opportunity"], ids=lambda p: f"note-{p}")
def test_deleting_a_parent_of_a_note_is_restricted(db_session: Session, parent: str) -> None:
    target = _only_parent(db_session, parent)
    create_note(db_session, **{f"{parent}_id": target.id})

    table = {"account": "accounts", "lead": "leads", "opportunity": "opportunities"}[parent]
    assert _violation(
        db_session, f"DELETE FROM {table} WHERE id = :id", {"id": target.id}
    ) == f"fk_notes_{parent}_id_{table}"


def test_deleting_a_lead_via_the_orm_is_restricted(db_session: Session) -> None:
    # Parent models have no collections, so the ORM never nulls children itself.
    lead = create_lead(db_session)
    create_activity(db_session, occurred_at=NOW, lead_id=lead.id)

    db_session.delete(lead)
    with pytest.raises(IntegrityError) as exc_info:
        db_session.flush()
    db_session.rollback()

    assert violated_constraint(exc_info.value) == "fk_activities_lead_id_leads"


def test_deleting_a_user_sets_creator_and_author_to_null(db_session: Session) -> None:
    user = create_user(db_session)
    account = create_account(db_session)
    activity = create_activity(
        db_session, occurred_at=NOW, account_id=account.id, created_by_id=user.id
    )
    note = create_note(db_session, account_id=account.id, author_id=user.id)

    db_session.delete(user)
    db_session.commit()
    db_session.expire_all()

    assert db_session.get(Activity, activity.id).created_by_id is None  # type: ignore[union-attr]
    assert db_session.get(Note, note.id).author_id is None  # type: ignore[union-attr]


def test_activity_created_at_is_database_generated_utc(db_session: Session) -> None:
    activity = create_activity(db_session, occurred_at=NOW, lead_id=create_lead(db_session).id)

    assert activity.created_at.utcoffset() is not None
    assert activity.created_at.utcoffset().total_seconds() == 0  # type: ignore[union-attr]
    assert activity.occurred_at == NOW
