"""Database-level constraints, exercised directly (bypassing services and schemas)."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Account, Contact, Lead, Opportunity, User
from app.models.enums import LeadSource, LeadStatus, OpportunityStage
from app.services.transaction import violated_constraint
from tests.integration.factories import create_account, create_user

pytestmark = pytest.mark.integration


def _lead(**fields: object) -> Lead:
    values: dict[str, object] = {
        "company_name": "Initech",
        "contact_name": "Peter",
        "lead_source": LeadSource.WEB,
        "status": LeadStatus.NEW,
    }
    values.update(fields)
    return Lead(**values)


def _assert_violates(session: Session, obj: object, constraint: str) -> None:
    session.add(obj)
    with pytest.raises(IntegrityError) as exc_info:
        session.flush()
    session.rollback()
    assert violated_constraint(exc_info.value) == constraint


def test_duplicate_user_email_is_rejected(db_session: Session) -> None:
    create_user(db_session, email="dup@example.com")

    _assert_violates(
        db_session,
        User(email="dup@example.com", password_hash="x", full_name="Dup", role="ADMIN"),
        "uq_users_email",
    )


def test_user_email_uniqueness_is_case_sensitive_in_the_database(db_session: Session) -> None:
    # Documents the Q2 decision: lowercase normalization is the user service's job.
    create_user(db_session, email="case@example.com")
    create_user(db_session, email="CASE@example.com")

    assert db_session.scalar(select(text("count(*)")).select_from(User)) == 2


@pytest.mark.parametrize(
    ("obj_factory", "constraint"),
    [
        (lambda: Account(name="A", employee_count=-1), "ck_accounts_employee_count_non_negative"),
        (
            lambda: Account(name="A", annual_revenue=Decimal("-0.01")),
            "ck_accounts_annual_revenue_non_negative",
        ),
        (lambda: _lead(company_size=-1), "ck_leads_company_size_non_negative"),
        (lambda: _lead(estimated_value=Decimal("-5")), "ck_leads_estimated_value_non_negative"),
    ],
)
def test_negative_values_are_rejected(
    db_session: Session, obj_factory: object, constraint: str
) -> None:
    _assert_violates(db_session, obj_factory(), constraint)  # type: ignore[operator]


def test_negative_opportunity_amount_is_rejected(db_session: Session) -> None:
    account = create_account(db_session)

    _assert_violates(
        db_session,
        Opportunity(
            account_id=account.id,
            name="Deal",
            stage=OpportunityStage.PROSPECTING,
            amount=Decimal("-1"),
        ),
        "ck_opportunities_amount_non_negative",
    )


def test_null_numeric_values_pass_non_negative_checks(db_session: Session) -> None:
    db_session.add(Account(name="A", employee_count=None, annual_revenue=None))
    db_session.commit()


def test_converted_lead_requires_converted_at(db_session: Session) -> None:
    _assert_violates(
        db_session,
        _lead(status=LeadStatus.CONVERTED, converted_at=None),
        "ck_leads_converted_requires_converted_at",
    )


def test_converted_lead_with_converted_at_is_accepted(db_session: Session) -> None:
    db_session.add(_lead(status=LeadStatus.CONVERTED, converted_at=datetime.now(UTC)))
    db_session.commit()


def test_converted_at_check_is_one_directional(db_session: Session) -> None:
    db_session.add(_lead(status=LeadStatus.NEW, converted_at=datetime.now(UTC)))
    db_session.commit()


@pytest.mark.parametrize(
    ("table", "column", "constraint"),
    [
        ("users", "role", "ck_users_user_role"),
        ("leads", "status", "ck_leads_lead_status"),
        ("leads", "lead_source", "ck_leads_lead_source"),
        ("opportunities", "stage", "ck_opportunities_opportunity_stage"),
    ],
)
def test_enum_checks_reject_unknown_values(
    db_session: Session, table: str, column: str, constraint: str
) -> None:
    user = create_user(db_session)
    account = create_account(db_session)
    lead = _lead()
    db_session.add(lead)
    opportunity = Opportunity(
        account_id=account.id, name="Deal", stage=OpportunityStage.PROSPECTING
    )
    db_session.add(opportunity)
    db_session.commit()
    row_id = {"users": user.id, "leads": lead.id, "opportunities": opportunity.id}[table]

    with pytest.raises(IntegrityError) as exc_info:
        db_session.execute(
            text(f"UPDATE {table} SET {column} = 'BOGUS' WHERE id = :id"), {"id": row_id}
        )
    db_session.rollback()
    assert violated_constraint(exc_info.value) == constraint


def test_contact_requires_existing_account(db_session: Session) -> None:
    _assert_violates(
        db_session,
        Contact(account_id=uuid.uuid4(), first_name="A", last_name="B"),
        "fk_contacts_account_id_accounts",
    )


def test_opportunity_requires_existing_account(db_session: Session) -> None:
    _assert_violates(
        db_session,
        Opportunity(account_id=uuid.uuid4(), name="Deal", stage=OpportunityStage.PROPOSAL),
        "fk_opportunities_account_id_accounts",
    )


@pytest.mark.parametrize("child", ["contact", "opportunity"])
def test_deleting_account_with_children_is_restricted(db_session: Session, child: str) -> None:
    account = create_account(db_session)
    if child == "contact":
        db_session.add(Contact(account_id=account.id, first_name="A", last_name="B"))
        constraint = "fk_contacts_account_id_accounts"
    else:
        db_session.add(
            Opportunity(account_id=account.id, name="Deal", stage=OpportunityStage.PROPOSAL)
        )
        constraint = "fk_opportunities_account_id_accounts"
    db_session.commit()

    db_session.delete(account)
    with pytest.raises(IntegrityError) as exc_info:
        db_session.flush()
    db_session.rollback()
    assert violated_constraint(exc_info.value) == constraint


def test_deleting_user_sets_owner_id_to_null(db_session: Session) -> None:
    owner = create_user(db_session)
    account = create_account(db_session, owner_id=owner.id)
    lead = _lead(owner_id=owner.id)
    opportunity = Opportunity(
        account_id=account.id,
        name="Deal",
        stage=OpportunityStage.PROPOSAL,
        owner_id=owner.id,
    )
    db_session.add_all([lead, opportunity])
    db_session.commit()

    db_session.delete(owner)
    db_session.commit()
    db_session.expire_all()

    assert db_session.get(Account, account.id).owner_id is None  # type: ignore[union-attr]
    assert db_session.get(Lead, lead.id).owner_id is None  # type: ignore[union-attr]
    assert db_session.get(Opportunity, opportunity.id).owner_id is None  # type: ignore[union-attr]


def test_timestamps_are_database_generated_and_utc(db_session: Session) -> None:
    account = create_account(db_session)

    assert account.created_at.utcoffset() == timedelta(0)
    assert account.updated_at.utcoffset() == timedelta(0)
    assert db_session.scalar(text("SHOW timezone")) == "UTC"


def test_updated_at_changes_only_on_real_update(db_session: Session) -> None:
    account = create_account(db_session)
    created_at, updated_at = account.created_at, account.updated_at

    db_session.commit()  # no changes: no UPDATE emitted
    assert account.updated_at == updated_at

    account.industry = "Software"
    db_session.commit()

    assert account.updated_at > updated_at
    assert account.created_at == created_at
