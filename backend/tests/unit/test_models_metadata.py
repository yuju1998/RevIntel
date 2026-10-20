"""The ORM metadata matches docs/DATABASE.md (no database needed)."""

import pytest
from sqlalchemy import CheckConstraint, ForeignKeyConstraint, Table, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

import app.models  # noqa: F401  # registers every model
from app.db.base import Base
from app.services import account_service, contact_service, lead_service, opportunity_service


def _table(name: str) -> Table:
    return Base.metadata.tables[name]


def _checks(name: str) -> set[str]:
    return {str(c.name) for c in _table(name).constraints if isinstance(c, CheckConstraint)}


def _indexes(name: str) -> set[str]:
    return {str(index.name) for index in _table(name).indexes}


def test_registered_tables_are_the_phase_1_schema() -> None:
    # Phase 1 only. This set grows when Phase 2 (activities, notes) and later phases are
    # implemented; update it together with docs/DATABASE.md.
    assert set(Base.metadata.tables) == {"users", "accounts", "contacts", "leads", "opportunities"}


@pytest.mark.parametrize(
    ("table", "columns"),
    [
        (
            "users",
            {
                "id": ("UUID", False),
                "email": ("VARCHAR(320)", False),
                "password_hash": ("VARCHAR(255)", False),
                "full_name": ("VARCHAR(255)", False),
                "role": ("VARCHAR(32)", False),
                "is_active": ("BOOLEAN", False),
                "created_at": ("TIMESTAMP WITH TIME ZONE", False),
                "updated_at": ("TIMESTAMP WITH TIME ZONE", False),
            },
        ),
        (
            "accounts",
            {
                "id": ("UUID", False),
                "name": ("VARCHAR(255)", False),
                "industry": ("VARCHAR(255)", True),
                "employee_count": ("INTEGER", True),
                "annual_revenue": ("NUMERIC(16, 2)", True),
                "website": ("VARCHAR(2048)", True),
                "owner_id": ("UUID", True),
                "created_at": ("TIMESTAMP WITH TIME ZONE", False),
                "updated_at": ("TIMESTAMP WITH TIME ZONE", False),
            },
        ),
        (
            "contacts",
            {
                "id": ("UUID", False),
                "account_id": ("UUID", False),
                "first_name": ("VARCHAR(255)", False),
                "last_name": ("VARCHAR(255)", False),
                "email": ("VARCHAR(320)", True),
                "phone": ("VARCHAR(50)", True),
                "job_title": ("VARCHAR(255)", True),
                "created_at": ("TIMESTAMP WITH TIME ZONE", False),
                "updated_at": ("TIMESTAMP WITH TIME ZONE", False),
            },
        ),
        (
            "leads",
            {
                "id": ("UUID", False),
                "company_name": ("VARCHAR(255)", False),
                "contact_name": ("VARCHAR(255)", False),
                "email": ("VARCHAR(320)", True),
                "lead_source": ("VARCHAR(32)", False),
                "industry": ("VARCHAR(255)", True),
                "company_size": ("INTEGER", True),
                "estimated_value": ("NUMERIC(14, 2)", True),
                "status": ("VARCHAR(32)", False),
                "owner_id": ("UUID", True),
                "created_at": ("TIMESTAMP WITH TIME ZONE", False),
                "updated_at": ("TIMESTAMP WITH TIME ZONE", False),
                "converted_at": ("TIMESTAMP WITH TIME ZONE", True),
            },
        ),
        (
            "opportunities",
            {
                "id": ("UUID", False),
                "account_id": ("UUID", False),
                "name": ("VARCHAR(255)", False),
                "stage": ("VARCHAR(32)", False),
                "amount": ("NUMERIC(14, 2)", True),
                "expected_close_date": ("DATE", True),
                "description": ("TEXT", True),
                "owner_id": ("UUID", True),
                "created_at": ("TIMESTAMP WITH TIME ZONE", False),
                "updated_at": ("TIMESTAMP WITH TIME ZONE", False),
            },
        ),
    ],
)
def test_columns_match_spec(table: str, columns: dict[str, tuple[str, bool]]) -> None:
    dialect = postgresql.dialect()
    actual = {
        column.name: (column.type.compile(dialect=dialect), column.nullable)
        for column in _table(table).columns
    }

    assert actual == columns


def test_check_constraints_match_spec() -> None:
    assert _checks("users") == {"ck_users_user_role"}
    assert _checks("accounts") == {
        "ck_accounts_employee_count_non_negative",
        "ck_accounts_annual_revenue_non_negative",
    }
    assert _checks("contacts") == set()
    assert _checks("leads") == {
        "ck_leads_lead_source",
        "ck_leads_lead_status",
        "ck_leads_company_size_non_negative",
        "ck_leads_estimated_value_non_negative",
        "ck_leads_converted_requires_converted_at",
    }
    assert _checks("opportunities") == {
        "ck_opportunities_opportunity_stage",
        "ck_opportunities_amount_non_negative",
    }


def test_indexes_match_spec() -> None:
    assert _indexes("users") == set()
    assert _indexes("accounts") == {"ix_accounts_owner_id"}
    assert _indexes("contacts") == {"ix_contacts_account_id"}
    assert _indexes("leads") == {"ix_leads_owner_id", "ix_leads_status", "ix_leads_lead_source"}
    assert _indexes("opportunities") == {
        "ix_opportunities_account_id",
        "ix_opportunities_owner_id",
        "ix_opportunities_stage",
    }
    unique = {
        str(c.name) for c in _table("users").constraints if isinstance(c, UniqueConstraint)
    }
    assert unique == {"uq_users_email"}


def test_foreign_keys_and_delete_behavior_match_spec() -> None:
    foreign_keys = {
        str(c.name): c.ondelete
        for table in Base.metadata.tables.values()
        for c in table.constraints
        if isinstance(c, ForeignKeyConstraint)
    }

    assert foreign_keys == {
        "fk_accounts_owner_id_users": "SET NULL",
        "fk_leads_owner_id_users": "SET NULL",
        "fk_opportunities_owner_id_users": "SET NULL",
        "fk_contacts_account_id_accounts": "RESTRICT",
        "fk_opportunities_account_id_accounts": "RESTRICT",
    }


def test_enum_checks_list_exact_values() -> None:
    ddl = str(CreateTable(_table("leads")).compile(dialect=postgresql.dialect()))

    assert "status IN ('NEW', 'WORKING', 'QUALIFIED', 'CONVERTED', 'DISQUALIFIED')" in ddl
    assert "lead_source IN ('WEB', 'REFERRAL', 'EVENT', 'OUTBOUND', 'PARTNER', 'OTHER')" in ddl
    assert "status <> 'CONVERTED' OR converted_at IS NOT NULL" in ddl


def test_service_constraint_mappings_refer_to_real_constraints() -> None:
    names = {
        str(c.name) for table in Base.metadata.tables.values() for c in table.constraints
    }
    mapped = (
        set(account_service.CONSTRAINT_ERRORS)
        | set(contact_service.CONSTRAINT_ERRORS)
        | set(lead_service.CONSTRAINT_ERRORS)
        | set(opportunity_service.CONSTRAINT_ERRORS)
    )

    assert mapped <= names
