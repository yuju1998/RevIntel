"""The ORM metadata matches docs/DATABASE.md (no database needed)."""

import pytest
from sqlalchemy import CheckConstraint, ForeignKeyConstraint, Table, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

import app.models  # noqa: F401  # registers every model
from app.db.base import Base
from app.services import (
    account_service,
    activity_service,
    contact_service,
    lead_service,
    note_service,
    opportunity_service,
)


def _table(name: str) -> Table:
    return Base.metadata.tables[name]


def _checks(name: str) -> set[str]:
    return {str(c.name) for c in _table(name).constraints if isinstance(c, CheckConstraint)}


def _indexes(name: str) -> set[str]:
    return {str(index.name) for index in _table(name).indexes}


def test_registered_tables_are_the_implemented_schema_phases() -> None:
    # Schema phases 1-4; update together with docs/DATABASE.md.
    assert set(Base.metadata.tables) == {
        "users",
        "accounts",
        "contacts",
        "leads",
        "opportunities",
        "activities",
        "notes",
        "lead_predictions",
        "document_chunks",
    }


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
        (
            "activities",
            {
                "id": ("UUID", False),
                "account_id": ("UUID", True),
                "lead_id": ("UUID", True),
                "opportunity_id": ("UUID", True),
                "contact_id": ("UUID", True),
                "created_by_id": ("UUID", True),
                "activity_type": ("VARCHAR(32)", False),
                "direction": ("VARCHAR(32)", False),
                "subject": ("VARCHAR(255)", True),
                "body": ("TEXT", True),
                "occurred_at": ("TIMESTAMP WITH TIME ZONE", False),
                "created_at": ("TIMESTAMP WITH TIME ZONE", False),
            },
        ),
        (
            "notes",
            {
                "id": ("UUID", False),
                "account_id": ("UUID", True),
                "lead_id": ("UUID", True),
                "opportunity_id": ("UUID", True),
                "author_id": ("UUID", True),
                "body": ("TEXT", False),
                "created_at": ("TIMESTAMP WITH TIME ZONE", False),
                "updated_at": ("TIMESTAMP WITH TIME ZONE", False),
            },
        ),
        (
            "lead_predictions",
            {
                "id": ("UUID", False),
                "lead_id": ("UUID", False),
                "prediction_timestamp": ("TIMESTAMP WITH TIME ZONE", False),
                "conversion_probability": ("FLOAT", False),
                "lead_tier": ("VARCHAR(32)", False),
                "model_name": ("VARCHAR(128)", False),
                "model_version": ("VARCHAR(64)", False),
                "features": ("JSONB", False),
                "inference_ms": ("FLOAT", True),
                "created_at": ("TIMESTAMP WITH TIME ZONE", False),
            },
        ),
        (
            "document_chunks",
            {
                "id": ("UUID", False),
                "account_id": ("UUID", False),
                "source_type": ("VARCHAR(32)", False),
                "source_id": ("UUID", False),
                "entity_type": ("VARCHAR(32)", False),
                "entity_id": ("UUID", False),
                "chunk_index": ("INTEGER", False),
                "content": ("TEXT", False),
                "embedding": ("VECTOR(384)", False),
                "embedding_model": ("VARCHAR(255)", False),
                "source_sha256": ("VARCHAR(64)", False),
                "metadata": ("JSONB", False),
                "occurred_at": ("TIMESTAMP WITH TIME ZONE", False),
                "created_at": ("TIMESTAMP WITH TIME ZONE", False),
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
    assert _checks("activities") == {
        "ck_activities_activity_type",
        "ck_activities_activity_direction",
        "ck_activities_at_least_one_parent",
    }
    assert _checks("notes") == {"ck_notes_at_least_one_parent"}
    assert _checks("lead_predictions") == {
        "ck_lead_predictions_lead_tier",
        "ck_lead_predictions_conversion_probability_range",
        "ck_lead_predictions_inference_ms_non_negative",
    }
    assert _checks("document_chunks") == {
        "ck_document_chunks_document_source_type",
        "ck_document_chunks_document_entity_type",
        "ck_document_chunks_chunk_index_non_negative",
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
    assert _indexes("activities") == {
        "ix_activities_account_id",
        "ix_activities_lead_id",
        "ix_activities_opportunity_id",
        "ix_activities_contact_id",
        "ix_activities_created_by_id",
        "ix_activities_occurred_at",
    }
    assert _indexes("notes") == {
        "ix_notes_account_id",
        "ix_notes_lead_id",
        "ix_notes_opportunity_id",
        "ix_notes_author_id",
    }
    assert _indexes("lead_predictions") == {"ix_lead_predictions_prediction_timestamp"}
    assert _indexes("document_chunks") == {"ix_document_chunks_account_id"}
    unique = {
        str(c.name)
        for table in Base.metadata.tables.values()
        for c in table.constraints
        if isinstance(c, UniqueConstraint)
    }
    assert unique == {
        "uq_users_email",
        "uq_lead_predictions_lead_id_timestamp_model",
        "uq_document_chunks_source_type_source_id_chunk_index",
    }


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
        "fk_activities_account_id_accounts": "RESTRICT",
        "fk_activities_lead_id_leads": "RESTRICT",
        "fk_activities_opportunity_id_opportunities": "RESTRICT",
        "fk_activities_contact_id_contacts": "RESTRICT",
        "fk_activities_created_by_id_users": "SET NULL",
        "fk_notes_account_id_accounts": "RESTRICT",
        "fk_notes_lead_id_leads": "RESTRICT",
        "fk_notes_opportunity_id_opportunities": "RESTRICT",
        "fk_notes_author_id_users": "SET NULL",
        "fk_lead_predictions_lead_id_leads": "CASCADE",
        "fk_document_chunks_account_id_accounts": "CASCADE",
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
        | set(activity_service.CONSTRAINT_ERRORS)
        | set(note_service.CONSTRAINT_ERRORS)
    )

    assert mapped <= names


def test_activity_and_note_checks_allow_several_parents() -> None:
    ddl = " ".join(
        str(CreateTable(_table(name)).compile(dialect=postgresql.dialect()))
        for name in ("activities", "notes")
    )

    # "At least one" (OR), never "exactly one".
    assert (
        "account_id IS NOT NULL OR lead_id IS NOT NULL "
        "OR opportunity_id IS NOT NULL OR contact_id IS NOT NULL"
    ) in ddl
    assert "account_id IS NOT NULL OR lead_id IS NOT NULL OR opportunity_id IS NOT NULL" in ddl
    assert "activity_type IN ('CALL', 'EMAIL', 'MEETING', 'DEMO', 'FOLLOW_UP')" in ddl
    assert "direction IN ('INBOUND', 'OUTBOUND', 'INTERNAL')" in ddl


def test_activities_are_append_only() -> None:
    assert "updated_at" not in _table("activities").columns
