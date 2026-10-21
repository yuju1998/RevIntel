"""Round trip of the Alembic migrations against revintel_test."""

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, inspect

from tests.integration.conftest import ALEMBIC_INI

pytestmark = pytest.mark.integration

PHASE_1_REVISION = "d9abf148e9ed"
PHASE_1_TABLES = {"users", "accounts", "contacts", "leads", "opportunities"}
PHASE_2_TABLES = {"activities", "notes"}
ALL_TABLES = PHASE_1_TABLES | PHASE_2_TABLES


@pytest.fixture
def alembic_config(migrated_database: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("DATABASE_URL", migrated_database)
    return Config(str(ALEMBIC_INI))


def _tables(engine: Engine) -> set[str]:
    engine.dispose()  # fresh connection, so the inspector sees the latest DDL
    return set(inspect(engine).get_table_names()) - {"alembic_version"}


def _version(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar()


def test_downgrade_to_base_and_upgrade_to_head(alembic_config: Config, db_engine: Engine) -> None:
    head = ScriptDirectory.from_config(alembic_config).get_current_head()
    assert _tables(db_engine) == ALL_TABLES

    command.downgrade(alembic_config, "base")
    # alembic_version is managed by Alembic and kept (empty) after a full downgrade.
    assert _tables(db_engine) == set()
    assert _version(db_engine) is None

    # Leave the database at head for the remaining tests.
    command.upgrade(alembic_config, "head")
    assert _tables(db_engine) == ALL_TABLES
    assert _version(db_engine) == head
    command.check(alembic_config)


def test_each_migration_applies_and_reverts_independently(
    alembic_config: Config, db_engine: Engine
) -> None:
    command.downgrade(alembic_config, "base")

    command.upgrade(alembic_config, PHASE_1_REVISION)
    assert _tables(db_engine) == PHASE_1_TABLES

    command.upgrade(alembic_config, "head")
    assert _tables(db_engine) == ALL_TABLES

    # Reverting migration 2 leaves the Phase 1 schema intact.
    command.downgrade(alembic_config, "-1")
    assert _tables(db_engine) == PHASE_1_TABLES
    assert _version(db_engine) == PHASE_1_REVISION

    command.upgrade(alembic_config, "head")
    command.check(alembic_config)


def test_migrations_create_documented_indexes(db_engine: Engine) -> None:
    inspector = inspect(db_engine)
    # Indexes that back a unique constraint (uq_users_email) are reported separately.
    indexes = {
        index["name"]
        for table in ALL_TABLES
        for index in inspector.get_indexes(table)
        if "duplicates_constraint" not in index
    }
    unique_constraints = {
        constraint["name"]
        for table in ALL_TABLES
        for constraint in inspector.get_unique_constraints(table)
    }

    assert unique_constraints == {"uq_users_email"}
    assert indexes == {
        "ix_accounts_owner_id",
        "ix_contacts_account_id",
        "ix_leads_owner_id",
        "ix_leads_status",
        "ix_leads_lead_source",
        "ix_opportunities_account_id",
        "ix_opportunities_owner_id",
        "ix_opportunities_stage",
        "ix_activities_account_id",
        "ix_activities_lead_id",
        "ix_activities_opportunity_id",
        "ix_activities_contact_id",
        "ix_activities_created_by_id",
        "ix_activities_occurred_at",
        "ix_notes_account_id",
        "ix_notes_lead_id",
        "ix_notes_opportunity_id",
        "ix_notes_author_id",
    }


def test_migrated_enum_checks_exist_exactly_once(db_engine: Engine) -> None:
    with db_engine.connect() as connection:
        names = connection.exec_driver_sql(
            "SELECT conname FROM pg_constraint WHERE conrelid = 'activities'::regclass "
            "AND contype = 'c'"
        ).scalars().all()

    assert sorted(names) == [
        "ck_activities_activity_direction",
        "ck_activities_activity_type",
        "ck_activities_at_least_one_parent",
    ]
