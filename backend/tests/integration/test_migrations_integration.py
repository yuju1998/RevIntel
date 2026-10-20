"""Round trip of the Alembic migrations against revintel_test."""

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, inspect

from tests.integration.conftest import ALEMBIC_INI

pytestmark = pytest.mark.integration

PHASE_1_TABLES = {"users", "accounts", "contacts", "leads", "opportunities"}


@pytest.fixture
def alembic_config(migrated_database: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("DATABASE_URL", migrated_database)
    return Config(str(ALEMBIC_INI))


def _tables(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def test_downgrade_to_base_and_upgrade_to_head(alembic_config: Config, db_engine: Engine) -> None:
    head = ScriptDirectory.from_config(alembic_config).get_current_head()
    assert PHASE_1_TABLES <= _tables(db_engine)

    command.downgrade(alembic_config, "base")
    db_engine.dispose()
    # alembic_version is managed by Alembic and kept (empty) after a full downgrade.
    assert _tables(db_engine) == {"alembic_version"}

    # Leave the database at head for the remaining tests.
    command.upgrade(alembic_config, "head")
    db_engine.dispose()
    assert PHASE_1_TABLES <= _tables(db_engine)
    with db_engine.connect() as connection:
        version = connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar()
    assert version == head
    command.check(alembic_config)


def test_migration_creates_documented_indexes(db_engine: Engine) -> None:
    inspector = inspect(db_engine)
    # Indexes that back a unique constraint (uq_users_email) are reported separately.
    indexes = {
        index["name"]
        for table in PHASE_1_TABLES
        for index in inspector.get_indexes(table)
        if "duplicates_constraint" not in index
    }
    unique_constraints = {
        constraint["name"]
        for table in PHASE_1_TABLES
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
    }
