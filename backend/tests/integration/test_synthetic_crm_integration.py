"""Synthetic CRM data against the real schema in revintel_test (constraints, guards, reset)."""

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from ml.synthetic.config import GenerationConfig
from ml.synthetic.generator import SyntheticDataset, generate
from ml.synthetic.writer import CRM_TABLES, UnsafeTargetError, write_dataset
from tests.integration.factories import create_account

pytestmark = pytest.mark.integration

CONFIG = GenerationConfig(
    seed=42, accounts=20, contacts=60, leads=200, activities=1000, reps=4, admins=1
)


@pytest.fixture(scope="module")
def dataset() -> SyntheticDataset:
    return generate(CONFIG)


def _counts(engine: Engine) -> dict[str, int]:
    with engine.connect() as connection:
        return {
            table: connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
            for table in CRM_TABLES
        }


def _checksum(engine: Engine) -> str:
    """Order-independent fingerprint of every CRM row as stored by PostgreSQL."""
    query = "SELECT md5(coalesce(string_agg(t::text, '|' ORDER BY t.id), '')) FROM {} t"
    with engine.connect() as connection:
        return ":".join(
            connection.execute(text(query.format(table))).scalar_one() for table in CRM_TABLES
        )


def test_small_dataset_is_accepted_by_database_constraints(
    dataset: SyntheticDataset, migrated_database: str, db_engine: Engine, clean_tables: None
) -> None:
    write_dataset(dataset, migrated_database, batch_size=128)

    assert _counts(db_engine) == {
        "activities": CONFIG.activities,
        "notes": 0,
        "opportunities": 0,
        "contacts": CONFIG.contacts,
        "leads": CONFIG.leads,
        "accounts": CONFIG.accounts,
        "users": CONFIG.reps + CONFIG.admins,
    }
    with db_engine.connect() as connection:
        stored_lead_ids = set(connection.execute(text("SELECT id FROM leads")).scalars())
        lead_activities = connection.execute(
            text("SELECT count(*) FROM activities WHERE lead_id IS NOT NULL")
        ).scalar_one()
    assert stored_lead_ids == {lead["id"] for lead in dataset.leads}
    assert lead_activities == dataset.lead_activity_count


def test_rerun_without_reset_is_refused(
    dataset: SyntheticDataset, migrated_database: str, db_engine: Engine, clean_tables: None
) -> None:
    write_dataset(dataset, migrated_database)
    before = _checksum(db_engine)

    with pytest.raises(UnsafeTargetError, match="--reset"):
        write_dataset(dataset, migrated_database)

    assert _checksum(db_engine) == before


def test_reset_with_the_same_seed_reproduces_identical_rows(
    dataset: SyntheticDataset, migrated_database: str, db_engine: Engine, clean_tables: None
) -> None:
    write_dataset(dataset, migrated_database, reset=True)
    first = _checksum(db_engine)
    write_dataset(generate(CONFIG), migrated_database, reset=True)

    assert _checksum(db_engine) == first


def test_protected_development_data_blocks_write_and_reset(
    dataset: SyntheticDataset, migrated_database: str, db_engine: Engine, clean_tables: None
) -> None:
    with Session(db_engine) as session:
        create_account(session, name="RevIntel Co")

    with pytest.raises(UnsafeTargetError, match="RevIntel Co"):
        write_dataset(dataset, migrated_database, reset=True)

    assert _counts(db_engine)["accounts"] == 1
    assert _counts(db_engine)["leads"] == 0


def test_failed_write_leaves_nothing_behind(
    dataset: SyntheticDataset, migrated_database: str, db_engine: Engine, clean_tables: None
) -> None:
    broken = generate(CONFIG)
    broken.activities[-1]["lead_id"] = None  # violates ck_activities_at_least_one_parent
    broken.activities[-1]["account_id"] = None
    broken.activities[-1]["contact_id"] = None

    with pytest.raises(Exception, match="ck_activities_at_least_one_parent"):
        write_dataset(broken, migrated_database)

    assert set(_counts(db_engine).values()) == {0}
