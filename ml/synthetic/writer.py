"""Write a synthetic dataset to a guarded database in one transaction.

Guards (all checked before any write):
- the database name must end in `_test` or `_ml` (never the dev `revintel`);
- the schema must be at the Alembic head;
- the database must not contain the protected real records (admin@example.com,
  "RevIntel Co");
- a non-empty target is refused unless `reset=True`, which truncates the CRM tables of
  that (guarded) database inside the same transaction.
"""

from collections.abc import Callable
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, insert, make_url, text

import app.models  # noqa: F401  # registers every table on Base.metadata
from app.core.config import DatabaseSettings
from app.db.base import Base
from app.db.session import create_db_engine
from ml.synthetic.generator import SyntheticDataset

SAFE_DATABASE_SUFFIXES = ("_test", "_ml")
PROTECTED_USER_EMAIL = "admin@example.com"
PROTECTED_ACCOUNT_NAME = "RevIntel Co"
CRM_TABLES = ("activities", "notes", "opportunities", "contacts", "leads", "accounts", "users")
# Derived from CRM rows (scores reference leads, chunks reference accounts), so a reset
# removes them as well.
DERIVED_TABLES = ("lead_predictions", "document_chunks")
ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


class UnsafeTargetError(RuntimeError):
    """The target database is not one the generator may write to."""


def require_safe_database_url(database_url: str) -> str:
    """Return the database name, or raise if it is not an explicitly safe target."""
    name = make_url(database_url).database or ""
    if not name.endswith(SAFE_DATABASE_SUFFIXES):
        raise UnsafeTargetError(
            f"Refusing to write synthetic data to database {name!r}: the name must end with "
            f"{' or '.join(SAFE_DATABASE_SUFFIXES)} (e.g. revintel_ml or revintel_test)."
        )
    return name


def _require_current_schema(connection: Connection) -> None:
    head = ScriptDirectory.from_config(Config(str(ALEMBIC_INI))).get_current_head()
    current = connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
    if current != head:
        raise UnsafeTargetError(
            f"Target schema is at {current!r}, not head {head!r}: run `alembic upgrade head` "
            "against it first."
        )


def _refuse_protected_data(connection: Connection) -> None:
    protected = connection.execute(
        text(
            "SELECT (SELECT count(*) FROM users WHERE email = :email) "
            "+ (SELECT count(*) FROM accounts WHERE name = :account)"
        ),
        {"email": PROTECTED_USER_EMAIL, "account": PROTECTED_ACCOUNT_NAME},
    ).scalar()
    if protected:
        raise UnsafeTargetError(
            f"Target contains {PROTECTED_USER_EMAIL} or {PROTECTED_ACCOUNT_NAME!r}; it looks "
            "like real development data. Refusing to write or reset it."
        )


def _row_counts(connection: Connection) -> dict[str, int]:
    return {
        table: int(connection.execute(text(f"SELECT count(*) FROM {table}")).scalar() or 0)
        for table in CRM_TABLES
    }


def write_dataset(
    dataset: SyntheticDataset,
    database_url: str,
    *,
    reset: bool = False,
    batch_size: int = 5000,
    progress: Callable[[str], None] = lambda _: None,
) -> None:
    name = require_safe_database_url(database_url)
    engine = create_db_engine(DatabaseSettings(_env_file=None, database_url=database_url))
    try:
        # One transaction: either the whole dataset is written or nothing is.
        with engine.begin() as connection:
            _require_current_schema(connection)
            _refuse_protected_data(connection)
            existing = {k: v for k, v in _row_counts(connection).items() if v}
            if existing:
                if not reset:
                    raise UnsafeTargetError(
                        f"Database {name!r} already has CRM rows {existing}; pass --reset to "
                        "replace them."
                    )
                tables = ", ".join((*DERIVED_TABLES, *CRM_TABLES))
                connection.execute(text(f"TRUNCATE {tables}"))
                progress(f"  reset: truncated CRM and derived tables in {name}")
            for table_name, rows in dataset.tables():
                table = Base.metadata.tables[table_name]
                for start in range(0, len(rows), batch_size):
                    connection.execute(insert(table), rows[start : start + batch_size])
                progress(f"  wrote {table_name:<12} {len(rows):>8,} rows")
    finally:
        engine.dispose()
