"""Read-only extraction of lead and lead-activity rows, shared by offline and online paths.

SQL only selects rows; every feature rule lives in `ml.features.feature_spec`.
"""

from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import Connection, Table, func, make_url, select

import app.models  # noqa: F401  # registers every table on Base.metadata
from app.core.config import DatabaseSettings
from app.db.base import Base
from app.db.session import create_db_engine
from ml.features.feature_spec import ACTIVITY_INPUT_COLUMNS
from ml.synthetic.writer import SAFE_DATABASE_SUFFIXES

LEADS: Table = Base.metadata.tables["leads"]
ACTIVITIES: Table = Base.metadata.tables["activities"]

# Lead columns: feature inputs plus the snapshot/label control columns (status,
# converted_at, updated_at), which only ml.features.snapshots reads.
LEAD_COLUMNS = (
    "id",
    "created_at",
    "lead_source",
    "industry",
    "company_size",
    "estimated_value",
    "status",
    "converted_at",
    "updated_at",
)
_LEAD_DATETIMES = ("created_at", "converted_at", "updated_at")
_ACTIVITY_DATETIMES = ("occurred_at", "created_at")


class UnsafeSourceError(RuntimeError):
    """The source database is not the synthetic CRM dataset this pipeline reads."""


def require_source_database(database_url: str) -> str:
    name = make_url(database_url).database or ""
    if not name.endswith(SAFE_DATABASE_SUFFIXES):
        raise UnsafeSourceError(
            f"Refusing to read features from database {name!r}: the name must end with "
            f"{' or '.join(SAFE_DATABASE_SUFFIXES)} (e.g. revintel_ml)."
        )
    return name


@contextmanager
def read_only_connection(database_url: str) -> Iterator[Connection]:
    """One REPEATABLE READ, READ ONLY transaction: consistent reads, writes impossible."""
    require_source_database(database_url)
    engine = create_db_engine(DatabaseSettings(_env_file=None, database_url=database_url))
    try:
        with engine.connect() as connection:
            connection = connection.execution_options(
                isolation_level="REPEATABLE READ", postgresql_readonly=True
            )
            with connection.begin():
                yield connection
    finally:
        engine.dispose()


def _utc_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.as_unit("us")
    return frame


def leads_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Typed lead frame (`id` becomes `lead_id`) from database or in-memory rows."""
    frame = pd.DataFrame([[row[c] for c in LEAD_COLUMNS] for row in rows], columns=LEAD_COLUMNS)
    frame = _utc_columns(frame, _LEAD_DATETIMES).rename(columns={"id": "lead_id"})
    frame["lead_id"] = frame["lead_id"].astype(str)
    return frame


def activities_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Typed lead-activity frame; rows without a lead are dropped."""
    frame = pd.DataFrame(
        [[row[c] for c in ACTIVITY_INPUT_COLUMNS] for row in rows if row["lead_id"] is not None],
        columns=ACTIVITY_INPUT_COLUMNS,
    )
    frame = _utc_columns(frame, _ACTIVITY_DATETIMES)
    frame["lead_id"] = frame["lead_id"].astype(str)
    return frame


def load_leads(
    connection: Connection, lead_ids: Collection[UUID | str] | None = None
) -> pd.DataFrame:
    stmt = select(*(LEADS.c[name] for name in LEAD_COLUMNS)).order_by(LEADS.c.id)
    if lead_ids is not None:
        stmt = stmt.where(LEADS.c.id.in_([UUID(str(i)) for i in lead_ids]))
    return leads_frame(connection.execute(stmt).mappings().all())


def load_lead_activities(
    connection: Connection,
    lead_ids: Collection[UUID | str] | None = None,
    until: datetime | None = None,
) -> pd.DataFrame:
    """Lead activities; with `until`, only those known by then (the same availability rule
    compute_features enforces, applied early to read less)."""
    stmt = (
        select(*(ACTIVITIES.c[name] for name in ACTIVITY_INPUT_COLUMNS))
        .where(ACTIVITIES.c.lead_id.is_not(None))
        .order_by(ACTIVITIES.c.lead_id, ACTIVITIES.c.occurred_at, ACTIVITIES.c.id)
    )
    if lead_ids is not None:
        stmt = stmt.where(ACTIVITIES.c.lead_id.in_([UUID(str(i)) for i in lead_ids]))
    if until is not None:
        if until.tzinfo is None:
            raise ValueError("until must be timezone-aware")
        stmt = stmt.where(ACTIVITIES.c.occurred_at <= until, ACTIVITIES.c.created_at <= until)
    return activities_frame(connection.execute(stmt).mappings().all())


def source_row_counts(connection: Connection) -> dict[str, int]:
    count = func.count()
    return {
        "leads": connection.execute(select(count).select_from(LEADS)).scalar_one(),
        "activities": connection.execute(select(count).select_from(ACTIVITIES)).scalar_one(),
        "lead_activities": connection.execute(
            select(count).select_from(ACTIVITIES).where(ACTIVITIES.c.lead_id.is_not(None))
        ).scalar_one(),
    }
