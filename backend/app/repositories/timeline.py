"""Account timeline data access: activities and notes as one paginated stream.

Ordering, LIMIT and OFFSET are applied by PostgreSQL to the UNION ALL of both tables,
so pagination is correct across the combined result set. Python only loads the full rows
for the keys of the requested page and keeps the database's order.
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import String, func, literal_column, select, union_all
from sqlalchemy.orm import Session

from app.models.activity import Activity
from app.models.note import Note

ACTIVITY = "activity"
NOTE = "note"


class TimelineRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def find_account_timeline(
        self, account_id: uuid.UUID, *, limit: int, offset: int
    ) -> list[Activity | Note]:
        # event_timestamp exists only in this query (occurred_at / created_at).
        keys = union_all(
            select(
                literal_column(f"'{ACTIVITY}'", String).label("record_type"),
                Activity.id.label("id"),
                Activity.occurred_at.label("event_timestamp"),
            ).where(Activity.account_id == account_id),
            select(
                literal_column(f"'{NOTE}'", String).label("record_type"),
                Note.id.label("id"),
                Note.created_at.label("event_timestamp"),
            ).where(Note.account_id == account_id),
        ).subquery("timeline")
        page = self._session.execute(
            select(keys.c.record_type, keys.c.id)
            # Total order: equal timestamps are broken by type, then id.
            .order_by(keys.c.event_timestamp.desc(), keys.c.record_type.asc(), keys.c.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()

        activities = self._by_id(Activity, [row.id for row in page if row.record_type == ACTIVITY])
        notes = self._by_id(Note, [row.id for row in page if row.record_type == NOTE])
        records: list[Activity | Note] = []
        for row in page:
            record = (activities if row.record_type == ACTIVITY else notes).get(row.id)
            # Skipped only if the row was deleted between the two statements.
            if record is not None:
                records.append(record)
        return records

    def count_account_timeline(self, account_id: uuid.UUID) -> int:
        activities = (
            select(func.count())
            .select_from(Activity)
            .where(Activity.account_id == account_id)
            .scalar_subquery()
        )
        notes = (
            select(func.count())
            .select_from(Note)
            .where(Note.account_id == account_id)
            .scalar_subquery()
        )
        return self._session.scalar(select(activities + notes)) or 0

    def _by_id(
        self, model: type[Activity] | type[Note], ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, Activity | Note]:
        if not ids:
            return {}
        rows = self._session.scalars(select(model).where(model.id.in_(ids)))
        return {row.id: row for row in rows}
