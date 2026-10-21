"""Activity data access. Repositories never commit or roll back; services own transactions."""

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.activity import Activity


class ActivityRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, activity: Activity) -> Activity:
        self._session.add(activity)
        self._session.flush()
        return activity

    def find_for_lead(self, lead_id: uuid.UUID, *, limit: int, offset: int) -> Sequence[Activity]:
        stmt = (
            select(Activity)
            .where(Activity.lead_id == lead_id)
            .order_by(Activity.occurred_at.desc(), Activity.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return self._session.scalars(stmt).all()

    def count_for_lead(self, lead_id: uuid.UUID) -> int:
        stmt = select(func.count()).select_from(Activity).where(Activity.lead_id == lead_id)
        return self._session.scalar(stmt) or 0
