"""Lead data access. Repositories never commit or roll back; services own transactions."""

import uuid
from collections.abc import Sequence

from sqlalchemy import ColumnElement, func, select
from sqlalchemy.orm import Session

from app.models.enums import LeadSource, LeadStatus
from app.models.lead import Lead


class LeadRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, lead: Lead) -> Lead:
        self._session.add(lead)
        self._session.flush()
        return lead

    def get(self, lead_id: uuid.UUID) -> Lead | None:
        return self._session.get(Lead, lead_id)

    def find(
        self,
        *,
        status: LeadStatus | None,
        lead_source: LeadSource | None,
        limit: int,
        offset: int,
    ) -> Sequence[Lead]:
        stmt = (
            select(Lead)
            .where(*self._filters(status=status, lead_source=lead_source))
            .order_by(Lead.created_at.desc(), Lead.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return self._session.scalars(stmt).all()

    def count(self, *, status: LeadStatus | None, lead_source: LeadSource | None) -> int:
        stmt = (
            select(func.count())
            .select_from(Lead)
            .where(*self._filters(status=status, lead_source=lead_source))
        )
        return self._session.scalar(stmt) or 0

    @staticmethod
    def _filters(
        *, status: LeadStatus | None, lead_source: LeadSource | None
    ) -> list[ColumnElement[bool]]:
        filters: list[ColumnElement[bool]] = []
        if status is not None:
            filters.append(Lead.status == status)
        if lead_source is not None:
            filters.append(Lead.lead_source == lead_source)
        return filters
