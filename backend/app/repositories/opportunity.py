"""Opportunity data access. Repositories never commit or roll back; services own transactions."""

import uuid

from sqlalchemy.orm import Session

from app.models.opportunity import Opportunity


class OpportunityRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, opportunity: Opportunity) -> Opportunity:
        self._session.add(opportunity)
        self._session.flush()
        return opportunity

    def get(self, opportunity_id: uuid.UUID) -> Opportunity | None:
        return self._session.get(Opportunity, opportunity_id)
