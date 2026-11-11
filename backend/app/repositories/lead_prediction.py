"""Lead prediction data access. Append-only; repositories never commit."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.lead_prediction import LeadPrediction


class LeadPredictionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, prediction: LeadPrediction) -> LeadPrediction:
        self._session.add(prediction)
        self._session.flush()
        return prediction

    def latest_for_lead(self, lead_id: uuid.UUID) -> LeadPrediction | None:
        """Newest score by prediction timestamp (ties: newest created, then id)."""
        stmt = (
            select(LeadPrediction)
            .where(LeadPrediction.lead_id == lead_id)
            .order_by(
                LeadPrediction.prediction_timestamp.desc(),
                LeadPrediction.created_at.desc(),
                LeadPrediction.id.desc(),
            )
            .limit(1)
        )
        return self._session.scalars(stmt).first()
