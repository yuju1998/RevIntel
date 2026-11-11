"""Lead score API schemas."""

import uuid
from datetime import datetime

from app.models.enums import LeadTier
from app.schemas.common import ResponseModel


class LeadScoreRead(ResponseModel):
    lead_id: uuid.UUID
    # P(converts in (T, T + 30d]) from the champion model.
    conversion_probability: float
    # Product-policy tier stored when the score was made; never recomputed on read.
    lead_tier: LeadTier
    model_name: str
    model_version: str
    prediction_timestamp: datetime
    inference_ms: float | None
