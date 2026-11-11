"""Conversion scores produced by the lead conversion model. Append-only."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import UUIDPrimaryKeyMixin
from app.db.types import str_enum
from app.models.enums import LeadTier


class LeadPrediction(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "lead_predictions"
    __table_args__ = (
        CheckConstraint(
            "conversion_probability >= 0 AND conversion_probability <= 1",
            name="conversion_probability_range",
        ),
        CheckConstraint("inference_ms >= 0", name="inference_ms_non_negative"),
        # Shortened name: the descriptive one exceeds PostgreSQL's 63-character limit.
        # Its index also serves "latest prediction for a lead" and the lead_id FK.
        UniqueConstraint(
            "lead_id",
            "prediction_timestamp",
            "model_name",
            "model_version",
            name="uq_lead_predictions_lead_id_timestamp_model",
        ),
    )

    lead_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"))
    # Feature cut-off time T; the score predicts conversion in (T, T + 30d].
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    # A probability, not money.
    conversion_probability: Mapped[float] = mapped_column(Float)
    # Product-policy tier stored at scoring time; never recomputed on read.
    lead_tier: Mapped[LeadTier] = mapped_column(str_enum(LeadTier, "lead_tier"))
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(64))
    # The exact model input (wire form) used for this score.
    features: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # Model inference time only (not feature computation or persistence).
    inference_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), sort_order=100
    )
