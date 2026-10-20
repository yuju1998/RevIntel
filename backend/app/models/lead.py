"""Prospects not yet tied to an account; the subject of lead-conversion predictions."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import str_enum
from app.models.enums import LeadSource, LeadStatus

if TYPE_CHECKING:
    from app.models.user import User


class Lead(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "leads"
    __table_args__ = (
        CheckConstraint("company_size >= 0", name="company_size_non_negative"),
        CheckConstraint("estimated_value >= 0", name="estimated_value_non_negative"),
        # One-directional on purpose: other statuses may still carry converted_at.
        CheckConstraint(
            "status <> 'CONVERTED' OR converted_at IS NOT NULL",
            name="converted_requires_converted_at",
        ),
    )

    company_name: Mapped[str] = mapped_column(String(255))
    contact_name: Mapped[str] = mapped_column(String(255))
    # Not unique; stored as entered.
    email: Mapped[str | None] = mapped_column(String(320))
    lead_source: Mapped[LeadSource] = mapped_column(str_enum(LeadSource, "lead_source"), index=True)
    industry: Mapped[str | None] = mapped_column(String(255))
    company_size: Mapped[int | None]
    estimated_value: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    # Never an ML feature (final status leaks the label).
    status: Mapped[LeadStatus] = mapped_column(str_enum(LeadStatus, "lead_status"), index=True)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    # ML label source. Never an ML feature.
    converted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), sort_order=101)

    owner: Mapped[User | None] = relationship(back_populates="leads")
