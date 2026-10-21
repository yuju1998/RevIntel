"""Calls, emails, meetings, demos and follow-ups. Append-only (no updated_at)."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.mixins import UUIDPrimaryKeyMixin
from app.db.types import str_enum
from app.models.enums import ActivityDirection, ActivityType

if TYPE_CHECKING:
    from app.models.account import Account
    from app.models.contact import Contact
    from app.models.lead import Lead
    from app.models.opportunity import Opportunity
    from app.models.user import User


class Activity(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "activities"
    __table_args__ = (
        # Several parents may be set at once; there is no "exactly one" rule.
        CheckConstraint(
            "account_id IS NOT NULL OR lead_id IS NOT NULL "
            "OR opportunity_id IS NOT NULL OR contact_id IS NOT NULL",
            name="at_least_one_parent",
        ),
    )

    # Canonical account: set explicitly or derived from the opportunity/contact by the
    # service. Lead-only activities keep it NULL.
    account_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("accounts.id", ondelete="RESTRICT"), index=True
    )
    lead_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("leads.id", ondelete="RESTRICT"), index=True
    )
    opportunity_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("opportunities.id", ondelete="RESTRICT"), index=True
    )
    contact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("contacts.id", ondelete="RESTRICT"), index=True
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    activity_type: Mapped[ActivityType] = mapped_column(str_enum(ActivityType, "activity_type"))
    direction: Mapped[ActivityDirection] = mapped_column(
        str_enum(ActivityDirection, "activity_direction")
    )
    subject: Mapped[str | None] = mapped_column(String(255))
    body: Mapped[str | None] = mapped_column(Text)
    # When it actually happened (business time); the ML time filter.
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    # When it was recorded.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), sort_order=100
    )

    # Child-side only: parent models have no collections, so the ORM never nulls these
    # FKs itself and the database's RESTRICT / SET NULL always apply.
    account: Mapped[Account | None] = relationship()
    lead: Mapped[Lead | None] = relationship()
    opportunity: Mapped[Opportunity | None] = relationship()
    contact: Mapped[Contact | None] = relationship()
    created_by: Mapped[User | None] = relationship()
