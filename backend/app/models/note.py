"""Free-text notes on accounts, leads and opportunities."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.account import Account
    from app.models.lead import Lead
    from app.models.opportunity import Opportunity
    from app.models.user import User


class Note(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "notes"
    __table_args__ = (
        CheckConstraint(
            "account_id IS NOT NULL OR lead_id IS NOT NULL OR opportunity_id IS NOT NULL",
            name="at_least_one_parent",
        ),
    )

    # Canonical account: set explicitly or derived from the opportunity by the service.
    # Lead-only notes keep it NULL.
    account_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("accounts.id", ondelete="RESTRICT"), index=True
    )
    lead_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("leads.id", ondelete="RESTRICT"), index=True
    )
    opportunity_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("opportunities.id", ondelete="RESTRICT"), index=True
    )
    author_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    # Stored exactly as submitted (formatting preserved).
    body: Mapped[str] = mapped_column(Text)

    # Child-side only (see Activity).
    account: Mapped[Account | None] = relationship()
    lead: Mapped[Lead | None] = relationship()
    opportunity: Mapped[Opportunity | None] = relationship()
    author: Mapped[User | None] = relationship()
