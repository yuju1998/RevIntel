"""Deals against an account."""

import uuid
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import str_enum
from app.models.enums import OpportunityStage

if TYPE_CHECKING:
    from app.models.account import Account
    from app.models.user import User


class Opportunity(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "opportunities"
    __table_args__ = (CheckConstraint("amount >= 0", name="amount_non_negative"),)

    account_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("accounts.id", ondelete="RESTRICT"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    stage: Mapped[OpportunityStage] = mapped_column(
        str_enum(OpportunityStage, "opportunity_stage"), index=True
    )
    amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    expected_close_date: Mapped[date | None]
    description: Mapped[str | None] = mapped_column(Text)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )

    account: Mapped[Account] = relationship(back_populates="opportunities")
    owner: Mapped[User | None] = relationship(back_populates="opportunities")
