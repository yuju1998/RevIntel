"""Customer companies."""

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.contact import Contact
    from app.models.opportunity import Opportunity
    from app.models.user import User


class Account(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "accounts"
    __table_args__ = (
        CheckConstraint("employee_count >= 0", name="employee_count_non_negative"),
        CheckConstraint("annual_revenue >= 0", name="annual_revenue_non_negative"),
    )

    # Not unique: different customers can share a name.
    name: Mapped[str] = mapped_column(String(255))
    industry: Mapped[str | None] = mapped_column(String(255))
    employee_count: Mapped[int | None]
    annual_revenue: Mapped[Decimal | None] = mapped_column(Numeric(16, 2))
    website: Mapped[str | None] = mapped_column(String(2048))
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )

    owner: Mapped[User | None] = relationship(back_populates="accounts")
    # passive_deletes="all": the ORM never touches children, so ON DELETE RESTRICT applies.
    contacts: Mapped[list[Contact]] = relationship(back_populates="account", passive_deletes="all")
    opportunities: Mapped[list[Opportunity]] = relationship(
        back_populates="account", passive_deletes="all"
    )
