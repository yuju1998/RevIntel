"""Application users. Authentication is implemented in a later phase."""

from typing import TYPE_CHECKING

from sqlalchemy import String, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import str_enum
from app.models.enums import UserRole

if TYPE_CHECKING:
    from app.models.account import Account
    from app.models.lead import Lead
    from app.models.opportunity import Opportunity


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    # Canonical lowercase; the future user/auth service normalizes before persistence.
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(str_enum(UserRole, "user_role"))
    is_active: Mapped[bool] = mapped_column(default=True, server_default=true())

    # passive_deletes: the database applies ON DELETE SET NULL to owned records.
    accounts: Mapped[list[Account]] = relationship(back_populates="owner", passive_deletes=True)
    leads: Mapped[list[Lead]] = relationship(back_populates="owner", passive_deletes=True)
    opportunities: Mapped[list[Opportunity]] = relationship(
        back_populates="owner", passive_deletes=True
    )
