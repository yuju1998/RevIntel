"""Column mixins implementing the key and timestamp conventions in docs/DATABASE.md."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column


class UUIDPrimaryKeyMixin:
    # Generated application-side so IDs are known before flush.
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4, sort_order=-1)


class TimestampMixin:
    # Database-generated; callers may still set created_at explicitly (e.g. synthetic data).
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), sort_order=100
    )
    # Refreshed by SQLAlchemy only when an UPDATE is actually emitted.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), sort_order=100
    )
