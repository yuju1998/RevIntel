"""Embedded CRM text for account-scoped semantic search (see docs/DATABASE.md)."""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import EMBEDDING_DIMENSION
from app.db.base import Base
from app.db.mixins import UUIDPrimaryKeyMixin
from app.db.types import str_enum
from app.models.enums import DocumentEntityType, DocumentSourceType


class DocumentChunk(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        CheckConstraint("chunk_index >= 0", name="chunk_index_non_negative"),
        # Explicit name; its index also serves "all chunks of a source" when re-indexing.
        UniqueConstraint(
            "source_type",
            "source_id",
            "chunk_index",
            name="uq_document_chunks_source_type_source_id_chunk_index",
        ),
    )

    # Isolation key: every retrieval filters on it in SQL before similarity ranking.
    account_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    source_type: Mapped[DocumentSourceType] = mapped_column(
        str_enum(DocumentSourceType, "document_source_type")
    )
    # ID of the row in the table named by source_type. No FK (polymorphic): the indexer
    # removes chunks whose source was deleted or no longer resolves to an account.
    source_id: Mapped[uuid.UUID]
    entity_type: Mapped[DocumentEntityType] = mapped_column(
        str_enum(DocumentEntityType, "document_entity_type")
    )
    entity_id: Mapped[uuid.UUID]
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    # Unit-length (normalized) embedding; cosine distance via pgvector's <=>.
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION))
    embedding_model: Mapped[str] = mapped_column(String(255))
    # SHA-256 of the indexed input (account, chunks, model): unchanged sources are skipped.
    source_sha256: Mapped[str] = mapped_column(String(64))
    # Column `metadata`; the attribute name avoids shadowing Base.metadata.
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default=text("'{}'::jsonb")
    )
    # Business time of the source (activity occurred_at, note created_at, opportunity or
    # account updated_at); never null.
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), sort_order=100
    )
