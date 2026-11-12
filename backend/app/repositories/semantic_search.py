"""Account-scoped semantic retrieval over document_chunks.

There is deliberately no unscoped search: `account_id` is a required argument and the
account predicate is part of the same SQL statement that ranks by cosine distance, so
another account's chunks are never candidates (no filtering after retrieval).
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk
from app.models.enums import DocumentSourceType


@dataclass(frozen=True)
class RetrievedChunk:
    account_id: uuid.UUID
    source_type: DocumentSourceType
    source_id: uuid.UUID
    content: str
    occurred_at: datetime
    similarity: float


def account_search_statement(
    account_id: uuid.UUID,
    query_embedding: Sequence[float],
    *,
    embedding_model: str,
    limit: int,
) -> Select:
    """Nearest chunks of ONE account, by cosine similarity (1 - cosine distance)."""
    if not isinstance(account_id, uuid.UUID):
        raise ValueError("account_id is required for semantic search")
    distance = DocumentChunk.embedding.cosine_distance(list(query_embedding))
    return (
        select(
            DocumentChunk.account_id,
            DocumentChunk.source_type,
            DocumentChunk.source_id,
            DocumentChunk.content,
            DocumentChunk.occurred_at,
            (1 - distance).label("similarity"),
        )
        .where(
            DocumentChunk.account_id == account_id,
            DocumentChunk.embedding_model == embedding_model,
        )
        .order_by(distance, DocumentChunk.id)
        .limit(limit)
    )


class SemanticSearchRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def search(
        self,
        account_id: uuid.UUID,
        query_embedding: Sequence[float],
        *,
        embedding_model: str,
        limit: int,
    ) -> list[RetrievedChunk]:
        stmt = account_search_statement(
            account_id, query_embedding, embedding_model=embedding_model, limit=limit
        )
        return [
            RetrievedChunk(
                account_id=row.account_id,
                source_type=row.source_type,
                source_id=row.source_id,
                content=row.content,
                occurred_at=row.occurred_at,
                similarity=float(row.similarity),
            )
            for row in self._session.execute(stmt)
        ]
