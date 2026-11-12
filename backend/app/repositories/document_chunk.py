"""Document chunk writes for the CRM indexer. Repositories never commit."""

import uuid
from collections.abc import Collection, Iterable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk
from app.models.enums import DocumentSourceType

# Keeps IN (...) lists to a reasonable number of bind parameters.
_ID_BATCH = 1000


def _batches(ids: Collection[uuid.UUID]) -> Iterable[list[uuid.UUID]]:
    items = list(ids)
    for start in range(0, len(items), _ID_BATCH):
        yield items[start : start + _ID_BATCH]


class DocumentChunkRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def source_hashes(
        self, source_type: DocumentSourceType, source_ids: Collection[uuid.UUID]
    ) -> dict[uuid.UUID, str]:
        """Stored `source_sha256` per already-indexed source (one value per source)."""
        hashes: dict[uuid.UUID, str] = {}
        for batch in _batches(source_ids):
            stmt = select(DocumentChunk.source_id, DocumentChunk.source_sha256).where(
                DocumentChunk.source_type == source_type,
                DocumentChunk.source_id.in_(batch),
                DocumentChunk.chunk_index == 0,
            )
            hashes.update({row.source_id: row.source_sha256 for row in self._session.execute(stmt)})
        return hashes

    def indexed_source_ids(
        self, source_type: DocumentSourceType, *, account_id: uuid.UUID | None = None
    ) -> set[uuid.UUID]:
        stmt = select(DocumentChunk.source_id).where(DocumentChunk.source_type == source_type)
        if account_id is not None:
            stmt = stmt.where(DocumentChunk.account_id == account_id)
        return set(self._session.scalars(stmt.distinct()))

    def replace_source(
        self,
        source_type: DocumentSourceType,
        source_id: uuid.UUID,
        chunks: list[DocumentChunk],
    ) -> None:
        """Delete every chunk of the source (whatever its account), then add `chunks`."""
        self._session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.source_type == source_type, DocumentChunk.source_id == source_id
            )
        )
        self._session.add_all(chunks)
        self._session.flush()

    def delete_sources(
        self, source_type: DocumentSourceType, source_ids: Collection[uuid.UUID]
    ) -> int:
        deleted = 0
        for batch in _batches(source_ids):
            result = self._session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.source_type == source_type,
                    DocumentChunk.source_id.in_(batch),
                )
            )
            deleted += result.rowcount
        return deleted
