"""Index CRM text into document_chunks for account-scoped semantic search.

Rerunnable and idempotent:
- each source's indexed input (account, entity, timestamp, metadata, chunk texts, model)
  is hashed into `source_sha256`; unchanged sources are skipped without embedding;
- a new or changed source has all its chunks replaced (whatever account they were under);
- chunks whose source was deleted, or no longer resolves to the account, are removed;
- the unique (source_type, source_id, chunk_index) constraint rules out duplicates.

Sources are read in pages; each page is one transaction and one batched embedding call.
"""

import hashlib
import json
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.ai.chunking import chunk_text
from app.ai.crm_documents import (
    SourceDocument,
    account_document,
    activity_document,
    note_document,
    opportunity_document,
)
from app.ai.embeddings import EmbeddingProvider
from app.models.document_chunk import DocumentChunk
from app.models.enums import DocumentSourceType
from app.repositories.crm_sources import CrmSourceRepository
from app.repositories.document_chunk import DocumentChunkRepository
from app.services.transaction import write_transaction


@dataclass
class SourceTypeStats:
    sources: int = 0  # source rows read
    skipped: int = 0  # no resolvable account, or no text
    unchanged: int = 0
    indexed: int = 0  # new or changed sources (re-embedded)
    chunks: int = 0  # chunks written
    removed: int = 0  # chunks of sources that disappeared or moved out of scope


@dataclass
class IndexingStats:
    by_type: dict[str, SourceTypeStats] = field(default_factory=dict)
    texts_embedded: int = 0


def source_hash(document: SourceDocument, chunks: list[str], embedding_model: str) -> str:
    payload = {
        "account_id": str(document.account_id),
        "entity": [document.entity_type.value, str(document.entity_id)],
        "occurred_at": document.occurred_at.isoformat(),
        "metadata": document.metadata,
        "chunks": chunks,
        "embedding_model": embedding_model,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class CrmIndexingService:
    def __init__(
        self,
        session: Session,
        sources: CrmSourceRepository,
        chunks: DocumentChunkRepository,
        embeddings: EmbeddingProvider,
        *,
        page_size: int = 500,
        progress: Callable[[str], None] = lambda _: None,
    ) -> None:
        self._session = session
        self._sources = sources
        self._chunks = chunks
        self._embeddings = embeddings
        self._page_size = page_size
        self._progress = progress

    def index(self, *, account_id: uuid.UUID | None = None) -> IndexingStats:
        """Index every account (or only `account_id`); safe to rerun."""
        stats = IndexingStats()
        kwargs = {"account_id": account_id, "batch_size": self._page_size}
        pipelines: list[tuple[DocumentSourceType, Iterator[list], Callable]] = [
            (DocumentSourceType.ACCOUNT, self._sources.accounts(**kwargs), account_document),
            (
                DocumentSourceType.OPPORTUNITY,
                self._sources.opportunities(**kwargs),
                opportunity_document,
            ),
            (DocumentSourceType.ACTIVITY, self._sources.activities(**kwargs), activity_document),
            (DocumentSourceType.NOTE, self._sources.notes(**kwargs), note_document),
        ]
        for source_type, pages, build in pipelines:
            type_stats = stats.by_type.setdefault(source_type.value, SourceTypeStats())
            indexed_ids: set[uuid.UUID] = set()
            for page in pages:
                documents = []
                for row in page:
                    type_stats.sources += 1
                    document = build(row)
                    # Scoped runs keep only documents that resolve to the requested account.
                    if document is None or (
                        account_id is not None and document.account_id != account_id
                    ):
                        type_stats.skipped += 1
                        continue
                    documents.append(document)
                    indexed_ids.add(document.source_id)
                stats.texts_embedded += self._index_page(source_type, documents, type_stats)
            type_stats.removed += self._remove_stale(source_type, indexed_ids, account_id)
            self._progress(f"  {source_type.value.lower():<12} {type_stats}")
        return stats

    def _index_page(
        self,
        source_type: DocumentSourceType,
        documents: list[SourceDocument],
        type_stats: SourceTypeStats,
    ) -> int:
        model = self._embeddings.model_name
        prepared = []
        for document in documents:
            pieces = chunk_text(
                document.text,
                count_tokens=self._embeddings.count_tokens,
                max_tokens=self._embeddings.max_tokens,
            )
            prepared.append((document, pieces, source_hash(document, pieces, model)))
        stored = self._chunks.source_hashes(source_type, [d.source_id for d in documents])
        changed = [item for item in prepared if stored.get(item[0].source_id) != item[2]]
        type_stats.unchanged += len(prepared) - len(changed)
        if not changed:
            return 0

        # One batched embedding call per page; identical texts are embedded once.
        unique_texts = list(dict.fromkeys(piece for _, pieces, _ in changed for piece in pieces))
        vectors = dict(
            zip(unique_texts, self._embeddings.embed_documents(unique_texts), strict=True)
        )
        with write_transaction(self._session):
            for document, pieces, digest in changed:
                self._chunks.replace_source(
                    source_type,
                    document.source_id,
                    [
                        DocumentChunk(
                            account_id=document.account_id,
                            source_type=source_type,
                            source_id=document.source_id,
                            entity_type=document.entity_type,
                            entity_id=document.entity_id,
                            chunk_index=index,
                            content=piece,
                            embedding=vectors[piece],
                            embedding_model=model,
                            source_sha256=digest,
                            chunk_metadata={**document.metadata, "chunk_count": len(pieces)},
                            occurred_at=document.occurred_at,
                        )
                        for index, piece in enumerate(pieces)
                    ],
                )
                type_stats.indexed += 1
                type_stats.chunks += len(pieces)
        return len(unique_texts)

    def _remove_stale(
        self,
        source_type: DocumentSourceType,
        indexed_ids: Iterable[uuid.UUID],
        account_id: uuid.UUID | None,
    ) -> int:
        stale = self._chunks.indexed_source_ids(source_type, account_id=account_id) - set(
            indexed_ids
        )
        if not stale:
            return 0
        with write_transaction(self._session):
            return self._chunks.delete_sources(source_type, stale)
