"""Account-scoped semantic search over indexed CRM text (no answer generation)."""

import uuid

import structlog

from app.ai.embeddings import EmbeddingError, EmbeddingProvider
from app.core.exceptions import NotFoundError, SearchUnavailableError
from app.repositories.account import AccountRepository
from app.repositories.semantic_search import RetrievedChunk, SemanticSearchRepository

logger = structlog.get_logger("app.search")


class SemanticSearchService:
    def __init__(
        self,
        accounts: AccountRepository,
        search: SemanticSearchRepository,
        embeddings: EmbeddingProvider,
    ) -> None:
        self._accounts = accounts
        self._search = search
        self._embeddings = embeddings

    def search(self, account_id: uuid.UUID, query: str, *, limit: int) -> list[RetrievedChunk]:
        """The `limit` chunks of this account most similar to `query` (most similar first)."""
        if self._accounts.get(account_id) is None:
            raise NotFoundError(f"Account {account_id} not found")
        try:
            embedding = self._embeddings.embed_query(query)
        except EmbeddingError as exc:
            logger.warning("semantic_search_unavailable", reason=str(exc))
            raise SearchUnavailableError(str(exc)) from exc
        return self._search.search(
            account_id,
            embedding,
            embedding_model=self._embeddings.model_name,
            limit=limit,
        )
