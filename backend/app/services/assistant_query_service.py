"""CRM Copilot: grounded answers from one account's retrieved CRM records.

question -> account-scoped semantic retrieval (SemanticSearchService, unchanged)
         -> prompt with exactly those records -> LLM provider -> validated citations.
"""

import uuid
from dataclasses import dataclass

import structlog

from app.ai.copilot import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    CitedSource,
    build_prompt,
    validate_answer,
)
from app.ai.llm import EvidenceRecord, LLMError, LLMProvider
from app.core.exceptions import AssistantUnavailableError, SearchUnavailableError
from app.repositories.semantic_search import RetrievedChunk
from app.services.semantic_search_service import SemanticSearchService

logger = structlog.get_logger("app.assistant")

# Records given to the model; matches the Recall@5 used by the RAG evaluation.
RETRIEVAL_LIMIT = 5


@dataclass(frozen=True)
class CopilotResult:
    answer: str
    sources: list[CitedSource]
    insufficient_evidence: bool
    # For evaluation and logs; not part of the API response.
    retrieved: list[RetrievedChunk]
    model_citations: list[str]
    rejected_citations: list[str]


class AssistantQueryService:
    def __init__(self, search: SemanticSearchService, llm: LLMProvider) -> None:
        self._search = search
        self._llm = llm

    def query(self, account_id: uuid.UUID, question: str) -> CopilotResult:
        try:
            # Raises NotFoundError for an unknown account; filtered by account in SQL.
            chunks = self._search.search(account_id, question, limit=RETRIEVAL_LIMIT)
        except SearchUnavailableError as exc:
            self._unavailable(account_id, exc.reason)
        if not chunks:
            return CopilotResult(INSUFFICIENT_EVIDENCE_ANSWER, [], True, [], [], [])

        records = [
            EvidenceRecord(chunk.source_type, chunk.source_id, chunk.occurred_at, chunk.content)
            for chunk in chunks
        ]
        try:
            raw = self._llm.generate(build_prompt(question, records))
        except LLMError as exc:
            self._unavailable(account_id, str(exc))
        validated = validate_answer(raw, records)
        if validated.rejected_citations:
            logger.warning(
                "assistant_citations_rejected",
                account_id=str(account_id),
                provider=self._llm.name,
                rejected=len(validated.rejected_citations),
            )
        return CopilotResult(
            answer=validated.answer,
            sources=validated.sources,
            insufficient_evidence=validated.insufficient_evidence,
            retrieved=chunks,
            model_citations=validated.model_citations,
            rejected_citations=validated.rejected_citations,
        )

    def _unavailable(self, account_id: uuid.UUID, reason: str) -> None:
        logger.warning("assistant_unavailable", account_id=str(account_id), reason=reason)
        raise AssistantUnavailableError(reason)
