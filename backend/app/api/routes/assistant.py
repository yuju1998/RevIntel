"""Assistant endpoints: account-scoped semantic search and the CRM Copilot."""

from fastapi import APIRouter

from app.api.auth import UNAUTHORIZED_RESPONSE, RequireUser
from app.api.deps import AssistantQueryServiceDep, SemanticSearchServiceDep
from app.schemas.assistant import (
    AssistantQueryRequest,
    AssistantQueryResponse,
    AssistantSource,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResult,
)

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post(
    "/search",
    response_model=SemanticSearchResponse,
    dependencies=[RequireUser],
    responses={
        **UNAUTHORIZED_RESPONSE,
        404: {"description": "Account not found"},
        503: {"description": "Semantic search is currently unavailable"},
    },
)
def search(
    request: SemanticSearchRequest, service: SemanticSearchServiceDep
) -> SemanticSearchResponse:
    """The account's CRM text most similar to the query. Only that account is searched."""
    chunks = service.search(request.account_id, request.query, limit=request.limit)
    return SemanticSearchResponse(
        account_id=request.account_id,
        results=[
            SemanticSearchResult(
                source_type=chunk.source_type,
                source_id=chunk.source_id,
                content=chunk.content,
                similarity=chunk.similarity,
                occurred_at=chunk.occurred_at,
            )
            for chunk in chunks
        ],
    )


@router.post(
    "/query",
    response_model=AssistantQueryResponse,
    dependencies=[RequireUser],
    responses={
        **UNAUTHORIZED_RESPONSE,
        404: {"description": "Account not found"},
        503: {"description": "The assistant is currently unavailable"},
    },
)
def query(
    request: AssistantQueryRequest, service: AssistantQueryServiceDep
) -> AssistantQueryResponse:
    """A grounded answer from this account's CRM records, with validated citations."""
    result = service.query(request.account_id, request.question)
    return AssistantQueryResponse(
        answer=result.answer,
        insufficient_evidence=result.insufficient_evidence,
        sources=[
            AssistantSource(source_id=source.source_id, source_type=source.source_type)
            for source in result.sources
        ],
    )
