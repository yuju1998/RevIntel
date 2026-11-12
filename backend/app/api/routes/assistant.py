"""Assistant endpoints: account-scoped semantic search (no answer generation yet)."""

from fastapi import APIRouter

from app.api.auth import UNAUTHORIZED_RESPONSE, RequireUser
from app.api.deps import SemanticSearchServiceDep
from app.schemas.assistant import (
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
