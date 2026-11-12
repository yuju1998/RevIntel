"""Assistant API schemas (semantic search over an account's CRM text)."""

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import Field, StringConstraints

from app.models.enums import DocumentSourceType
from app.schemas.common import RequestModel, ResponseModel

SearchQuery = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
]


class SemanticSearchRequest(RequestModel):
    account_id: uuid.UUID
    query: SearchQuery
    limit: int = Field(default=5, ge=1, le=20)


class SemanticSearchResult(ResponseModel):
    source_type: DocumentSourceType
    source_id: uuid.UUID
    content: str
    # Cosine similarity between the query and the chunk (higher is more similar).
    similarity: float
    occurred_at: datetime


class SemanticSearchResponse(ResponseModel):
    account_id: uuid.UUID
    results: list[SemanticSearchResult]
