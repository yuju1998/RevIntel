"""CRM semantic search: chunking, documents, embeddings, retrieval SQL, service, routes."""

import math
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from app.ai.chunking import chunk_text
from app.ai.crm_documents import (
    AccountSource,
    ActivitySource,
    NoteSource,
    OpportunitySource,
    account_document,
    activity_document,
    note_document,
    opportunity_document,
    resolve_account,
)
from app.ai.embeddings import EmbeddingError, SentenceTransformerProvider, check_vectors
from app.api.auth import get_current_user
from app.api.deps import get_semantic_search_service
from app.core.config import Settings
from app.core.exceptions import NotFoundError, SearchUnavailableError
from app.main import create_app
from app.models.enums import DocumentEntityType, DocumentSourceType
from app.repositories.semantic_search import RetrievedChunk, account_search_statement
from app.services.semantic_search_service import SemanticSearchService
from tests.embedding_fakes import FakeEmbeddingProvider
from tests.unit.helpers import make_user

T = datetime(2026, 10, 1, 12, tzinfo=UTC)
ACCOUNT, OTHER = uuid.uuid4(), uuid.uuid4()


def words(text: str) -> int:
    return len(text.split())


# --- chunking ---------------------------------------------------------------------------------


def test_short_text_is_one_unchanged_chunk() -> None:
    text = "Call (inbound): Intro call\nAsked about pricing tiers."

    assert chunk_text(text, count_tokens=words, max_tokens=254) == [text]
    assert chunk_text("   ", count_tokens=words, max_tokens=254) == []


def test_long_text_windows_respect_the_token_limit_and_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(25))

    chunks = chunk_text(text, count_tokens=words, max_tokens=10, overlap_tokens=3)

    assert all(words(chunk) <= 10 for chunk in chunks)
    assert [chunk.split()[0] for chunk in chunks] == ["w0", "w7", "w14", "w21"]  # 3 repeated
    assert chunks[-1].split()[-1] == "w24"  # nothing lost


def test_windows_are_slices_that_keep_inner_whitespace() -> None:
    text = "alpha beta\ngamma delta\nepsilon zeta"

    chunks = chunk_text(text, count_tokens=words, max_tokens=4, overlap_tokens=0)

    assert chunks == ["alpha beta\ngamma delta", "epsilon zeta"]


def test_a_word_longer_than_the_limit_is_split_rather_than_truncated() -> None:
    chunks = chunk_text("x" * 23, count_tokens=len, max_tokens=10)

    assert "".join(chunks) == "x" * 23
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_tokenizer_count_is_authoritative_over_word_sums() -> None:
    # Joining words can cost more tokens than the words alone (here: one per gap).
    def gaps(text: str) -> int:
        return len(text.split()) * 2 - 1

    chunks = chunk_text(" ".join(["a"] * 12), count_tokens=gaps, max_tokens=9, overlap_tokens=0)

    assert all(gaps(chunk) <= 9 for chunk in chunks)


def test_invalid_limit() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", count_tokens=words, max_tokens=0)


# --- documents and account resolution ---------------------------------------------------------


def _activity(**fields: Any) -> ActivitySource:
    values: dict[str, Any] = {
        "id": uuid.uuid4(),
        "activity_type": "CALL",
        "direction": "INBOUND",
        "subject": "Intro call",
        "body": "Asked about pricing.",
        "occurred_at": T,
        "account_id": None,
        "opportunity_id": None,
        "opportunity_account_id": None,
        "contact_id": None,
        "contact_account_id": None,
        "lead_id": None,
    }
    return ActivitySource(**{**values, **fields})


def test_resolve_account_requires_exactly_one_account() -> None:
    assert resolve_account(ACCOUNT, None, ACCOUNT) == ACCOUNT
    assert resolve_account(None, None) is None
    assert resolve_account(ACCOUNT, OTHER) is None  # conflicting references


def test_activity_with_a_direct_account() -> None:
    document = activity_document(_activity(account_id=ACCOUNT))

    assert document is not None
    assert document.account_id == ACCOUNT
    assert (document.entity_type, document.entity_id) == (DocumentEntityType.ACCOUNT, ACCOUNT)
    assert document.text == "Call (inbound): Intro call\nAsked about pricing."
    assert document.occurred_at == T
    assert document.metadata == {"activity_type": "CALL", "direction": "INBOUND"}


def test_activity_account_is_resolved_through_its_opportunity_or_contact() -> None:
    opportunity, contact = uuid.uuid4(), uuid.uuid4()

    via_opportunity = activity_document(
        _activity(opportunity_id=opportunity, opportunity_account_id=ACCOUNT)
    )
    via_contact = activity_document(_activity(contact_id=contact, contact_account_id=ACCOUNT))

    assert via_opportunity is not None and via_contact is not None
    assert via_opportunity.account_id == via_contact.account_id == ACCOUNT
    assert via_opportunity.entity_type == DocumentEntityType.OPPORTUNITY
    assert via_opportunity.entity_id == opportunity
    assert (via_contact.entity_type, via_contact.entity_id) == (DocumentEntityType.CONTACT, contact)


@pytest.mark.parametrize(
    "fields",
    [
        {"lead_id": uuid.uuid4()},  # leads have no account in the schema
        {},  # no parent at all
        {"account_id": ACCOUNT, "contact_id": uuid.uuid4(), "contact_account_id": OTHER},
        {"account_id": ACCOUNT, "subject": None, "body": "  "},  # nothing to index
    ],
    ids=["lead-only", "no-parent", "conflicting-accounts", "no-text"],
)
def test_activities_without_a_single_resolved_account_or_text_are_skipped(
    fields: dict[str, Any],
) -> None:
    assert activity_document(_activity(**fields)) is None


def test_note_documents() -> None:
    opportunity = uuid.uuid4()
    note = NoteSource(
        id=uuid.uuid4(),
        body=" Pricing concern raised. ",
        created_at=T,
        account_id=None,
        opportunity_id=opportunity,
        opportunity_account_id=ACCOUNT,
        lead_id=None,
    )

    document = note_document(note)

    assert document is not None and document.account_id == ACCOUNT
    assert document.entity_type == DocumentEntityType.OPPORTUNITY
    assert document.text == "Pricing concern raised."
    assert (
        note_document(note.__class__(**{**note.__dict__, "opportunity_account_id": None})) is None
    )


def test_opportunity_and_account_documents() -> None:
    opportunity = OpportunitySource(
        id=uuid.uuid4(),
        account_id=ACCOUNT,
        name="Renewal 2027",
        stage="CLOSED_WON",
        amount=Decimal("12000.00"),
        expected_close_date=date(2027, 1, 31),
        description="Customer wants a multi-year discount.",
        updated_at=T,
    )
    account = AccountSource(
        id=ACCOUNT,
        name="Acme",
        industry="Software",
        employee_count=40,
        website=None,
        updated_at=T,
    )

    assert opportunity_document(opportunity).text == (
        "Opportunity: Renewal 2027\nStage: Closed won\nAmount: 12000.00\n"
        "Expected close: 2027-01-31\nCustomer wants a multi-year discount."
    )
    assert opportunity_document(opportunity).source_type == DocumentSourceType.OPPORTUNITY
    assert account_document(account).text == "Account: Acme\nIndustry: Software\nEmployees: 40"
    assert account_document(account).account_id == ACCOUNT


# --- embedding provider -----------------------------------------------------------------------


class FakeSentenceModel:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.max_seq_length = 256
        self.encode_calls: list[dict[str, Any]] = []

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.encode_calls.append({"texts": texts, **kwargs})
        return [[1 / math.sqrt(self.dimension)] * self.dimension for _ in texts]


def test_provider_loads_the_model_once_even_concurrently() -> None:
    loads: list[str] = []
    model = FakeSentenceModel()

    def loader(name: str) -> FakeSentenceModel:
        loads.append(name)
        time.sleep(0.05)
        return model

    provider = SentenceTransformerProvider("mini", 384, batch_size=16, loader=loader)
    threads = [threading.Thread(target=provider.embed_query, args=("q",)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    provider.embed_documents(["a", "b", "c"])

    assert loads == ["mini"]
    assert provider.max_tokens == 254  # 256 minus [CLS] and [SEP]
    batch = model.encode_calls[-1]
    assert batch["texts"] == ["a", "b", "c"]  # one call for the whole batch
    assert batch["batch_size"] == 16 and batch["normalize_embeddings"] is True


def test_provider_rejects_a_model_of_the_wrong_dimension_and_retries_later() -> None:
    models = iter([FakeSentenceModel(dimension=768), FakeSentenceModel()])
    provider = SentenceTransformerProvider("mini", 384, loader=lambda _: next(models))

    with pytest.raises(EmbeddingError, match="768"):
        provider.embed_query("q")
    assert len(provider.embed_query("q")) == 384  # the failure was not cached


@pytest.mark.parametrize(
    ("vectors", "match"),
    [
        ([[0.0] * 383], "384"),
        ([[float("nan")] * 384], "non-finite"),
        ([[0.0] * 384, [0.0] * 384], "expected 1 embeddings"),
    ],
)
def test_vectors_are_validated(vectors: list[list[float]], match: str) -> None:
    with pytest.raises(EmbeddingError, match=match):
        check_vectors(vectors, 1, 384)


# --- retrieval SQL ----------------------------------------------------------------------------


def test_retrieval_filters_by_account_inside_the_ranking_query() -> None:
    stmt = account_search_statement(ACCOUNT, [0.1] * 384, embedding_model="m", limit=5)
    sql = " ".join(str(stmt.compile(dialect=postgresql.dialect())).split())
    where, order = sql.split(" WHERE ")[1].split(" ORDER BY ")

    # One statement: the account predicate restricts candidates before ranking and LIMIT.
    assert "document_chunks.account_id = " in where
    assert "document_chunks.embedding_model = " in where
    assert "<=>" in order
    assert sql.count("SELECT") == 1 and "LIMIT" in sql


def test_retrieval_without_an_account_is_impossible() -> None:
    with pytest.raises(ValueError, match="account_id is required"):
        account_search_statement(None, [0.1] * 384, embedding_model="m", limit=5)  # type: ignore[arg-type]


# --- service ----------------------------------------------------------------------------------


def _service(account_exists: bool = True) -> tuple[SemanticSearchService, Any, MagicMock]:
    accounts = MagicMock()
    accounts.get.return_value = SimpleNamespace(id=ACCOUNT) if account_exists else None
    search = MagicMock()
    search.search.return_value = []
    embeddings = FakeEmbeddingProvider()
    return SemanticSearchService(accounts, search, embeddings), embeddings, search


def test_service_embeds_the_query_and_searches_only_that_account() -> None:
    service, embeddings, search = _service()

    service.search(ACCOUNT, "pricing concerns", limit=3)

    assert embeddings.queries == ["pricing concerns"]
    args, kwargs = search.search.call_args
    assert args[0] == ACCOUNT
    assert kwargs == {"embedding_model": "fake-bag-of-words", "limit": 3}


def test_unknown_account_is_404_without_embedding() -> None:
    service, embeddings, search = _service(account_exists=False)

    with pytest.raises(NotFoundError):
        service.search(ACCOUNT, "pricing", limit=5)
    assert embeddings.queries == [] and not search.search.called


def test_embedding_failure_is_a_generic_503() -> None:
    service, embeddings, _ = _service()
    embeddings.embed_query = MagicMock(side_effect=EmbeddingError("cannot load mini: offline"))

    with pytest.raises(SearchUnavailableError) as error:
        service.search(ACCOUNT, "pricing", limit=5)
    assert error.value.message == "Semantic search is currently unavailable."
    assert "offline" in error.value.reason


# --- routes -----------------------------------------------------------------------------------

URL = "/api/v1/assistant/search"


@pytest.fixture
def search_client(settings: Settings) -> Iterator[tuple[TestClient, MagicMock]]:
    service = MagicMock(spec=SemanticSearchService)
    app = create_app(settings)
    app.dependency_overrides[get_semantic_search_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: make_user()
    with TestClient(app) as client:
        yield client, service


def test_search_returns_scoped_results(search_client: tuple[TestClient, MagicMock]) -> None:
    client, service = search_client
    source_id = uuid.uuid4()
    service.search.return_value = [
        RetrievedChunk(ACCOUNT, DocumentSourceType.NOTE, source_id, "Pricing concern.", T, 0.87)
    ]

    response = client.post(URL, json={"account_id": str(ACCOUNT), "query": "  pricing  "})

    assert response.status_code == 200
    assert response.json() == {
        "account_id": str(ACCOUNT),
        "results": [
            {
                "source_type": "NOTE",
                "source_id": str(source_id),
                "content": "Pricing concern.",
                "similarity": 0.87,
                "occurred_at": "2026-10-01T12:00:00Z",
            }
        ],
    }
    service.search.assert_called_once_with(ACCOUNT, "pricing", limit=5)


@pytest.mark.parametrize(
    "body",
    [
        {"query": "pricing"},
        {"account_id": str(ACCOUNT), "query": "   "},
        {"account_id": str(ACCOUNT), "query": "x" * 1001},
        {"account_id": str(ACCOUNT), "query": "pricing", "limit": 0},
        {"account_id": str(ACCOUNT), "query": "pricing", "limit": 21},
        {"account_id": str(ACCOUNT), "query": "pricing", "all_accounts": True},
    ],
    ids=["no-account", "blank-query", "long-query", "limit-0", "limit-21", "unknown-field"],
)
def test_invalid_requests_are_422(
    search_client: tuple[TestClient, MagicMock], body: dict[str, Any]
) -> None:
    client, service = search_client

    assert client.post(URL, json=body).status_code == 422
    assert not service.search.called


def test_unknown_account_and_unavailable_search(
    search_client: tuple[TestClient, MagicMock],
) -> None:
    client, service = search_client
    body = {"account_id": str(ACCOUNT), "query": "pricing"}

    service.search.side_effect = NotFoundError(f"Account {ACCOUNT} not found")
    assert client.post(URL, json=body).status_code == 404
    service.search.side_effect = SearchUnavailableError("model download failed: secret path")
    response = client.post(URL, json=body)
    assert response.status_code == 503
    assert response.json() == {"detail": "Semantic search is currently unavailable."}


def test_search_requires_authentication(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post(URL, json={"account_id": str(ACCOUNT), "query": "pricing"})

    assert response.status_code == 401
