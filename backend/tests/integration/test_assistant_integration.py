"""CRM Copilot against PostgreSQL + pgvector: prompts and sources stay inside one account.

Account B holds the exact-match record for the question and a real, indexed source_id.
Even so, account A's prompt never contains it and a model citing it is never believed.
"""

import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.ai.llm import CopilotAnswer, CopilotPrompt, MockLLMProvider
from app.models import Account, Note
from app.repositories.account import AccountRepository
from app.repositories.crm_sources import CrmSourceRepository
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.semantic_search import SemanticSearchRepository
from app.services.assistant_query_service import AssistantQueryService
from app.services.crm_indexing_service import CrmIndexingService
from app.services.semantic_search_service import SemanticSearchService
from tests.embedding_fakes import FakeEmbeddingProvider
from tests.integration.factories import create_account, create_note

pytestmark = pytest.mark.integration

T = datetime(2026, 10, 1, 12, tzinfo=UTC)
QUESTION = "The CFO pushed back on pricing and asked for a discount."


class CapturingProvider:
    """Records prompts; optionally answers with a scripted (possibly hallucinated) answer."""

    name = "capturing"

    def __init__(self, answer: CopilotAnswer | None = None) -> None:
        self.answer = answer
        self.prompts: list[CopilotPrompt] = []

    def generate(self, prompt: CopilotPrompt) -> CopilotAnswer:
        self.prompts.append(prompt)
        return self.answer or MockLLMProvider().generate(prompt)


@pytest.fixture
def accounts(db_session: Session) -> tuple[Account, Account, Note, Note]:
    acme = create_account(db_session, name="Acme")
    globex = create_account(db_session, name="Globex")
    acme_note = create_note(
        db_session, created_at=T, account_id=acme.id, body="Renewal call went well."
    )
    globex_note = create_note(db_session, created_at=T, account_id=globex.id, body=QUESTION)
    CrmIndexingService(
        db_session,
        CrmSourceRepository(db_session),
        DocumentChunkRepository(db_session),
        FakeEmbeddingProvider(),
    ).index()
    return acme, globex, acme_note, globex_note


def _service(session: Session, provider: CapturingProvider) -> AssistantQueryService:
    search = SemanticSearchService(
        AccountRepository(session), SemanticSearchRepository(session), FakeEmbeddingProvider()
    )
    return AssistantQueryService(search, provider)


def test_prompt_contains_only_the_requested_accounts_records(
    db_session: Session, accounts: tuple[Account, Account, Note, Note]
) -> None:
    acme, globex, acme_note, globex_note = accounts
    provider = CapturingProvider()

    _service(db_session, provider).query(acme.id, QUESTION)
    db_session.commit()

    prompt = provider.prompts[0]
    assert {record.source_id for record in prompt.records} == {acme.id, acme_note.id}
    assert str(globex_note.id) not in prompt.input and str(globex.id) not in prompt.input
    assert QUESTION in prompt.input  # only as the question, never as a record
    assert prompt.input.count(QUESTION) == 1


def test_a_citation_of_another_accounts_real_record_is_never_returned(
    db_session: Session, accounts: tuple[Account, Account, Note, Note]
) -> None:
    acme, _, acme_note, globex_note = accounts
    hallucinated = CopilotAnswer(
        answer=f"Pricing pushback [SOURCE:{globex_note.id}]; renewal [SOURCE:{acme_note.id}].",
        cited_source_ids=[str(globex_note.id), str(acme_note.id), str(uuid.uuid4())],
        insufficient_evidence=False,
    )

    result = _service(db_session, CapturingProvider(hallucinated)).query(acme.id, QUESTION)
    db_session.commit()

    assert [source.source_id for source in result.sources] == [acme_note.id]
    assert str(globex_note.id) not in result.answer
    assert len(result.rejected_citations) == 3


def test_query_api_returns_only_the_requested_accounts_sources(
    api_client: TestClient, accounts: tuple[Account, Account, Note, Note]
) -> None:
    acme, globex, acme_note, globex_note = accounts
    api_client.app.state.embedding_provider = FakeEmbeddingProvider()
    api_client.app.state.llm_provider = CapturingProvider(
        CopilotAnswer(
            answer=f"Facts: [SOURCE:{globex_note.id}] [SOURCE:{acme_note.id}]",
            cited_source_ids=[str(globex_note.id), str(acme_note.id)],
            insufficient_evidence=False,
        )
    )

    response = api_client.post(
        "/api/v1/assistant/query", json={"account_id": str(acme.id), "question": QUESTION}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == [{"source_id": str(acme_note.id), "source_type": "NOTE"}]
    assert str(globex_note.id) not in body["answer"]
    for unknown in (str(uuid.uuid4()), str(globex.id)):
        assert unknown not in {source["source_id"] for source in body["sources"]}

    missing = api_client.post(
        "/api/v1/assistant/query", json={"account_id": str(uuid.uuid4()), "question": QUESTION}
    )
    assert missing.status_code == 404


def test_account_without_indexed_records_gets_insufficient_evidence(
    db_session: Session, accounts: tuple[Account, Account, Note, Note]
) -> None:
    empty = create_account(db_session, name="Initech")  # created after indexing
    provider = CapturingProvider()

    result = _service(db_session, provider).query(empty.id, QUESTION)
    db_session.commit()

    assert result.insufficient_evidence is True and result.sources == []
    assert provider.prompts == []  # the LLM is never asked without evidence
