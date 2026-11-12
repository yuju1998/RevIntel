"""CRM indexing and account-scoped semantic search against PostgreSQL + pgvector.

Uses a deterministic fake embedding provider (identical text -> identical vector), so
isolation is tested in the hardest case: another account's chunk is the exact nearest
neighbour of the query, and must still never be returned.
"""

import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.models import Account, Activity, DocumentChunk, Note
from app.models.enums import DocumentEntityType, DocumentSourceType
from app.repositories.crm_sources import CrmSourceRepository
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.semantic_search import SemanticSearchRepository
from app.services.crm_indexing_service import CrmIndexingService, IndexingStats
from ml.synthetic.config import GenerationConfig
from ml.synthetic.generator import generate
from ml.synthetic.writer import write_dataset
from tests.embedding_fakes import FakeEmbeddingProvider, bag_of_words
from tests.integration.factories import (
    create_account,
    create_activity,
    create_contact,
    create_lead,
    create_note,
    create_opportunity,
)

pytestmark = pytest.mark.integration

T = datetime(2026, 10, 1, 12, tzinfo=UTC)
PRICING = "Customer raised pricing concerns about the enterprise tier discount."


def _index(
    session: Session, provider: FakeEmbeddingProvider, account_id: uuid.UUID | None = None
) -> IndexingStats:
    return CrmIndexingService(
        session,
        CrmSourceRepository(session),
        DocumentChunkRepository(session),
        provider,
        page_size=2,  # several pages even with a handful of sources
    ).index(account_id=account_id)


def _chunks(session: Session) -> list[DocumentChunk]:
    session.expire_all()
    chunks = list(session.scalars(select(DocumentChunk).order_by(DocumentChunk.content)))
    # End the read transaction: an open snapshot would block TRUNCATE on other connections.
    session.commit()
    return chunks


def _search(session: Session, account_id: uuid.UUID, query: str, limit: int = 20) -> list:
    results = SemanticSearchRepository(session).search(
        account_id, bag_of_words(query), embedding_model="fake-bag-of-words", limit=limit
    )
    session.commit()
    return results


# --- indexing ---------------------------------------------------------------------------------


def test_every_source_type_is_indexed_under_its_resolved_account(db_session: Session) -> None:
    acme = create_account(db_session, name="Acme", industry="Software")
    contact = create_contact(db_session, acme)
    deal = create_opportunity(db_session, acme, name="Renewal", description="Multi-year deal.")
    lead = create_lead(db_session)
    direct = create_activity(db_session, occurred_at=T, account_id=acme.id, subject="Kickoff")
    via_deal = create_activity(db_session, occurred_at=T, opportunity_id=deal.id, subject="Demo")
    via_contact = create_activity(db_session, occurred_at=T, contact_id=contact.id, subject="Call")
    lead_only = create_activity(db_session, occurred_at=T, lead_id=lead.id, subject="Lead call")
    note = create_note(db_session, created_at=T, account_id=acme.id, body=PRICING)
    deal_note = create_note(db_session, created_at=T, opportunity_id=deal.id, body="Deal note")
    create_note(db_session, created_at=T, lead_id=lead.id, body="Lead note")

    stats = _index(db_session, FakeEmbeddingProvider())

    chunks = {(c.source_type, c.source_id): c for c in _chunks(db_session)}
    assert set(chunks) == {
        (DocumentSourceType.ACCOUNT, acme.id),
        (DocumentSourceType.OPPORTUNITY, deal.id),
        (DocumentSourceType.ACTIVITY, direct.id),
        (DocumentSourceType.ACTIVITY, via_deal.id),
        (DocumentSourceType.ACTIVITY, via_contact.id),
        (DocumentSourceType.NOTE, note.id),
        (DocumentSourceType.NOTE, deal_note.id),
    }
    assert (DocumentSourceType.ACTIVITY, lead_only.id) not in chunks  # no account to resolve
    assert all(chunk.account_id == acme.id for chunk in chunks.values())
    entities = {key: (c.entity_type, c.entity_id) for key, c in chunks.items()}
    assert entities[(DocumentSourceType.ACTIVITY, via_deal.id)] == (
        DocumentEntityType.OPPORTUNITY,
        deal.id,
    )
    assert entities[(DocumentSourceType.ACTIVITY, via_contact.id)] == (
        DocumentEntityType.CONTACT,
        contact.id,
    )
    stored = chunks[(DocumentSourceType.NOTE, note.id)]
    assert stored.content == PRICING
    assert stored.occurred_at == T
    assert stored.embedding_model == "fake-bag-of-words"
    assert list(stored.embedding) == pytest.approx(bag_of_words(PRICING))
    assert len(stored.source_sha256) == 64
    assert stats.by_type["ACTIVITY"].skipped == 1 and stats.by_type["NOTE"].skipped == 1


def test_rerun_is_idempotent_and_tracks_edits_moves_and_deletes(db_session: Session) -> None:
    acme, globex = (
        create_account(db_session, name="Acme"),
        create_account(db_session, name="Globex"),
    )
    edited = create_activity(db_session, occurred_at=T, account_id=acme.id, subject="Old")
    moved = create_activity(db_session, occurred_at=T, account_id=acme.id, subject="Moves")
    deleted = create_activity(db_session, occurred_at=T, account_id=acme.id, subject="Gone")
    provider = FakeEmbeddingProvider()
    _index(db_session, provider)
    first = {(c.source_type, c.source_id): c.id for c in _chunks(db_session)}

    calls = len(provider.document_calls)
    rerun = _index(db_session, provider)
    assert len(provider.document_calls) == calls  # nothing re-embedded
    assert {(c.source_type, c.source_id): c.id for c in _chunks(db_session)} == first
    assert rerun.by_type["ACTIVITY"].unchanged == 3 and rerun.by_type["ACTIVITY"].indexed == 0

    db_session.execute(update(Activity).where(Activity.id == edited.id).values(subject="New"))
    db_session.execute(update(Activity).where(Activity.id == moved.id).values(account_id=globex.id))
    db_session.execute(delete(Activity).where(Activity.id == deleted.id))
    db_session.commit()
    calls = len(provider.document_calls)
    stats = _index(db_session, provider)

    chunks = {c.source_id: c for c in _chunks(db_session)}
    assert chunks[edited.id].content == "Call (outbound): New"
    assert chunks[moved.id].account_id == globex.id
    assert deleted.id not in chunks
    assert stats.by_type["ACTIVITY"].indexed == 2 and stats.by_type["ACTIVITY"].removed == 1
    embedded = [text for call in provider.document_calls[calls:] for text in call]
    assert sorted(embedded) == ["Call (outbound): Moves", "Call (outbound): New"]  # only these
    keys = [(c.source_type, c.source_id, c.chunk_index) for c in _chunks(db_session)]
    assert len(keys) == len(set(keys))  # never duplicated


def test_scoped_index_only_touches_that_account(db_session: Session) -> None:
    acme, globex = (
        create_account(db_session, name="Acme"),
        create_account(db_session, name="Globex"),
    )
    create_note(db_session, created_at=T, account_id=globex.id, body="Globex note")

    _index(db_session, FakeEmbeddingProvider(), account_id=acme.id)

    assert {c.account_id for c in _chunks(db_session)} == {acme.id}


def test_deleting_an_account_removes_its_chunks(db_session: Session) -> None:
    acme = create_account(db_session, name="Acme")
    _index(db_session, FakeEmbeddingProvider())

    db_session.execute(delete(Account).where(Account.id == acme.id))
    db_session.commit()

    assert _chunks(db_session) == []


def test_long_text_is_chunked_within_the_token_limit(db_session: Session) -> None:
    acme = create_account(db_session, name="Acme")
    body = " ".join(f"word{i}" for i in range(40))
    note = create_note(db_session, created_at=T, account_id=acme.id, body=body)

    _index(db_session, FakeEmbeddingProvider(max_tokens=16))

    pieces = [c for c in _chunks(db_session) if c.source_id == note.id]
    assert len(pieces) > 1
    assert sorted(c.chunk_index for c in pieces) == list(range(len(pieces)))
    assert all(len(c.content.split()) <= 16 for c in pieces)
    assert {c.chunk_metadata["chunk_count"] for c in pieces} == {len(pieces)}


# --- account isolation ------------------------------------------------------------------------


@pytest.fixture
def two_accounts(db_session: Session) -> tuple[Account, Account, Note, Note]:
    acme, globex = (
        create_account(db_session, name="Acme"),
        create_account(db_session, name="Globex"),
    )
    shared = "Asked for a security review of the integration."
    create_note(db_session, created_at=T, account_id=acme.id, body=shared)
    create_note(db_session, created_at=T, account_id=globex.id, body=shared)
    acme_pricing = create_note(
        db_session, created_at=T, account_id=acme.id, body="Budget approved for next quarter."
    )
    globex_pricing = create_note(db_session, created_at=T, account_id=globex.id, body=PRICING)
    _index(db_session, FakeEmbeddingProvider())
    return acme, globex, acme_pricing, globex_pricing


def test_another_accounts_exact_match_is_never_returned(
    db_session: Session, two_accounts: tuple[Account, Account, Note, Note]
) -> None:
    acme, globex, _, globex_pricing = two_accounts
    globex_ids = {c.source_id for c in _chunks(db_session) if c.account_id == globex.id}

    # The query IS Globex's note text: its chunk is the global nearest neighbour (similarity 1).
    results = _search(db_session, acme.id, PRICING, limit=20)  # more than Acme has

    assert results and all(r.account_id == acme.id for r in results)
    assert not {r.source_id for r in results} & globex_ids
    assert globex_pricing.id not in {r.source_id for r in results}
    assert len(results) == sum(c.account_id == acme.id for c in _chunks(db_session))

    own = _search(db_session, globex.id, PRICING, limit=1)
    assert own[0].source_id == globex_pricing.id and own[0].similarity == pytest.approx(1.0)


def test_identical_text_in_two_accounts_returns_only_the_requested_copy(
    db_session: Session, two_accounts: tuple[Account, Account, Note, Note]
) -> None:
    acme, globex, _, _ = two_accounts

    for account in (acme, globex):
        results = _search(db_session, account.id, "security review integration")
        assert {r.account_id for r in results} == {account.id}
        assert results[0].content == "Asked for a security review of the integration."


def test_an_account_without_chunks_gets_no_results(
    db_session: Session, two_accounts: tuple[Account, Account, Note, Note]
) -> None:
    empty = create_account(db_session, name="Initech")  # created after indexing

    assert _search(db_session, empty.id, PRICING) == []


def test_search_api_is_scoped_to_the_requested_account(
    api_client: TestClient, db_session: Session, two_accounts: tuple[Account, Account, Note, Note]
) -> None:
    acme, globex, acme_note, globex_pricing = two_accounts
    api_client.app.state.embedding_provider = FakeEmbeddingProvider()

    response = api_client.post(
        "/api/v1/assistant/search",
        json={"account_id": str(acme.id), "query": PRICING, "limit": 20},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["account_id"] == str(acme.id)
    returned = {r["source_id"] for r in body["results"]}
    assert str(globex_pricing.id) not in returned and str(acme_note.id) in returned
    assert all(
        set(r) == {"source_type", "source_id", "content", "similarity", "occurred_at"}
        for r in body["results"]
    )
    similarities = [r["similarity"] for r in body["results"]]
    assert similarities == sorted(similarities, reverse=True)

    missing = api_client.post(
        "/api/v1/assistant/search", json={"account_id": str(uuid.uuid4()), "query": PRICING}
    )
    assert missing.status_code == 404


# --- synthetic dataset ------------------------------------------------------------------------


def test_synthetic_reset_also_removes_chunks(
    migrated_database: str, db_session: Session, clean_tables: None
) -> None:
    config = GenerationConfig(
        seed=5, accounts=4, contacts=8, leads=10, activities=60, reps=2, admins=1
    )
    write_dataset(generate(config), migrated_database)
    stats = _index(db_session, FakeEmbeddingProvider())
    assert stats.by_type["ACTIVITY"].skipped > 0  # lead-linked activities have no account
    assert _chunks(db_session)

    write_dataset(generate(config), migrated_database, reset=True)

    assert _chunks(db_session) == []
