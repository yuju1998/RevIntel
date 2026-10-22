"""POST /activities, POST /notes and GET /leads/{id}/activities against PostgreSQL."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import Account, User
from tests.integration.factories import (
    create_account,
    create_activity,
    create_contact,
    create_lead,
    create_opportunity,
    create_user,
)

pytestmark = pytest.mark.integration

ACTIVITIES = "/api/v1/activities"
NOTES = "/api/v1/notes"
HOUR_AGO = (datetime.now(UTC) - timedelta(hours=1)).isoformat()


def _post_activity(client: TestClient, **fields: object):  # noqa: ANN202
    payload = {"activity_type": "MEETING", "direction": "OUTBOUND", "occurred_at": HOUR_AGO}
    return client.post(ACTIVITIES, json={**payload, **{k: _json(v) for k, v in fields.items()}})


def _post_note(client: TestClient, **fields: object):  # noqa: ANN202
    payload = {"body": "Spoke with the CFO."}
    return client.post(NOTES, json={**payload, **{k: _json(v) for k, v in fields.items()}})


def _json(value: object) -> object:
    return str(value) if isinstance(value, uuid.UUID) else value


@pytest.fixture
def account(db_session: Session) -> Account:
    return create_account(db_session)


# --- activities -------------------------------------------------------------------------


def test_create_activity_with_all_consistent_parents(
    api_client: TestClient, db_session: Session, account: Account, auth_user: User
) -> None:
    lead = create_lead(db_session)
    opportunity = create_opportunity(db_session, account)
    contact = create_contact(db_session, account)
    body = "  Agenda:\n  - pricing\n"

    response = _post_activity(
        api_client,
        account_id=account.id,
        lead_id=lead.id,
        opportunity_id=opportunity.id,
        contact_id=contact.id,
        subject="  QBR  ",
        body=body,
    )

    assert response.status_code == 201, response.json()
    created = response.json()
    assert created["account_id"] == str(account.id)
    assert created["lead_id"] == str(lead.id)
    # Authorship comes from the access token, not the request.
    assert created["created_by_id"] == str(auth_user.id)
    assert created["subject"] == "QBR"
    assert created["body"] == body
    assert created["occurred_at"].endswith("Z")
    assert created["created_at"].endswith("Z")


@pytest.mark.parametrize("via", ["opportunity", "contact", "both"])
def test_account_id_is_derived_when_omitted(
    api_client: TestClient, db_session: Session, account: Account, via: str
) -> None:
    refs = {}
    if via in {"opportunity", "both"}:
        refs["opportunity_id"] = create_opportunity(db_session, account).id
    if via in {"contact", "both"}:
        refs["contact_id"] = create_contact(db_session, account).id

    response = _post_activity(api_client, **refs)

    assert response.status_code == 201, response.json()
    assert response.json()["account_id"] == str(account.id)


def test_lead_only_activity_keeps_account_null(
    api_client: TestClient, db_session: Session
) -> None:
    response = _post_activity(api_client, lead_id=create_lead(db_session).id)

    assert response.status_code == 201
    assert response.json()["account_id"] is None


def test_lead_may_coexist_with_account_scoped_references(
    api_client: TestClient, db_session: Session, account: Account
) -> None:
    response = _post_activity(
        api_client,
        lead_id=create_lead(db_session).id,
        contact_id=create_contact(db_session, account).id,
    )

    assert response.status_code == 201
    assert response.json()["account_id"] == str(account.id)


@pytest.mark.parametrize(
    "combination", ["account+opportunity", "account+contact", "opportunity+contact", "all"]
)
def test_cross_account_references_are_rejected(
    api_client: TestClient, db_session: Session, combination: str
) -> None:
    mine, other = create_account(db_session, name="Mine"), create_account(db_session, name="Other")
    refs = {
        "account+opportunity": {
            "account_id": mine.id,
            "opportunity_id": create_opportunity(db_session, other).id,
        },
        "account+contact": {
            "account_id": mine.id,
            "contact_id": create_contact(db_session, other).id,
        },
        "opportunity+contact": {
            "opportunity_id": create_opportunity(db_session, mine).id,
            "contact_id": create_contact(db_session, other).id,
        },
        "all": {
            "account_id": mine.id,
            "opportunity_id": create_opportunity(db_session, mine).id,
            "contact_id": create_contact(db_session, other).id,
        },
    }[combination]

    response = _post_activity(api_client, **refs)

    assert response.status_code == 422
    assert "refer to different accounts" in response.json()["detail"]


@pytest.mark.parametrize("field", ["account_id", "lead_id", "opportunity_id", "contact_id"])
def test_nonexistent_parent_returns_422(api_client: TestClient, field: str) -> None:
    response = _post_activity(api_client, **{field: uuid.uuid4()})

    assert response.status_code == 422
    assert field in response.json()["detail"]


def test_created_by_id_cannot_be_supplied(
    api_client: TestClient, db_session: Session, account: Account
) -> None:
    other_user = create_user(db_session)

    response = _post_activity(api_client, account_id=account.id, created_by_id=other_user.id)

    # Impersonation attempt: the field is not part of the request schema.
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"occurred_at": "2026-01-01T12:00:00"},
        {"occurred_at": (datetime.now(UTC) + timedelta(minutes=10)).isoformat()},
        {"activity_type": "LUNCH"},
        {"direction": "SIDEWAYS"},
        {"subject": "x" * 256},
        {"body": "   "},
        {"account_id": "not-a-uuid"},
        {"unexpected": True},
    ],
    ids=[
        "no-parent",
        "naive",
        "future",
        "bad-type",
        "bad-direction",
        "long-subject",
        "blank-body",
        "bad-uuid",
        "unknown-field",
    ],
)
def test_invalid_activity_returns_422(
    api_client: TestClient, db_session: Session, fields: dict
) -> None:
    parent = {} if fields == {} else {"lead_id": create_lead(db_session).id}

    assert _post_activity(api_client, **parent, **fields).status_code == 422


def test_occurred_at_within_clock_skew_is_accepted(
    api_client: TestClient, db_session: Session
) -> None:
    soon = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()

    response = _post_activity(api_client, lead_id=create_lead(db_session).id, occurred_at=soon)

    assert response.status_code == 201


# --- notes ------------------------------------------------------------------------------


def test_create_note_preserves_body_and_derives_account(
    api_client: TestClient, db_session: Session, account: Account, auth_user: User
) -> None:
    opportunity = create_opportunity(db_session, account)
    body = "\n  - renewal risk\n  - champion left\n"

    response = _post_note(api_client, opportunity_id=opportunity.id, body=body)

    assert response.status_code == 201, response.json()
    note = response.json()
    assert note["body"] == body
    assert note["account_id"] == str(account.id)
    assert note["author_id"] == str(auth_user.id)
    assert note["created_at"] == note["updated_at"]


def test_lead_only_note(api_client: TestClient, db_session: Session) -> None:
    response = _post_note(api_client, lead_id=create_lead(db_session).id)

    assert response.status_code == 201
    assert response.json()["account_id"] is None


def test_note_cross_account_reference_is_rejected(
    api_client: TestClient, db_session: Session, account: Account
) -> None:
    other_opportunity = create_opportunity(db_session, create_account(db_session, name="Other"))

    response = _post_note(api_client, account_id=account.id, opportunity_id=other_opportunity.id)

    assert response.status_code == 422
    assert "refer to different accounts" in response.json()["detail"]


def test_author_id_cannot_be_supplied(
    api_client: TestClient, db_session: Session, account: Account
) -> None:
    other_user = create_user(db_session)

    response = _post_note(api_client, account_id=account.id, author_id=other_user.id)

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    "fields",
    [{}, {"body": "  \n "}, {"contact_id": str(uuid.uuid4())}, {"lead_id": str(uuid.uuid4())}],
    ids=["no-parent", "blank-body", "contact-not-a-note-parent", "missing-lead"],
)
def test_invalid_note_returns_422(api_client: TestClient, fields: dict) -> None:
    assert _post_note(api_client, **fields).status_code == 422


# --- lead activities --------------------------------------------------------------------


def test_lead_activities_are_newest_first_and_paginated(
    api_client: TestClient, db_session: Session
) -> None:
    lead, other_lead = create_lead(db_session), create_lead(db_session)
    base = datetime(2026, 9, 1, tzinfo=UTC)
    created = [
        create_activity(db_session, occurred_at=base + timedelta(days=i), lead_id=lead.id)
        for i in range(3)
    ]
    # Same timestamp as the newest one: ordered by id DESC.
    tie = create_activity(db_session, occurred_at=base + timedelta(days=2), lead_id=lead.id)
    create_activity(db_session, occurred_at=base, lead_id=other_lead.id)
    url = f"/api/v1/leads/{lead.id}/activities"

    first = api_client.get(url, params={"limit": 2}).json()
    second = api_client.get(url, params={"limit": 2, "offset": 2}).json()

    newest = sorted([created[2], tie], key=lambda a: a.id, reverse=True)
    expected = [str(a.id) for a in [*newest, created[1], created[0]]]
    assert first["total"] == second["total"] == 4
    assert [item["id"] for item in first["items"] + second["items"]] == expected


def test_lead_with_no_activities_returns_empty_page(
    api_client: TestClient, db_session: Session
) -> None:
    response = api_client.get(f"/api/v1/leads/{create_lead(db_session).id}/activities")

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 50, "offset": 0}


def test_lead_activities_for_unknown_lead_returns_404(api_client: TestClient) -> None:
    assert api_client.get(f"/api/v1/leads/{uuid.uuid4()}/activities").status_code == 404


def test_lead_activities_with_malformed_id_returns_422(api_client: TestClient) -> None:
    assert api_client.get("/api/v1/leads/nope/activities").status_code == 422
