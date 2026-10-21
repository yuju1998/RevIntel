"""GET /accounts/{id}/timeline: one paginated stream of activities and notes."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import Account, Activity, Note
from tests.integration.factories import (
    create_account,
    create_activity,
    create_contact,
    create_lead,
    create_note,
    create_opportunity,
)

pytestmark = pytest.mark.integration

BASE = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def _url(account_id: uuid.UUID) -> str:
    return f"/api/v1/accounts/{account_id}/timeline"


def _all_pages(client: TestClient, account_id: uuid.UUID, limit: int) -> list[dict]:
    items: list[dict] = []
    offset = 0
    while True:
        page = client.get(_url(account_id), params={"limit": limit, "offset": offset}).json()
        items.extend(page["items"])
        offset += limit
        if offset >= page["total"]:
            return items


@pytest.fixture
def account(db_session: Session) -> Account:
    return create_account(db_session)


@pytest.fixture
def interleaved(db_session: Session, account: Account) -> list[Activity | Note]:
    """Activities and notes alternating in time, inserted out of order. Newest first."""
    records: list[Activity | Note] = []
    for hours in (5, 0, 3, 6, 1, 4, 2):
        when = BASE + timedelta(hours=hours)
        if hours % 2:
            records.append(create_note(db_session, account_id=account.id, created_at=when))
        else:
            records.append(create_activity(db_session, occurred_at=when, account_id=account.id))
    return sorted(
        records,
        key=lambda r: r.occurred_at if isinstance(r, Activity) else r.created_at,
        reverse=True,
    )


def _ids(items: list[dict]) -> list[str]:
    return [item["id"] for item in items]


def test_timeline_contains_both_record_types_newest_first(
    api_client: TestClient, account: Account, interleaved: list[Activity | Note]
) -> None:
    page = api_client.get(_url(account.id)).json()

    assert page["total"] == 7
    assert _ids(page["items"]) == [str(r.id) for r in interleaved]
    assert {item["record_type"] for item in page["items"]} == {"activity", "note"}
    timestamps = [item["event_timestamp"] for item in page["items"]]
    assert timestamps == sorted(timestamps, reverse=True)
    for item in page["items"]:
        source = "occurred_at" if item["record_type"] == "activity" else "created_at"
        assert item["event_timestamp"] == item[source]


@pytest.mark.parametrize("limit", [1, 2, 3, 7])
def test_pagination_is_global_across_both_record_types(
    api_client: TestClient, account: Account, interleaved: list[Activity | Note], limit: int
) -> None:
    items = _all_pages(api_client, account.id, limit)

    # Concatenated pages equal the full ordering: nothing duplicated, nothing skipped.
    assert _ids(items) == [str(r.id) for r in interleaved]


def test_a_page_can_mix_record_types(
    api_client: TestClient, account: Account, interleaved: list[Activity | Note]
) -> None:
    page = api_client.get(_url(account.id), params={"limit": 3, "offset": 2}).json()

    assert page["total"] == 7 and page["limit"] == 3 and page["offset"] == 2
    assert _ids(page["items"]) == [str(r.id) for r in interleaved[2:5]]
    assert [item["record_type"] for item in page["items"]] == ["activity", "note", "activity"]


def test_offset_past_the_end_returns_empty_items_with_total(
    api_client: TestClient, account: Account, interleaved: list[Activity | Note]
) -> None:
    page = api_client.get(_url(account.id), params={"offset": 50}).json()

    assert page == {"items": [], "total": 7, "limit": 50, "offset": 50}


def test_equal_timestamps_are_ordered_deterministically(
    api_client: TestClient, db_session: Session, account: Account
) -> None:
    notes = [create_note(db_session, account_id=account.id, created_at=BASE) for _ in range(3)]
    activities = [
        create_activity(db_session, occurred_at=BASE, account_id=account.id) for _ in range(3)
    ]
    # Same event_timestamp: record_type ASC ("activity" before "note"), then id DESC.
    expected = [
        *(str(a.id) for a in sorted(activities, key=lambda a: a.id, reverse=True)),
        *(str(n.id) for n in sorted(notes, key=lambda n: n.id, reverse=True)),
    ]

    assert _ids(_all_pages(api_client, account.id, limit=2)) == expected
    assert _ids(_all_pages(api_client, account.id, limit=4)) == expected


def test_lead_only_and_other_account_records_are_excluded(
    api_client: TestClient, db_session: Session, account: Account
) -> None:
    mine = create_activity(db_session, occurred_at=BASE, account_id=account.id)
    lead = create_lead(db_session)
    create_activity(db_session, occurred_at=BASE, lead_id=lead.id)
    create_note(db_session, lead_id=lead.id, created_at=BASE)
    other = create_account(db_session, name="Other")
    create_activity(db_session, occurred_at=BASE, account_id=other.id)
    create_note(db_session, account_id=other.id, created_at=BASE)

    page = api_client.get(_url(account.id)).json()

    assert page["total"] == 1
    assert _ids(page["items"]) == [str(mine.id)]


def test_records_created_via_opportunity_or_contact_appear_on_the_timeline(
    api_client: TestClient, db_session: Session, account: Account
) -> None:
    opportunity = create_opportunity(db_session, account)
    contact = create_contact(db_session, account)
    hour_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    activity = api_client.post(
        "/api/v1/activities",
        json={
            "contact_id": str(contact.id),
            "activity_type": "CALL",
            "direction": "INBOUND",
            "occurred_at": hour_ago,
        },
    ).json()
    note = api_client.post(
        "/api/v1/notes", json={"opportunity_id": str(opportunity.id), "body": "Next steps"}
    ).json()

    page = api_client.get(_url(account.id)).json()

    assert page["total"] == 2
    assert set(_ids(page["items"])) == {activity["id"], note["id"]}


def test_empty_timeline(api_client: TestClient, account: Account) -> None:
    assert api_client.get(_url(account.id)).json() == {
        "items": [],
        "total": 0,
        "limit": 50,
        "offset": 0,
    }


def test_unknown_account_returns_404(api_client: TestClient) -> None:
    assert api_client.get(_url(uuid.uuid4())).status_code == 404


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}])
def test_invalid_pagination_returns_422(
    api_client: TestClient, account: Account, params: dict
) -> None:
    assert api_client.get(_url(account.id), params=params).status_code == 422


def test_malformed_account_id_returns_422(api_client: TestClient) -> None:
    assert api_client.get("/api/v1/accounts/not-a-uuid/timeline").status_code == 422
