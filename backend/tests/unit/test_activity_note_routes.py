"""Activity, note, timeline and lead-activity routes with services overridden."""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.api.deps import get_activity_service, get_note_service, get_timeline_service
from app.core.config import Settings
from app.core.exceptions import BusinessRuleError, NotFoundError, RelatedResourceNotFoundError
from app.main import create_app
from app.models import Activity, Note
from app.models.enums import ActivityDirection, ActivityType
from app.schemas.timeline import timeline_item
from app.services.activity_service import ActivityService
from app.services.note_service import NoteService
from app.services.pagination import PageResult
from app.services.timeline_service import TimelineService
from tests.unit.helpers import make_user

OCCURRED = datetime(2026, 10, 20, 15, 0, tzinfo=UTC)
RECORDED = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)

AUTH_USER = make_user()


def _activity() -> Activity:
    return Activity(
        id=uuid.uuid4(),
        account_id=uuid.uuid4(),
        activity_type=ActivityType.MEETING,
        direction=ActivityDirection.OUTBOUND,
        occurred_at=OCCURRED,
        created_at=RECORDED,
    )


def _note() -> Note:
    return Note(
        id=uuid.uuid4(),
        account_id=uuid.uuid4(),
        body="  formatted\n",
        created_at=RECORDED,
        updated_at=RECORDED,
    )


@pytest.fixture
def services() -> dict[str, MagicMock]:
    return {
        "activity": MagicMock(spec=ActivityService),
        "note": MagicMock(spec=NoteService),
        "timeline": MagicMock(spec=TimelineService),
    }


@pytest.fixture
def client(settings: Settings, services: dict[str, MagicMock]) -> Iterator[TestClient]:
    app = create_app(settings)
    app.dependency_overrides[get_activity_service] = lambda: services["activity"]
    app.dependency_overrides[get_note_service] = lambda: services["note"]
    app.dependency_overrides[get_timeline_service] = lambda: services["timeline"]
    # Authentication itself is covered by test_auth_routes / integration tests.
    app.dependency_overrides[get_current_user] = lambda: AUTH_USER
    with TestClient(app) as test_client:
        yield test_client


def test_timeline_item_uses_occurred_at_for_activities_and_created_at_for_notes() -> None:
    activity_item = timeline_item(_activity())
    note_item = timeline_item(_note())

    assert (activity_item.record_type, activity_item.event_timestamp) == ("activity", OCCURRED)
    assert (note_item.record_type, note_item.event_timestamp) == ("note", RECORDED)
    assert note_item.body == "  formatted\n"


def test_timeline_route_returns_discriminated_items_in_service_order(
    client: TestClient, services: dict[str, MagicMock]
) -> None:
    activity, note = _activity(), _note()
    services["timeline"].get_account_timeline.return_value = PageResult(
        items=[note, activity], total=7, limit=2, offset=4
    )
    account_id = uuid.uuid4()

    response = client.get(
        f"/api/v1/accounts/{account_id}/timeline", params={"limit": 2, "offset": 4}
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["total"], body["limit"], body["offset"]) == (7, 2, 4)
    assert [item["record_type"] for item in body["items"]] == ["note", "activity"]
    assert [item["id"] for item in body["items"]] == [str(note.id), str(activity.id)]
    assert body["items"][1]["event_timestamp"] == "2026-10-20T15:00:00Z"
    services["timeline"].get_account_timeline.assert_called_once_with(
        account_id, limit=2, offset=4
    )


@pytest.mark.parametrize(
    "path", ["/api/v1/accounts/{id}/timeline", "/api/v1/leads/{id}/activities"]
)
@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}])
def test_list_endpoints_validate_pagination(client: TestClient, path: str, params: dict) -> None:
    assert client.get(path.format(id=uuid.uuid4()), params=params).status_code == 422


@pytest.mark.parametrize("path", ["/api/v1/accounts/x/timeline", "/api/v1/leads/x/activities"])
def test_list_endpoints_reject_malformed_uuid(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 422


def test_missing_account_timeline_returns_404(
    client: TestClient, services: dict[str, MagicMock]
) -> None:
    services["timeline"].get_account_timeline.side_effect = NotFoundError("Account x not found")

    assert client.get(f"/api/v1/accounts/{uuid.uuid4()}/timeline").status_code == 404


def test_lead_activities_route(client: TestClient, services: dict[str, MagicMock]) -> None:
    services["activity"].list_lead_activities.return_value = PageResult(
        items=[_activity()], total=1, limit=50, offset=0
    )
    lead_id = uuid.uuid4()

    response = client.get(f"/api/v1/leads/{lead_id}/activities")

    assert response.status_code == 200
    assert response.json()["items"][0]["activity_type"] == "MEETING"
    services["activity"].list_lead_activities.assert_called_once_with(
        lead_id, limit=50, offset=0
    )


def test_create_activity_returns_201(client: TestClient, services: dict[str, MagicMock]) -> None:
    services["activity"].create_activity.return_value = _activity()

    response = client.post(
        "/api/v1/activities",
        json={
            "account_id": str(uuid.uuid4()),
            "activity_type": "MEETING",
            "direction": "OUTBOUND",
            "occurred_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        },
    )

    assert response.status_code == 201
    assert response.json()["occurred_at"] == "2026-10-20T15:00:00Z"
    # Authorship comes from the authenticated user, not the request.
    _, kwargs = services["activity"].create_activity.call_args
    assert kwargs == {"created_by_id": AUTH_USER.id}


@pytest.mark.parametrize(
    ("path", "field"), [("/api/v1/activities", "created_by_id"), ("/api/v1/notes", "author_id")]
)
def test_authorship_fields_cannot_be_supplied(
    client: TestClient, services: dict[str, MagicMock], path: str, field: str
) -> None:
    payload = {
        "account_id": str(uuid.uuid4()),
        "activity_type": "CALL",
        "direction": "INBOUND",
        "occurred_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        "body": "x",
        field: str(uuid.uuid4()),
    }
    if path.endswith("notes"):
        payload = {"account_id": payload["account_id"], "body": "x", field: payload[field]}

    response = client.post(path, json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"
    services["activity"].create_activity.assert_not_called()
    services["note"].create_note.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {
            "activity_type": "MEETING",
            "direction": "OUTBOUND",
            "occurred_at": "2026-01-01T00:00:00Z",
        },
        {
            "account_id": str(uuid.uuid4()),
            "activity_type": "MEETING",
            "direction": "OUTBOUND",
            "occurred_at": "2026-01-01T00:00:00",
        },
    ],
    ids=["no-parent", "naive-occurred-at"],
)
def test_invalid_activity_returns_422_without_calling_service(
    client: TestClient, services: dict[str, MagicMock], payload: dict
) -> None:
    assert client.post("/api/v1/activities", json=payload).status_code == 422
    services["activity"].create_activity.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        RelatedResourceNotFoundError("account_id does not refer to an existing account"),
        BusinessRuleError("account_id, opportunity_id refer to different accounts"),
        BusinessRuleError("author_id refers to an inactive user"),
    ],
)
def test_note_domain_errors_return_422(
    client: TestClient, services: dict[str, MagicMock], error: Exception
) -> None:
    services["note"].create_note.side_effect = error

    response = client.post(
        "/api/v1/notes", json={"account_id": str(uuid.uuid4()), "body": "Hello"}
    )

    assert response.status_code == 422
    assert response.json() == {"detail": str(error)}
