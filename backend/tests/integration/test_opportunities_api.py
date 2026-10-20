import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from tests.integration.factories import create_account, create_user

pytestmark = pytest.mark.integration

URL = "/api/v1/opportunities"


@pytest.fixture
def opportunity(api_client: TestClient, db_session: Session) -> dict:
    account = create_account(db_session)
    response = api_client.post(
        URL,
        json={
            "account_id": str(account.id),
            "name": "Big Deal",
            "stage": "PROSPECTING",
            "amount": "5000.00",
            "expected_close_date": "2026-12-31",
            "description": "First contact",
        },
    )
    assert response.status_code == 201, response.json()
    return response.json()


def test_create_opportunity(opportunity: dict) -> None:
    assert opportunity["stage"] == "PROSPECTING"
    assert opportunity["amount"] == "5000.00"
    assert opportunity["expected_close_date"] == "2026-12-31"
    assert opportunity["owner_id"] is None


def test_create_with_nonexistent_account_returns_422(api_client: TestClient) -> None:
    response = api_client.post(
        URL, json={"account_id": str(uuid.uuid4()), "name": "Deal", "stage": "PROPOSAL"}
    )

    assert response.status_code == 422
    assert "account_id" in response.json()["detail"]


def test_create_with_negative_amount_returns_422(
    api_client: TestClient, db_session: Session
) -> None:
    account = create_account(db_session)
    response = api_client.post(
        URL,
        json={"account_id": str(account.id), "name": "Deal", "stage": "PROPOSAL", "amount": -1},
    )

    assert response.status_code == 422


def test_patch_changes_only_supplied_fields(api_client: TestClient, opportunity: dict) -> None:
    response = api_client.patch(f"{URL}/{opportunity['id']}", json={"stage": "NEGOTIATION"})

    assert response.status_code == 200
    updated = response.json()
    assert updated["stage"] == "NEGOTIATION"
    for field in ("name", "amount", "expected_close_date", "description", "account_id"):
        assert updated[field] == opportunity[field]
    assert updated["created_at"] == opportunity["created_at"]
    assert updated["updated_at"] > opportunity["updated_at"]


def test_patch_null_clears_nullable_fields(api_client: TestClient, opportunity: dict) -> None:
    response = api_client.patch(
        f"{URL}/{opportunity['id']}",
        json={"amount": None, "description": None, "expected_close_date": None},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["amount"] is None
    assert body["description"] is None
    assert body["expected_close_date"] is None
    assert body["name"] == opportunity["name"]


def test_empty_patch_is_a_noop(api_client: TestClient, opportunity: dict) -> None:
    response = api_client.patch(f"{URL}/{opportunity['id']}", json={})

    assert response.status_code == 200
    assert response.json() == opportunity


@pytest.mark.parametrize(
    "payload",
    [
        {"name": None},
        {"stage": None},
        {"stage": "WON"},
        {"amount": -5},
        {"account_id": str(uuid.uuid4())},
        {"created_at": "2026-01-01T00:00:00Z"},
    ],
)
def test_invalid_patch_returns_422_and_changes_nothing(
    api_client: TestClient, opportunity: dict, payload: dict
) -> None:
    response = api_client.patch(f"{URL}/{opportunity['id']}", json=payload)

    assert response.status_code == 422
    unchanged = api_client.patch(f"{URL}/{opportunity['id']}", json={}).json()
    assert unchanged == opportunity


def test_patch_unknown_opportunity_returns_404(api_client: TestClient) -> None:
    response = api_client.patch(f"{URL}/{uuid.uuid4()}", json={"stage": "PROPOSAL"})

    assert response.status_code == 404


def test_patch_owner_validation(
    api_client: TestClient, db_session: Session, opportunity: dict
) -> None:
    active = create_user(db_session)
    inactive = create_user(db_session, is_active=False)
    url = f"{URL}/{opportunity['id']}"

    assert api_client.patch(url, json={"owner_id": str(uuid.uuid4())}).status_code == 422
    assert api_client.patch(url, json={"owner_id": str(inactive.id)}).status_code == 422
    assigned = api_client.patch(url, json={"owner_id": str(active.id)})
    assert assigned.status_code == 200
    assert assigned.json()["owner_id"] == str(active.id)
    cleared = api_client.patch(url, json={"owner_id": None})
    assert cleared.json()["owner_id"] is None
