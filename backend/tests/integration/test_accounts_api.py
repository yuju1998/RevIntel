import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from tests.integration.factories import create_user

pytestmark = pytest.mark.integration

URL = "/api/v1/accounts"


def _create(client: TestClient, **fields: object) -> dict:
    response = client.post(URL, json={"name": "Acme", **fields})
    assert response.status_code == 201, response.json()
    return response.json()


def test_create_and_get_account(api_client: TestClient, db_session: Session) -> None:
    owner = create_user(db_session)

    created = _create(
        api_client,
        name="  Acme Corp  ",
        industry="Software",
        employee_count=250,
        annual_revenue="1234567.50",
        website="https://acme.example",
        owner_id=str(owner.id),
    )

    assert created["name"] == "Acme Corp"
    assert created["annual_revenue"] == "1234567.50"
    assert created["owner_id"] == str(owner.id)
    assert created["created_at"] == created["updated_at"]
    fetched = api_client.get(f"{URL}/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.json() == created


def test_get_unknown_account_returns_404(api_client: TestClient) -> None:
    response = api_client.get(f"{URL}/{uuid.uuid4()}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_account_with_malformed_uuid_returns_422(api_client: TestClient) -> None:
    assert api_client.get(f"{URL}/not-a-uuid").status_code == 422


def test_account_names_are_not_unique(api_client: TestClient) -> None:
    _create(api_client, name="Same")
    _create(api_client, name="Same")


@pytest.mark.parametrize(
    "fields",
    [{"employee_count": -1}, {"annual_revenue": "-0.01"}, {"annual_revenue": "1.001"}],
)
def test_invalid_numbers_return_422(api_client: TestClient, fields: dict) -> None:
    assert api_client.post(URL, json={"name": "Acme", **fields}).status_code == 422


def test_unknown_owner_returns_422(api_client: TestClient) -> None:
    response = api_client.post(URL, json={"name": "Acme", "owner_id": str(uuid.uuid4())})

    assert response.status_code == 422
    assert "owner_id" in response.json()["detail"]


def test_inactive_owner_returns_422(api_client: TestClient, db_session: Session) -> None:
    owner = create_user(db_session, is_active=False)

    response = api_client.post(URL, json={"name": "Acme", "owner_id": str(owner.id)})

    assert response.status_code == 422
    assert "inactive" in response.json()["detail"]


def test_list_accounts_paginates_newest_first(api_client: TestClient) -> None:
    ids = [_create(api_client, name=f"Account {i}")["id"] for i in range(3)]

    first = api_client.get(URL, params={"limit": 2}).json()
    second = api_client.get(URL, params={"limit": 2, "offset": 2}).json()
    beyond = api_client.get(URL, params={"offset": 10}).json()

    assert first["total"] == 3 and first["limit"] == 2 and first["offset"] == 0
    assert [item["id"] for item in first["items"] + second["items"]] == list(reversed(ids))
    assert beyond == {"items": [], "total": 3, "limit": 50, "offset": 10}


def test_industry_filter_is_exact_and_total_counts_matches(api_client: TestClient) -> None:
    for i in range(3):
        _create(api_client, name=f"Soft {i}", industry="Software")
    _create(api_client, name="Bank", industry="Finance")
    _create(api_client, name="Lower", industry="software")
    _create(api_client, name="Unknown")

    page = api_client.get(URL, params={"industry": "Software", "limit": 2, "offset": 1}).json()

    assert page["total"] == 3
    assert len(page["items"]) == 2
    assert {item["industry"] for item in page["items"]} == {"Software"}


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}])
def test_invalid_pagination_returns_422(api_client: TestClient, params: dict) -> None:
    assert api_client.get(URL, params=params).status_code == 422
