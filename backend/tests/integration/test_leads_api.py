import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from tests.integration.factories import create_user

pytestmark = pytest.mark.integration

URL = "/api/v1/leads"


def _create(client: TestClient, **fields: object) -> dict:
    payload = {"company_name": "Initech", "contact_name": "Peter", "lead_source": "WEB"}
    response = client.post(URL, json={**payload, **fields})
    assert response.status_code == 201, response.json()
    return response.json()


def test_create_and_get_lead(api_client: TestClient, db_session: Session) -> None:
    owner = create_user(db_session)

    created = _create(
        api_client,
        email="Peter@Initech.example",
        industry="Software",
        company_size=40,
        estimated_value=25000,
        owner_id=str(owner.id),
    )

    assert created["status"] == "NEW"
    assert created["converted_at"] is None
    assert created["estimated_value"] == "25000.00"
    assert created["email"] == "Peter@Initech.example"
    fetched = api_client.get(f"{URL}/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.json() == created


def test_explicit_non_converted_status_is_accepted(api_client: TestClient) -> None:
    assert _create(api_client, status="QUALIFIED")["status"] == "QUALIFIED"


def test_converted_status_on_create_returns_422(api_client: TestClient) -> None:
    response = api_client.post(
        URL,
        json={
            "company_name": "Initech",
            "contact_name": "Peter",
            "lead_source": "WEB",
            "status": "CONVERTED",
        },
    )

    assert response.status_code == 422
    assert "CONVERTED" in response.json()["detail"]


def test_converted_at_is_not_accepted(api_client: TestClient) -> None:
    response = api_client.post(
        URL,
        json={
            "company_name": "Initech",
            "contact_name": "Peter",
            "lead_source": "WEB",
            "converted_at": "2026-01-01T00:00:00Z",
        },
    )

    assert response.status_code == 422


def test_get_unknown_lead_returns_404(api_client: TestClient) -> None:
    assert api_client.get(f"{URL}/{uuid.uuid4()}").status_code == 404


@pytest.mark.parametrize(
    "fields",
    [
        {"company_size": -1},
        {"estimated_value": -1},
        {"lead_source": "CARRIER_PIGEON"},
        {"owner_id": str(uuid.uuid4())},
    ],
)
def test_invalid_lead_returns_422(api_client: TestClient, fields: dict) -> None:
    payload = {"company_name": "Initech", "contact_name": "Peter", "lead_source": "WEB"}
    assert api_client.post(URL, json={**payload, **fields}).status_code == 422


@pytest.fixture
def seeded_leads(api_client: TestClient) -> None:
    for status, source, count in [
        ("QUALIFIED", "WEB", 3),
        ("QUALIFIED", "REFERRAL", 2),
        ("NEW", "WEB", 1),
        ("WORKING", "EVENT", 1),
    ]:
        for _ in range(count):
            _create(api_client, status=status, lead_source=source)


@pytest.mark.usefixtures("seeded_leads")
@pytest.mark.parametrize(
    ("params", "expected_total"),
    [
        ({}, 7),
        ({"status": "QUALIFIED"}, 5),
        ({"lead_source": "WEB"}, 4),
        ({"status": "QUALIFIED", "lead_source": "WEB"}, 3),
        ({"status": "DISQUALIFIED"}, 0),
    ],
)
def test_filters_and_filtered_total(
    api_client: TestClient, params: dict, expected_total: int
) -> None:
    page = api_client.get(URL, params=params).json()

    assert page["total"] == expected_total
    assert len(page["items"]) == expected_total
    for item in page["items"]:
        for key, value in params.items():
            assert item[key] == value


@pytest.mark.usefixtures("seeded_leads")
def test_pagination_combines_with_filters(api_client: TestClient) -> None:
    params = {"status": "QUALIFIED", "lead_source": "WEB", "limit": 2}

    first = api_client.get(URL, params=params).json()
    second = api_client.get(URL, params={**params, "offset": 2}).json()

    assert first["total"] == second["total"] == 3
    assert len(first["items"]) == 2 and len(second["items"]) == 1
    ids = [item["id"] for item in first["items"] + second["items"]]
    assert len(set(ids)) == 3


@pytest.mark.parametrize("params", [{"status": "BOGUS"}, {"lead_source": "BOGUS"}])
def test_invalid_enum_filter_returns_422(api_client: TestClient, params: dict) -> None:
    assert api_client.get(URL, params=params).status_code == 422
