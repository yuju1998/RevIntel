import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from tests.integration.factories import create_account

pytestmark = pytest.mark.integration

URL = "/api/v1/contacts"


def test_create_contact(api_client: TestClient, db_session: Session) -> None:
    account = create_account(db_session)

    response = api_client.post(
        URL,
        json={
            "account_id": str(account.id),
            "first_name": "Ada",
            "last_name": "Lovelace",
            "email": "  Ada.Lovelace@Example.COM ",
            "phone": "+1 555 0100",
            "job_title": "CTO",
        },
    )

    assert response.status_code == 201, response.json()
    body = response.json()
    assert body["account_id"] == str(account.id)
    # Trimmed and validated, but casing preserved (stored as entered).
    assert body["email"] == "Ada.Lovelace@Example.COM"


def test_contact_emails_are_not_unique(api_client: TestClient, db_session: Session) -> None:
    account = create_account(db_session)
    payload = {
        "account_id": str(account.id),
        "first_name": "A",
        "last_name": "B",
        "email": "same@example.com",
    }

    assert api_client.post(URL, json=payload).status_code == 201
    assert api_client.post(URL, json=payload).status_code == 201


def test_nonexistent_account_returns_422(api_client: TestClient) -> None:
    response = api_client.post(
        URL, json={"account_id": str(uuid.uuid4()), "first_name": "A", "last_name": "B"}
    )

    assert response.status_code == 422
    assert "account_id" in response.json()["detail"]


@pytest.mark.parametrize(
    "fields",
    [
        {"account_id": "not-a-uuid"},
        {"email": "not-an-email"},
        {"first_name": "   "},
        {"phone": "1" * 51},
    ],
)
def test_invalid_contact_returns_422(
    api_client: TestClient, db_session: Session, fields: dict
) -> None:
    account = create_account(db_session)
    payload = {"account_id": str(account.id), "first_name": "A", "last_name": "B", **fields}

    assert api_client.post(URL, json=payload).status_code == 422
