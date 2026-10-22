"""Route behavior with services overridden: status codes, response shapes, error mapping."""

import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.api.deps import (
    get_account_service,
    get_contact_service,
    get_lead_service,
    get_opportunity_service,
)
from app.core.config import Settings
from app.core.exceptions import BusinessRuleError, NotFoundError, RelatedResourceNotFoundError
from app.main import create_app
from app.models import Account, Contact, Lead, Opportunity, User
from app.models.enums import LeadSource, LeadStatus, OpportunityStage, UserRole
from app.services.account_service import AccountService
from app.services.contact_service import ContactService
from app.services.lead_service import LeadService
from app.services.opportunity_service import OpportunityService
from app.services.pagination import PageResult

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)

AUTH_USER = User(
    id=uuid.uuid4(),
    email="rep@example.com",
    full_name="Sales Rep",
    role=UserRole.SALES_REP,
    is_active=True,
)


def _account(**fields: object) -> Account:
    values: dict[str, object] = {
        "id": uuid.uuid4(),
        "name": "Acme",
        "annual_revenue": Decimal("1234.50"),
        "created_at": NOW,
        "updated_at": NOW,
    }
    return Account(**{**values, **fields})


def _lead() -> Lead:
    return Lead(
        id=uuid.uuid4(),
        company_name="Initech",
        contact_name="Peter",
        lead_source=LeadSource.WEB,
        status=LeadStatus.NEW,
        created_at=NOW,
        updated_at=NOW,
    )


def _opportunity() -> Opportunity:
    return Opportunity(
        id=uuid.uuid4(),
        account_id=uuid.uuid4(),
        name="Deal",
        stage=OpportunityStage.PROPOSAL,
        amount=Decimal("10.00"),
        expected_close_date=date(2026, 12, 31),
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.fixture
def services() -> dict[str, MagicMock]:
    return {
        "account": MagicMock(spec=AccountService),
        "contact": MagicMock(spec=ContactService),
        "lead": MagicMock(spec=LeadService),
        "opportunity": MagicMock(spec=OpportunityService),
    }


@pytest.fixture
def client(settings: Settings, services: dict[str, MagicMock]) -> Iterator[TestClient]:
    app = create_app(settings)
    app.dependency_overrides[get_account_service] = lambda: services["account"]
    app.dependency_overrides[get_contact_service] = lambda: services["contact"]
    app.dependency_overrides[get_lead_service] = lambda: services["lead"]
    app.dependency_overrides[get_opportunity_service] = lambda: services["opportunity"]
    # Authentication itself is covered by test_auth_routes / integration tests.
    app.dependency_overrides[get_current_user] = lambda: AUTH_USER
    with TestClient(app) as test_client:
        yield test_client


def test_create_account_returns_201_with_money_as_string(
    client: TestClient, services: dict[str, MagicMock]
) -> None:
    account = _account()
    services["account"].create_account.return_value = account

    response = client.post("/api/v1/accounts", json={"name": "Acme", "annual_revenue": 1234.5})

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == str(account.id)
    assert body["annual_revenue"] == "1234.50"
    assert body["created_at"].startswith("2026-10-20T12:00:00")


def test_list_accounts_returns_page_and_passes_filters(
    client: TestClient, services: dict[str, MagicMock]
) -> None:
    services["account"].list_accounts.return_value = PageResult(
        items=[_account()], total=11, limit=1, offset=5
    )

    response = client.get(
        "/api/v1/accounts", params={"industry": "Software", "limit": 1, "offset": 5}
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["total"], body["limit"], body["offset"], len(body["items"])) == (11, 1, 5, 1)
    services["account"].list_accounts.assert_called_once_with(
        industry="Software", limit=1, offset=5
    )


def test_list_defaults_to_limit_50_offset_0(
    client: TestClient, services: dict[str, MagicMock]
) -> None:
    services["lead"].list_leads.return_value = PageResult(items=[], total=0, limit=50, offset=0)

    client.get("/api/v1/leads")

    services["lead"].list_leads.assert_called_once_with(
        status=None, lead_source=None, limit=50, offset=0
    )


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
        {"limit": "ten"},
        {"status": "BOGUS"},
        {"lead_source": "BOGUS"},
    ],
)
def test_invalid_lead_list_params_return_422(client: TestClient, params: dict) -> None:
    assert client.get("/api/v1/leads", params=params).status_code == 422


def test_lead_filters_are_passed_as_enums(
    client: TestClient, services: dict[str, MagicMock]
) -> None:
    services["lead"].list_leads.return_value = PageResult(
        items=[_lead()], total=1, limit=50, offset=0
    )

    response = client.get("/api/v1/leads", params={"status": "QUALIFIED", "lead_source": "WEB"})

    assert response.status_code == 200
    services["lead"].list_leads.assert_called_once_with(
        status=LeadStatus.QUALIFIED, lead_source=LeadSource.WEB, limit=50, offset=0
    )


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (NotFoundError("Account x not found"), 404),
        (RelatedResourceNotFoundError("owner_id does not refer to an existing user"), 422),
        (BusinessRuleError("owner_id refers to an inactive user"), 422),
    ],
)
def test_domain_errors_map_to_http_status(
    client: TestClient, services: dict[str, MagicMock], error: Exception, status_code: int
) -> None:
    services["account"].get_account.side_effect = error

    response = client.get(f"/api/v1/accounts/{uuid.uuid4()}")

    assert response.status_code == status_code
    assert response.json() == {"detail": str(error)}


def test_malformed_uuid_returns_422_without_calling_service(
    client: TestClient, services: dict[str, MagicMock]
) -> None:
    assert client.get("/api/v1/leads/not-a-uuid").status_code == 422
    services["lead"].get_lead.assert_not_called()


def test_create_contact_returns_201(client: TestClient, services: dict[str, MagicMock]) -> None:
    account_id = uuid.uuid4()
    services["contact"].create_contact.return_value = Contact(
        id=uuid.uuid4(),
        account_id=account_id,
        first_name="Ada",
        last_name="Lovelace",
        created_at=NOW,
        updated_at=NOW,
    )

    response = client.post(
        "/api/v1/contacts",
        json={"account_id": str(account_id), "first_name": "Ada", "last_name": "Lovelace"},
    )

    assert response.status_code == 201
    assert response.json()["account_id"] == str(account_id)


def test_create_lead_returns_201(client: TestClient, services: dict[str, MagicMock]) -> None:
    services["lead"].create_lead.return_value = _lead()

    response = client.post(
        "/api/v1/leads",
        json={"company_name": "Initech", "contact_name": "Peter", "lead_source": "WEB"},
    )

    assert response.status_code == 201
    assert response.json()["status"] == "NEW"


def test_patch_opportunity_passes_only_supplied_fields(
    client: TestClient, services: dict[str, MagicMock]
) -> None:
    opportunity = _opportunity()
    services["opportunity"].update_opportunity.return_value = opportunity

    response = client.patch(
        f"/api/v1/opportunities/{opportunity.id}", json={"stage": "PROPOSAL", "amount": None}
    )

    assert response.status_code == 200
    opportunity_id, update = services["opportunity"].update_opportunity.call_args.args
    assert opportunity_id == opportunity.id
    assert update.model_dump(exclude_unset=True) == {
        "stage": OpportunityStage.PROPOSAL,
        "amount": None,
    }


def test_patch_unknown_field_returns_422(
    client: TestClient, services: dict[str, MagicMock]
) -> None:
    response = client.patch(
        f"/api/v1/opportunities/{uuid.uuid4()}", json={"account_id": str(uuid.uuid4())}
    )

    assert response.status_code == 422
    services["opportunity"].update_opportunity.assert_not_called()
