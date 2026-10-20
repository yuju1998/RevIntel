import uuid
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.models.enums import LeadStatus, OpportunityStage
from app.schemas.account import AccountCreate
from app.schemas.contact import ContactCreate
from app.schemas.lead import LeadCreate
from app.schemas.opportunity import OpportunityCreate, OpportunityUpdate

LEAD = {"company_name": "Initech", "contact_name": "Peter", "lead_source": "WEB"}


def test_account_create_trims_and_accepts_numbers_or_strings_for_money() -> None:
    from_number = AccountCreate(name="  Acme ", annual_revenue=1234.5)
    from_string = AccountCreate(name="Acme", annual_revenue="1234.50")

    assert from_number.name == "Acme"
    assert from_number.annual_revenue == Decimal("1234.50")
    assert str(from_string.annual_revenue) == "1234.50"


def test_money_is_scaled_to_cents_without_rounding() -> None:
    assert str(AccountCreate(name="A", annual_revenue=25000).annual_revenue) == "25000.00"


def test_money_is_serialized_as_a_string() -> None:
    assert AccountCreate(name="A", annual_revenue="10").model_dump(mode="json")[
        "annual_revenue"
    ] == "10.00"


@pytest.mark.parametrize(
    "fields",
    [
        {"name": ""},
        {"name": "   "},
        {"name": "x" * 256},
        {"industry": "x" * 256},
        {"website": "x" * 2049},
        {"employee_count": -1},
        {"employee_count": 2_147_483_648},
        {"annual_revenue": -1},
        {"annual_revenue": "1.001"},
        {"annual_revenue": "NaN"},
        {"annual_revenue": "Infinity"},
        # NUMERIC(16,2) allows at most 14 digits before the decimal point.
        {"annual_revenue": "1" * 15},
        {"owner_id": "not-a-uuid"},
        {"unexpected": "field"},
    ],
)
def test_account_create_rejects_invalid_input(fields: dict) -> None:
    with pytest.raises(ValidationError):
        AccountCreate(**{"name": "Acme", **fields})


def test_account_create_accepts_maximum_values() -> None:
    account = AccountCreate(
        name="x" * 255, employee_count=2_147_483_647, annual_revenue="99999999999999.99"
    )

    assert account.annual_revenue == Decimal("99999999999999.99")


def test_crm_email_is_trimmed_and_validated_but_keeps_casing() -> None:
    contact = ContactCreate(
        account_id=uuid.uuid4(), first_name="A", last_name="B", email="  Ada@Example.COM "
    )

    assert contact.email == "Ada@Example.COM"


@pytest.mark.parametrize("email", ["not-an-email", "a@", "@example.com", "a b@example.com", ""])
def test_crm_email_rejects_invalid_addresses(email: str) -> None:
    with pytest.raises(ValidationError):
        ContactCreate(account_id=uuid.uuid4(), first_name="A", last_name="B", email=email)


def test_lead_create_defaults_status_to_new() -> None:
    assert LeadCreate(**LEAD).status == LeadStatus.NEW


def test_lead_create_does_not_accept_converted_at() -> None:
    with pytest.raises(ValidationError):
        LeadCreate(**LEAD, converted_at="2026-01-01T00:00:00Z")


@pytest.mark.parametrize(
    "fields",
    [
        {"lead_source": "CARRIER_PIGEON"},
        {"status": "WON"},
        {"company_size": -1},
        {"estimated_value": -1},
        # NUMERIC(14,2) allows at most 12 digits before the decimal point.
        {"estimated_value": "1" * 13},
    ],
)
def test_lead_create_rejects_invalid_input(fields: dict) -> None:
    with pytest.raises(ValidationError):
        LeadCreate(**{**LEAD, **fields})


def test_opportunity_create_requires_account_name_and_stage() -> None:
    with pytest.raises(ValidationError) as exc_info:
        OpportunityCreate()  # type: ignore[call-arg]

    missing = {error["loc"][0] for error in exc_info.value.errors()}
    assert missing == {"account_id", "name", "stage"}


def test_opportunity_update_tracks_only_supplied_fields() -> None:
    update = OpportunityUpdate(stage="NEGOTIATION", amount=None)

    assert update.model_dump(exclude_unset=True) == {
        "stage": OpportunityStage.NEGOTIATION,
        "amount": None,
    }
    assert OpportunityUpdate().model_dump(exclude_unset=True) == {}


@pytest.mark.parametrize(
    "payload",
    [
        {"name": None},
        {"stage": None},
        {"name": "  "},
        {"amount": -1},
        {"account_id": str(uuid.uuid4())},
        {"created_at": "2026-01-01T00:00:00Z"},
    ],
)
def test_opportunity_update_rejects_invalid_input(payload: dict) -> None:
    with pytest.raises(ValidationError):
        OpportunityUpdate(**payload)


@pytest.mark.parametrize("field", ["amount", "expected_close_date", "description", "owner_id"])
def test_opportunity_update_allows_null_for_nullable_fields(field: str) -> None:
    assert OpportunityUpdate(**{field: None}).model_dump(exclude_unset=True) == {field: None}
