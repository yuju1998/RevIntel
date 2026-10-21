"""Cross-account consistency and account_id derivation (fake repositories, no database)."""

import uuid
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import BusinessRuleError, RelatedResourceNotFoundError
from app.models import Account, Contact, Lead, Opportunity
from app.repositories.account import AccountRepository
from app.repositories.contact import ContactRepository
from app.repositories.lead import LeadRepository
from app.repositories.opportunity import OpportunityRepository
from app.services.crm_references import CrmRepositories, resolve_account_id

ACCOUNT_A = uuid.uuid4()
ACCOUNT_B = uuid.uuid4()
OPP_A, OPP_B = uuid.uuid4(), uuid.uuid4()
CONTACT_A, CONTACT_B = uuid.uuid4(), uuid.uuid4()
LEAD = uuid.uuid4()


@pytest.fixture
def repos() -> CrmRepositories:
    accounts = MagicMock(spec=AccountRepository)
    accounts.get.side_effect = lambda i: Account(id=i) if i in {ACCOUNT_A, ACCOUNT_B} else None
    opportunities = MagicMock(spec=OpportunityRepository)
    opportunities.get.side_effect = lambda i: {
        OPP_A: Opportunity(id=OPP_A, account_id=ACCOUNT_A),
        OPP_B: Opportunity(id=OPP_B, account_id=ACCOUNT_B),
    }.get(i)
    contacts = MagicMock(spec=ContactRepository)
    contacts.get.side_effect = lambda i: {
        CONTACT_A: Contact(id=CONTACT_A, account_id=ACCOUNT_A),
        CONTACT_B: Contact(id=CONTACT_B, account_id=ACCOUNT_B),
    }.get(i)
    leads = MagicMock(spec=LeadRepository)
    leads.get.side_effect = lambda i: Lead(id=i) if i == LEAD else None
    return CrmRepositories(
        accounts=accounts, leads=leads, opportunities=opportunities, contacts=contacts
    )


def _resolve(repos: CrmRepositories, **refs: uuid.UUID) -> uuid.UUID | None:
    values = {"account_id": None, "lead_id": None, "opportunity_id": None, "contact_id": None}
    return resolve_account_id(repos, **{**values, **refs})


@pytest.mark.parametrize(
    "refs",
    [
        {"account_id": ACCOUNT_A},
        {"opportunity_id": OPP_A},
        {"contact_id": CONTACT_A},
        {"opportunity_id": OPP_A, "contact_id": CONTACT_A},
        {"account_id": ACCOUNT_A, "opportunity_id": OPP_A, "contact_id": CONTACT_A},
        # A lead may coexist with account-scoped references (leads have no account).
        {"lead_id": LEAD, "opportunity_id": OPP_A},
        {"lead_id": LEAD, "account_id": ACCOUNT_A, "contact_id": CONTACT_A},
    ],
)
def test_consistent_references_resolve_to_their_account(
    repos: CrmRepositories, refs: dict
) -> None:
    assert _resolve(repos, **refs) == ACCOUNT_A


def test_account_is_derived_from_opportunity_when_omitted(repos: CrmRepositories) -> None:
    assert _resolve(repos, opportunity_id=OPP_B) == ACCOUNT_B


def test_account_is_derived_from_contact_when_omitted(repos: CrmRepositories) -> None:
    assert _resolve(repos, contact_id=CONTACT_B) == ACCOUNT_B


def test_lead_alone_implies_no_account(repos: CrmRepositories) -> None:
    assert _resolve(repos, lead_id=LEAD) is None


@pytest.mark.parametrize(
    ("refs", "fields"),
    [
        ({"account_id": ACCOUNT_A, "opportunity_id": OPP_B}, "account_id, opportunity_id"),
        ({"account_id": ACCOUNT_A, "contact_id": CONTACT_B}, "account_id, contact_id"),
        ({"opportunity_id": OPP_A, "contact_id": CONTACT_B}, "opportunity_id, contact_id"),
        (
            {"account_id": ACCOUNT_A, "opportunity_id": OPP_A, "contact_id": CONTACT_B},
            "account_id, opportunity_id, contact_id",
        ),
        ({"lead_id": LEAD, "opportunity_id": OPP_A, "contact_id": CONTACT_B}, "opportunity_id"),
    ],
)
def test_references_to_different_accounts_are_rejected(
    repos: CrmRepositories, refs: dict, fields: str
) -> None:
    with pytest.raises(BusinessRuleError, match="refer to different accounts") as exc_info:
        _resolve(repos, **refs)

    assert fields in exc_info.value.message


@pytest.mark.parametrize(
    ("field", "entity"),
    [
        ("account_id", "account"),
        ("lead_id", "lead"),
        ("opportunity_id", "opportunity"),
        ("contact_id", "contact"),
    ],
)
def test_missing_reference_names_its_field(
    repos: CrmRepositories, field: str, entity: str
) -> None:
    with pytest.raises(RelatedResourceNotFoundError) as exc_info:
        _resolve(repos, **{field: uuid.uuid4()})

    assert exc_info.value.message == f"{field} does not refer to an existing {entity}"
