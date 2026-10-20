"""Service business rules with fake repositories and session (no database)."""

import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import BusinessRuleError, NotFoundError, RelatedResourceNotFoundError
from app.models import Account, Contact, Lead, Opportunity, User
from app.models.enums import LeadSource, LeadStatus, OpportunityStage
from app.repositories.account import AccountRepository
from app.repositories.contact import ContactRepository
from app.repositories.lead import LeadRepository
from app.repositories.opportunity import OpportunityRepository
from app.repositories.user import UserRepository
from app.schemas.account import AccountCreate
from app.schemas.contact import ContactCreate
from app.schemas.lead import LeadCreate
from app.schemas.opportunity import OpportunityCreate, OpportunityUpdate
from app.services.account_service import AccountService
from app.services.contact_service import ContactService
from app.services.lead_service import LeadService
from app.services.opportunity_service import OpportunityService


def _repo(cls: type) -> MagicMock:
    repo = MagicMock(spec=cls)
    repo.add.side_effect = lambda obj: obj
    return repo


@pytest.fixture
def session() -> MagicMock:
    return MagicMock(spec=Session)


@pytest.fixture
def users() -> MagicMock:
    repo = _repo(UserRepository)
    repo.get.return_value = User(is_active=True)
    return repo


@pytest.fixture
def accounts() -> MagicMock:
    repo = _repo(AccountRepository)
    repo.get.return_value = Account(name="Acme")
    return repo


# --- owner validation (shared by account, lead and opportunity services) -----------------


@pytest.mark.parametrize(
    ("owner", "error"),
    [(None, RelatedResourceNotFoundError), (User(is_active=False), BusinessRuleError)],
)
def test_owner_must_exist_and_be_active(
    session: MagicMock, users: MagicMock, owner: User | None, error: type[Exception]
) -> None:
    users.get.return_value = owner
    service = AccountService(session, _repo(AccountRepository), users)

    with pytest.raises(error):
        service.create_account(AccountCreate(name="Acme", owner_id=uuid.uuid4()))

    session.commit.assert_not_called()
    session.rollback.assert_called_once()


def test_owner_is_optional(session: MagicMock, users: MagicMock) -> None:
    service = AccountService(session, _repo(AccountRepository), users)

    service.create_account(AccountCreate(name="Acme"))

    users.get.assert_not_called()
    session.commit.assert_called_once()


# --- accounts -----------------------------------------------------------------------------


def test_create_account_persists_fields_and_commits_once(
    session: MagicMock, users: MagicMock
) -> None:
    accounts = _repo(AccountRepository)
    service = AccountService(session, accounts, users)
    owner_id = uuid.uuid4()

    account = service.create_account(
        AccountCreate(name="Acme", annual_revenue="10.5", owner_id=owner_id)
    )

    assert isinstance(account, Account)
    assert account.annual_revenue == Decimal("10.50")
    assert account.owner_id == owner_id
    accounts.add.assert_called_once_with(account)
    session.commit.assert_called_once()


def test_get_unknown_account_raises_not_found(session: MagicMock, users: MagicMock) -> None:
    accounts = _repo(AccountRepository)
    accounts.get.return_value = None

    with pytest.raises(NotFoundError):
        AccountService(session, accounts, users).get_account(uuid.uuid4())


def test_list_accounts_passes_filters_and_counts_matches(
    session: MagicMock, users: MagicMock
) -> None:
    accounts = _repo(AccountRepository)
    accounts.find.return_value = [Account(name="A")]
    accounts.count.return_value = 7

    page = AccountService(session, accounts, users).list_accounts(
        industry="Software", limit=1, offset=3
    )

    accounts.find.assert_called_once_with(industry="Software", limit=1, offset=3)
    accounts.count.assert_called_once_with(industry="Software")
    assert (page.total, page.limit, page.offset, len(page.items)) == (7, 1, 3, 1)
    session.commit.assert_not_called()


def test_foreign_key_race_maps_to_related_resource_error(
    session: MagicMock, users: MagicMock
) -> None:
    orig = Exception("fk")
    orig.diag = SimpleNamespace(constraint_name="fk_accounts_owner_id_users")  # type: ignore[attr-defined]
    session.commit.side_effect = IntegrityError("INSERT", {}, orig)
    service = AccountService(session, _repo(AccountRepository), users)

    with pytest.raises(RelatedResourceNotFoundError):
        service.create_account(AccountCreate(name="Acme", owner_id=uuid.uuid4()))
    session.rollback.assert_called_once()


# --- contacts -----------------------------------------------------------------------------


def test_create_contact_requires_existing_account(session: MagicMock, accounts: MagicMock) -> None:
    accounts.get.return_value = None
    contacts = _repo(ContactRepository)

    with pytest.raises(RelatedResourceNotFoundError, match="account_id"):
        ContactService(session, contacts, accounts).create_contact(
            ContactCreate(account_id=uuid.uuid4(), first_name="A", last_name="B")
        )

    contacts.add.assert_not_called()
    session.commit.assert_not_called()


def test_create_contact(session: MagicMock, accounts: MagicMock) -> None:
    contact = ContactService(session, _repo(ContactRepository), accounts).create_contact(
        ContactCreate(account_id=uuid.uuid4(), first_name="A", last_name="B")
    )

    assert isinstance(contact, Contact)
    session.commit.assert_called_once()


# --- leads --------------------------------------------------------------------------------


def _lead_create(**fields: object) -> LeadCreate:
    return LeadCreate(company_name="Initech", contact_name="Peter", lead_source="WEB", **fields)


def test_create_lead_rejects_converted_status(session: MagicMock, users: MagicMock) -> None:
    leads = _repo(LeadRepository)

    with pytest.raises(BusinessRuleError, match="CONVERTED"):
        LeadService(session, leads, users).create_lead(_lead_create(status="CONVERTED"))

    leads.add.assert_not_called()
    session.commit.assert_not_called()


@pytest.mark.parametrize(
    "status", [s for s in LeadStatus if s != LeadStatus.CONVERTED]
)
def test_create_lead_accepts_other_statuses(
    session: MagicMock, users: MagicMock, status: LeadStatus
) -> None:
    lead = LeadService(session, _repo(LeadRepository), users).create_lead(
        _lead_create(status=status)
    )

    assert isinstance(lead, Lead)
    assert lead.status == status
    assert lead.converted_at is None


def test_list_leads_passes_filters(session: MagicMock, users: MagicMock) -> None:
    leads = _repo(LeadRepository)
    leads.find.return_value = []
    leads.count.return_value = 0

    LeadService(session, leads, users).list_leads(
        status=LeadStatus.QUALIFIED, lead_source=LeadSource.WEB, limit=10, offset=0
    )

    leads.find.assert_called_once_with(
        status=LeadStatus.QUALIFIED, lead_source=LeadSource.WEB, limit=10, offset=0
    )
    leads.count.assert_called_once_with(status=LeadStatus.QUALIFIED, lead_source=LeadSource.WEB)


def test_get_unknown_lead_raises_not_found(session: MagicMock, users: MagicMock) -> None:
    leads = _repo(LeadRepository)
    leads.get.return_value = None

    with pytest.raises(NotFoundError):
        LeadService(session, leads, users).get_lead(uuid.uuid4())


# --- opportunities ------------------------------------------------------------------------


@pytest.fixture
def opportunities() -> MagicMock:
    return _repo(OpportunityRepository)


@pytest.fixture
def opportunity_service(
    session: MagicMock, opportunities: MagicMock, accounts: MagicMock, users: MagicMock
) -> OpportunityService:
    return OpportunityService(session, opportunities, accounts, users)


def test_create_opportunity_requires_existing_account(
    opportunity_service: OpportunityService, accounts: MagicMock, opportunities: MagicMock
) -> None:
    accounts.get.return_value = None

    with pytest.raises(RelatedResourceNotFoundError, match="account_id"):
        opportunity_service.create_opportunity(
            OpportunityCreate(account_id=uuid.uuid4(), name="Deal", stage="PROPOSAL")
        )
    opportunities.add.assert_not_called()


def _existing_opportunity() -> Opportunity:
    return Opportunity(
        account_id=uuid.uuid4(),
        name="Deal",
        stage=OpportunityStage.PROSPECTING,
        amount=Decimal("100.00"),
        description="Keep me",
    )


def test_update_applies_only_supplied_fields(
    opportunity_service: OpportunityService, opportunities: MagicMock, session: MagicMock
) -> None:
    existing = _existing_opportunity()
    opportunities.get.return_value = existing

    updated = opportunity_service.update_opportunity(
        uuid.uuid4(), OpportunityUpdate(stage="NEGOTIATION", amount=None)
    )

    assert updated is existing
    assert updated.stage == OpportunityStage.NEGOTIATION
    assert updated.amount is None
    assert updated.name == "Deal"
    assert updated.description == "Keep me"
    session.commit.assert_called_once()


def test_empty_update_changes_nothing(
    opportunity_service: OpportunityService, opportunities: MagicMock, users: MagicMock
) -> None:
    existing = _existing_opportunity()
    opportunities.get.return_value = existing

    opportunity_service.update_opportunity(uuid.uuid4(), OpportunityUpdate())

    assert (existing.name, existing.stage, existing.amount) == (
        "Deal",
        OpportunityStage.PROSPECTING,
        Decimal("100.00"),
    )
    users.get.assert_not_called()


def test_update_unknown_opportunity_raises_not_found(
    opportunity_service: OpportunityService, opportunities: MagicMock, session: MagicMock
) -> None:
    opportunities.get.return_value = None

    with pytest.raises(NotFoundError):
        opportunity_service.update_opportunity(uuid.uuid4(), OpportunityUpdate(name="X"))
    session.commit.assert_not_called()


def test_update_validates_new_owner_but_allows_clearing(
    opportunity_service: OpportunityService, opportunities: MagicMock, users: MagicMock
) -> None:
    opportunities.get.return_value = _existing_opportunity()
    users.get.return_value = User(is_active=False)

    with pytest.raises(BusinessRuleError):
        opportunity_service.update_opportunity(
            uuid.uuid4(), OpportunityUpdate(owner_id=uuid.uuid4())
        )

    users.get.reset_mock()
    opportunity_service.update_opportunity(uuid.uuid4(), OpportunityUpdate(owner_id=None))
    users.get.assert_not_called()
