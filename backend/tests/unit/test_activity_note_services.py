"""Activity, note and timeline services with fake repositories and session."""

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import BusinessRuleError, NotFoundError, RelatedResourceNotFoundError
from app.models import Account, Activity, Lead, Note, Opportunity, User
from app.repositories.account import AccountRepository
from app.repositories.activity import ActivityRepository
from app.repositories.contact import ContactRepository
from app.repositories.lead import LeadRepository
from app.repositories.note import NoteRepository
from app.repositories.opportunity import OpportunityRepository
from app.repositories.timeline import TimelineRepository
from app.repositories.user import UserRepository
from app.schemas.activity import ActivityCreate
from app.schemas.note import NoteCreate
from app.services.activity_service import ActivityService
from app.services.crm_references import CrmRepositories
from app.services.note_service import NoteService
from app.services.timeline_service import TimelineService

ACCOUNT = uuid.uuid4()
OPPORTUNITY = uuid.uuid4()


def _repo(cls: type) -> MagicMock:
    repo = MagicMock(spec=cls)
    repo.add.side_effect = lambda obj: obj
    return repo


@pytest.fixture
def session() -> MagicMock:
    return MagicMock(spec=Session)


@pytest.fixture
def crm() -> CrmRepositories:
    accounts = _repo(AccountRepository)
    accounts.get.return_value = Account(id=ACCOUNT)
    opportunities = _repo(OpportunityRepository)
    opportunities.get.return_value = Opportunity(id=OPPORTUNITY, account_id=ACCOUNT)
    leads = _repo(LeadRepository)
    leads.get.return_value = Lead()
    return CrmRepositories(
        accounts=accounts,
        leads=leads,
        opportunities=opportunities,
        contacts=_repo(ContactRepository),
    )


@pytest.fixture
def users() -> MagicMock:
    repo = _repo(UserRepository)
    repo.get.return_value = User(is_active=True)
    return repo


@pytest.fixture
def activities() -> MagicMock:
    return _repo(ActivityRepository)


@pytest.fixture
def activity_service(
    session: MagicMock, activities: MagicMock, crm: CrmRepositories, users: MagicMock
) -> ActivityService:
    return ActivityService(session, activities, crm, users)


@pytest.fixture
def note_service(session: MagicMock, crm: CrmRepositories, users: MagicMock) -> NoteService:
    return NoteService(session, _repo(NoteRepository), crm, users)


def _activity_create(**fields: object) -> ActivityCreate:
    values: dict[str, object] = {
        "activity_type": "MEETING",
        "direction": "OUTBOUND",
        "occurred_at": datetime.now(UTC) - timedelta(days=1),
    }
    return ActivityCreate(**{**values, **fields})


def test_create_activity_stores_derived_account_and_commits_once(
    activity_service: ActivityService, activities: MagicMock, session: MagicMock
) -> None:
    activity = activity_service.create_activity(_activity_create(opportunity_id=OPPORTUNITY))

    assert isinstance(activity, Activity)
    assert activity.account_id == ACCOUNT
    assert activity.opportunity_id == OPPORTUNITY
    activities.add.assert_called_once_with(activity)
    session.commit.assert_called_once()


def test_create_lead_only_activity_keeps_account_null(
    activity_service: ActivityService,
) -> None:
    activity = activity_service.create_activity(_activity_create(lead_id=uuid.uuid4()))

    assert activity.account_id is None


@pytest.mark.parametrize(
    ("user", "error", "message"),
    [
        (None, RelatedResourceNotFoundError, "created_by_id does not refer to an existing user"),
        (User(is_active=False), BusinessRuleError, "created_by_id refers to an inactive user"),
    ],
)
def test_activity_creator_must_exist_and_be_active(
    activity_service: ActivityService,
    users: MagicMock,
    session: MagicMock,
    user: User | None,
    error: type[Exception],
    message: str,
) -> None:
    users.get.return_value = user

    with pytest.raises(error, match=message):
        activity_service.create_activity(
            _activity_create(account_id=ACCOUNT, created_by_id=uuid.uuid4())
        )
    session.commit.assert_not_called()
    session.rollback.assert_called_once()


def test_activity_cross_account_conflict_writes_nothing(
    activity_service: ActivityService, activities: MagicMock, session: MagicMock
) -> None:
    with pytest.raises(BusinessRuleError, match="different accounts"):
        activity_service.create_activity(
            _activity_create(account_id=uuid.uuid4(), opportunity_id=OPPORTUNITY)
        )
    activities.add.assert_not_called()
    session.commit.assert_not_called()


def _integrity_error(constraint: str) -> IntegrityError:
    orig = Exception("violation")
    orig.diag = SimpleNamespace(constraint_name=constraint)  # type: ignore[attr-defined]
    return IntegrityError("INSERT", {}, orig)


@pytest.mark.parametrize(
    ("constraint", "error"),
    [
        ("fk_activities_opportunity_id_opportunities", RelatedResourceNotFoundError),
        ("fk_activities_created_by_id_users", RelatedResourceNotFoundError),
        ("ck_activities_at_least_one_parent", BusinessRuleError),
    ],
)
def test_known_activity_constraint_violations_become_domain_errors(
    activity_service: ActivityService, session: MagicMock, constraint: str, error: type
) -> None:
    session.commit.side_effect = _integrity_error(constraint)

    with pytest.raises(error):
        activity_service.create_activity(_activity_create(opportunity_id=OPPORTUNITY))


def test_unknown_integrity_errors_are_reraised(
    activity_service: ActivityService, session: MagicMock
) -> None:
    session.commit.side_effect = _integrity_error("ck_activities_activity_type")

    with pytest.raises(IntegrityError):
        activity_service.create_activity(_activity_create(account_id=ACCOUNT))
    session.rollback.assert_called_once()


def test_list_lead_activities_requires_existing_lead(
    activity_service: ActivityService, crm: CrmRepositories, activities: MagicMock
) -> None:
    crm.leads.get.return_value = None  # type: ignore[attr-defined]

    with pytest.raises(NotFoundError):
        activity_service.list_lead_activities(uuid.uuid4(), limit=50, offset=0)
    activities.find_for_lead.assert_not_called()


def test_list_lead_activities_paginates(
    activity_service: ActivityService, activities: MagicMock, session: MagicMock
) -> None:
    lead_id = uuid.uuid4()
    activities.find_for_lead.return_value = [Activity()]
    activities.count_for_lead.return_value = 9

    page = activity_service.list_lead_activities(lead_id, limit=1, offset=4)

    activities.find_for_lead.assert_called_once_with(lead_id, limit=1, offset=4)
    assert (page.total, page.limit, page.offset, len(page.items)) == (9, 1, 4, 1)
    session.commit.assert_not_called()


def test_create_note_derives_account_from_opportunity(note_service: NoteService) -> None:
    note = note_service.create_note(NoteCreate(opportunity_id=OPPORTUNITY, body="Hi"))

    assert isinstance(note, Note)
    assert note.account_id == ACCOUNT


def test_create_note_rejects_inactive_author(note_service: NoteService, users: MagicMock) -> None:
    users.get.return_value = User(is_active=False)

    with pytest.raises(BusinessRuleError, match="author_id refers to an inactive user"):
        note_service.create_note(
            NoteCreate(account_id=ACCOUNT, author_id=uuid.uuid4(), body="Hi")
        )


def test_create_note_rejects_conflicting_accounts(note_service: NoteService) -> None:
    with pytest.raises(BusinessRuleError, match="account_id, opportunity_id"):
        note_service.create_note(
            NoteCreate(account_id=uuid.uuid4(), opportunity_id=OPPORTUNITY, body="Hi")
        )


def test_timeline_requires_existing_account() -> None:
    accounts = _repo(AccountRepository)
    accounts.get.return_value = None
    timeline = MagicMock(spec=TimelineRepository)

    with pytest.raises(NotFoundError):
        TimelineService(accounts, timeline).get_account_timeline(uuid.uuid4(), limit=5, offset=0)
    timeline.find_account_timeline.assert_not_called()


def test_timeline_passes_pagination_and_total_through() -> None:
    accounts = _repo(AccountRepository)
    accounts.get.return_value = Account()
    timeline = MagicMock(spec=TimelineRepository)
    records = [Activity(), Note()]
    timeline.find_account_timeline.return_value = records
    timeline.count_account_timeline.return_value = 12

    page = TimelineService(accounts, timeline).get_account_timeline(ACCOUNT, limit=2, offset=6)

    timeline.find_account_timeline.assert_called_once_with(ACCOUNT, limit=2, offset=6)
    assert list(page.items) == records
    assert (page.total, page.limit, page.offset) == (12, 2, 6)
