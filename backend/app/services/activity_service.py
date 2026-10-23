"""Activity business logic."""

import uuid

from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.core.security import invalid_token
from app.models.activity import Activity
from app.repositories.activity import ActivityRepository
from app.schemas.activity import PARENT_FIELDS, ActivityCreate
from app.services.crm_references import (
    CrmRepositories,
    not_found_for,
    parent_required_for,
    resolve_account_id,
)
from app.services.pagination import PageResult
from app.services.transaction import write_transaction

# Backstops for races between validation and commit; other IntegrityErrors are re-raised.
CONSTRAINT_ERRORS = {
    "fk_activities_account_id_accounts": not_found_for("account_id", "account"),
    "fk_activities_lead_id_leads": not_found_for("lead_id", "lead"),
    "fk_activities_opportunity_id_opportunities": not_found_for("opportunity_id", "opportunity"),
    "fk_activities_contact_id_contacts": not_found_for("contact_id", "contact"),
    # The authenticated user was deleted mid-request: their credentials are no longer valid.
    "fk_activities_created_by_id_users": invalid_token,
    "ck_activities_at_least_one_parent": parent_required_for(PARENT_FIELDS),
}


class ActivityService:
    def __init__(
        self,
        session: Session,
        activities: ActivityRepository,
        crm: CrmRepositories,
    ) -> None:
        self._session = session
        self._activities = activities
        self._crm = crm

    def create_activity(self, data: ActivityCreate, *, created_by_id: uuid.UUID) -> Activity:
        """created_by_id is the authenticated (active) user, never client input."""
        with write_transaction(self._session, CONSTRAINT_ERRORS):
            account_id = resolve_account_id(
                self._crm,
                account_id=data.account_id,
                lead_id=data.lead_id,
                opportunity_id=data.opportunity_id,
                contact_id=data.contact_id,
            )
            # Store the canonical (possibly derived) account so account timelines are complete.
            activity = self._activities.add(
                Activity(
                    **data.model_dump(exclude={"account_id"}),
                    account_id=account_id,
                    created_by_id=created_by_id,
                )
            )
        return activity

    def list_lead_activities(
        self, lead_id: uuid.UUID, *, limit: int, offset: int
    ) -> PageResult[Activity]:
        if self._crm.leads.get(lead_id) is None:
            raise NotFoundError(f"Lead {lead_id} not found")
        items = self._activities.find_for_lead(lead_id, limit=limit, offset=offset)
        total = self._activities.count_for_lead(lead_id)
        return PageResult(items=items, total=total, limit=limit, offset=offset)
