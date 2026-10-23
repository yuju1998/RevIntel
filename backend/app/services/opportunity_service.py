"""Opportunity business logic."""

import uuid

from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.models.opportunity import Opportunity
from app.repositories.account import AccountRepository
from app.repositories.opportunity import OpportunityRepository
from app.repositories.user import UserRepository
from app.schemas.opportunity import OpportunityCreate, OpportunityUpdate
from app.services.crm_references import not_found, not_found_for
from app.services.transaction import write_transaction
from app.services.user_validation import ensure_active_user, user_not_found_for

CONSTRAINT_ERRORS = {
    "fk_opportunities_account_id_accounts": not_found_for("account_id", "account"),
    "fk_opportunities_owner_id_users": user_not_found_for("owner_id"),
}


class OpportunityService:
    def __init__(
        self,
        session: Session,
        opportunities: OpportunityRepository,
        accounts: AccountRepository,
        users: UserRepository,
    ) -> None:
        self._session = session
        self._opportunities = opportunities
        self._accounts = accounts
        self._users = users

    def create_opportunity(self, data: OpportunityCreate) -> Opportunity:
        with write_transaction(self._session, CONSTRAINT_ERRORS):
            if self._accounts.get(data.account_id) is None:
                raise not_found("account_id", "account")
            ensure_active_user(self._users, data.owner_id, field="owner_id")
            opportunity = self._opportunities.add(Opportunity(**data.model_dump()))
        return opportunity

    def update_opportunity(
        self, opportunity_id: uuid.UUID, data: OpportunityUpdate
    ) -> Opportunity:
        # Only fields present in the request are applied; omitted fields keep their values.
        changes = data.model_dump(exclude_unset=True)
        with write_transaction(self._session, CONSTRAINT_ERRORS):
            opportunity = self._opportunities.get(opportunity_id)
            if opportunity is None:
                raise NotFoundError(f"Opportunity {opportunity_id} not found")
            if "owner_id" in changes:
                ensure_active_user(self._users, changes["owner_id"], field="owner_id")
            for field, value in changes.items():
                setattr(opportunity, field, value)
        return opportunity
