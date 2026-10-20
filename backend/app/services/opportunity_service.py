"""Opportunity business logic."""

import uuid

from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError, RelatedResourceNotFoundError
from app.models.opportunity import Opportunity
from app.repositories.account import AccountRepository
from app.repositories.opportunity import OpportunityRepository
from app.repositories.user import UserRepository
from app.schemas.opportunity import OpportunityCreate, OpportunityUpdate
from app.services.ownership import ensure_active_owner, owner_not_found
from app.services.transaction import write_transaction


def _account_not_found() -> RelatedResourceNotFoundError:
    return RelatedResourceNotFoundError("account_id does not refer to an existing account")


CONSTRAINT_ERRORS = {
    "fk_opportunities_account_id_accounts": _account_not_found,
    "fk_opportunities_owner_id_users": owner_not_found,
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
                raise _account_not_found()
            ensure_active_owner(self._users, data.owner_id)
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
                ensure_active_owner(self._users, changes["owner_id"])
            for field, value in changes.items():
                setattr(opportunity, field, value)
        return opportunity
