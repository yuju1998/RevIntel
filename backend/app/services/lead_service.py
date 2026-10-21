"""Lead business logic."""

import uuid

from sqlalchemy.orm import Session

from app.core.exceptions import BusinessRuleError, NotFoundError
from app.models.enums import LeadSource, LeadStatus
from app.models.lead import Lead
from app.repositories.lead import LeadRepository
from app.repositories.user import UserRepository
from app.schemas.lead import LeadCreate
from app.services.pagination import PageResult
from app.services.transaction import write_transaction
from app.services.user_validation import ensure_active_user, user_not_found_for

CONSTRAINT_ERRORS = {"fk_leads_owner_id_users": user_not_found_for("owner_id")}


class LeadService:
    def __init__(self, session: Session, leads: LeadRepository, users: UserRepository) -> None:
        self._session = session
        self._leads = leads
        self._users = users

    def create_lead(self, data: LeadCreate) -> Lead:
        if data.status == LeadStatus.CONVERTED:
            # Conversion will be an explicit workflow; the DB CHECK is the final backstop.
            raise BusinessRuleError("Leads cannot be created with status CONVERTED")
        with write_transaction(self._session, CONSTRAINT_ERRORS):
            ensure_active_user(self._users, data.owner_id, field="owner_id")
            lead = self._leads.add(Lead(**data.model_dump()))
        return lead

    def get_lead(self, lead_id: uuid.UUID) -> Lead:
        lead = self._leads.get(lead_id)
        if lead is None:
            raise NotFoundError(f"Lead {lead_id} not found")
        return lead

    def list_leads(
        self,
        *,
        status: LeadStatus | None,
        lead_source: LeadSource | None,
        limit: int,
        offset: int,
    ) -> PageResult[Lead]:
        items = self._leads.find(status=status, lead_source=lead_source, limit=limit, offset=offset)
        total = self._leads.count(status=status, lead_source=lead_source)
        return PageResult(items=items, total=total, limit=limit, offset=offset)
