"""Note business logic."""

import uuid

from sqlalchemy.orm import Session

from app.core.exceptions import BusinessRuleError
from app.core.security import invalid_token
from app.models.note import Note
from app.repositories.note import NoteRepository
from app.schemas.note import PARENT_FIELDS, NoteCreate
from app.services.crm_references import CrmRepositories, not_found_for, resolve_account_id
from app.services.transaction import write_transaction


def _parent_required() -> BusinessRuleError:
    return BusinessRuleError(f"at least one of {', '.join(PARENT_FIELDS)} is required")


# Backstops for races between validation and commit; other IntegrityErrors are re-raised.
CONSTRAINT_ERRORS = {
    "fk_notes_account_id_accounts": not_found_for("account_id", "account"),
    "fk_notes_lead_id_leads": not_found_for("lead_id", "lead"),
    "fk_notes_opportunity_id_opportunities": not_found_for("opportunity_id", "opportunity"),
    # The authenticated user was deleted mid-request: their credentials are no longer valid.
    "fk_notes_author_id_users": invalid_token,
    "ck_notes_at_least_one_parent": _parent_required,
}


class NoteService:
    def __init__(
        self,
        session: Session,
        notes: NoteRepository,
        crm: CrmRepositories,
    ) -> None:
        self._session = session
        self._notes = notes
        self._crm = crm

    def create_note(self, data: NoteCreate, *, author_id: uuid.UUID) -> Note:
        """author_id is the authenticated (active) user, never client input."""
        with write_transaction(self._session, CONSTRAINT_ERRORS):
            account_id = resolve_account_id(
                self._crm,
                account_id=data.account_id,
                lead_id=data.lead_id,
                opportunity_id=data.opportunity_id,
            )
            # Store the canonical (possibly derived) account so account timelines are complete.
            note = self._notes.add(
                Note(
                    **data.model_dump(exclude={"account_id"}),
                    account_id=account_id,
                    author_id=author_id,
                )
            )
        return note
