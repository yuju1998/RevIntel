"""Read CRM rows for indexing, in keyset-paged batches (ordered by id). Read-only."""

import uuid
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session, aliased

from app.ai.crm_documents import (
    AccountSource,
    ActivitySource,
    NoteSource,
    OpportunitySource,
)
from app.models import Account, Activity, Contact, Note, Opportunity

_ActivityOpportunity = aliased(Opportunity)
_NoteOpportunity = aliased(Opportunity)


class CrmSourceRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def _pages(
        self,
        stmt: Select[Any],
        id_column: Any,
        build: Callable[[Any], Any],
        batch_size: int,
    ) -> Iterator[list[Any]]:
        after: uuid.UUID | None = None
        while True:
            page_stmt = stmt.order_by(id_column).limit(batch_size)
            if after is not None:
                page_stmt = page_stmt.where(id_column > after)
            rows: Sequence[Any] = self._session.execute(page_stmt).all()
            if not rows:
                return
            yield [build(row) for row in rows]
            after = rows[-1].id

    def accounts(
        self, *, account_id: uuid.UUID | None = None, batch_size: int = 500
    ) -> Iterator[list[AccountSource]]:
        stmt = select(
            Account.id,
            Account.name,
            Account.industry,
            Account.employee_count,
            Account.website,
            Account.updated_at,
        )
        if account_id is not None:
            stmt = stmt.where(Account.id == account_id)
        return self._pages(stmt, Account.id, lambda row: AccountSource(**row._mapping), batch_size)

    def opportunities(
        self, *, account_id: uuid.UUID | None = None, batch_size: int = 500
    ) -> Iterator[list[OpportunitySource]]:
        stmt = select(
            Opportunity.id,
            Opportunity.account_id,
            Opportunity.name,
            Opportunity.stage,
            Opportunity.amount,
            Opportunity.expected_close_date,
            Opportunity.description,
            Opportunity.updated_at,
        )
        if account_id is not None:
            stmt = stmt.where(Opportunity.account_id == account_id)
        return self._pages(
            stmt, Opportunity.id, lambda row: OpportunitySource(**row._mapping), batch_size
        )

    def activities(
        self, *, account_id: uuid.UUID | None = None, batch_size: int = 500
    ) -> Iterator[list[ActivitySource]]:
        """Activities with the accounts of their opportunity and contact (for resolution)."""
        stmt = (
            select(
                Activity.id,
                Activity.activity_type,
                Activity.direction,
                Activity.subject,
                Activity.body,
                Activity.occurred_at,
                Activity.account_id,
                Activity.opportunity_id,
                _ActivityOpportunity.account_id.label("opportunity_account_id"),
                Activity.contact_id,
                Contact.account_id.label("contact_account_id"),
                Activity.lead_id,
            )
            .outerjoin(_ActivityOpportunity, _ActivityOpportunity.id == Activity.opportunity_id)
            .outerjoin(Contact, Contact.id == Activity.contact_id)
        )
        if account_id is not None:
            stmt = stmt.where(
                or_(
                    Activity.account_id == account_id,
                    _ActivityOpportunity.account_id == account_id,
                    Contact.account_id == account_id,
                )
            )
        return self._pages(
            stmt, Activity.id, lambda row: ActivitySource(**row._mapping), batch_size
        )

    def notes(
        self, *, account_id: uuid.UUID | None = None, batch_size: int = 500
    ) -> Iterator[list[NoteSource]]:
        stmt = select(
            Note.id,
            Note.body,
            Note.created_at,
            Note.account_id,
            Note.opportunity_id,
            _NoteOpportunity.account_id.label("opportunity_account_id"),
            Note.lead_id,
        ).outerjoin(_NoteOpportunity, _NoteOpportunity.id == Note.opportunity_id)
        if account_id is not None:
            stmt = stmt.where(
                or_(Note.account_id == account_id, _NoteOpportunity.account_id == account_id)
            )
        return self._pages(stmt, Note.id, lambda row: NoteSource(**row._mapping), batch_size)
