"""CRM rows → searchable source documents (pure functions; no database access).

Every document has a resolved `account_id`; sources that resolve to no account (or to
conflicting accounts) produce no document and are never indexed. Leads have no account in
the schema, so a lead-only activity or note cannot be resolved and is skipped.
"""

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from app.models.enums import DocumentEntityType, DocumentSourceType


@dataclass(frozen=True)
class SourceDocument:
    source_type: DocumentSourceType
    source_id: uuid.UUID
    account_id: uuid.UUID
    entity_type: DocumentEntityType
    entity_id: uuid.UUID
    text: str
    occurred_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActivitySource:
    id: uuid.UUID
    activity_type: str
    direction: str
    subject: str | None
    body: str | None
    occurred_at: datetime
    account_id: uuid.UUID | None
    opportunity_id: uuid.UUID | None
    opportunity_account_id: uuid.UUID | None
    contact_id: uuid.UUID | None
    contact_account_id: uuid.UUID | None
    lead_id: uuid.UUID | None


@dataclass(frozen=True)
class NoteSource:
    id: uuid.UUID
    body: str
    created_at: datetime
    account_id: uuid.UUID | None
    opportunity_id: uuid.UUID | None
    opportunity_account_id: uuid.UUID | None
    lead_id: uuid.UUID | None


@dataclass(frozen=True)
class OpportunitySource:
    id: uuid.UUID
    account_id: uuid.UUID
    name: str
    stage: str
    amount: Decimal | None
    expected_close_date: date | None
    description: str | None
    updated_at: datetime


@dataclass(frozen=True)
class AccountSource:
    id: uuid.UUID
    name: str
    industry: str | None
    employee_count: int | None
    website: str | None
    updated_at: datetime


def resolve_account(*candidates: uuid.UUID | None) -> uuid.UUID | None:
    """The single account the references agree on, or None (no account, or a conflict)."""
    accounts = {candidate for candidate in candidates if candidate is not None}
    return accounts.pop() if len(accounts) == 1 else None


def _value(enum_or_str: Any) -> str:
    return str(getattr(enum_or_str, "value", enum_or_str))


def _label(value: str) -> str:
    return value.replace("_", " ").capitalize()


def activity_document(row: ActivitySource) -> SourceDocument | None:
    # Direct account first, then through the opportunity and contact (all must agree).
    account_id = resolve_account(row.account_id, row.opportunity_account_id, row.contact_account_id)
    subject, body = (row.subject or "").strip(), (row.body or "").strip()
    if account_id is None or not (subject or body):
        return None
    activity_type, direction = _value(row.activity_type), _value(row.direction)
    header = f"{_label(activity_type)} ({direction.lower()})"
    text = "\n".join(part for part in (f"{header}: {subject}" if subject else header, body) if part)
    if row.opportunity_id is not None:
        entity = (DocumentEntityType.OPPORTUNITY, row.opportunity_id)
    elif row.contact_id is not None:
        entity = (DocumentEntityType.CONTACT, row.contact_id)
    else:
        entity = (DocumentEntityType.ACCOUNT, account_id)
    return SourceDocument(
        source_type=DocumentSourceType.ACTIVITY,
        source_id=row.id,
        account_id=account_id,
        entity_type=entity[0],
        entity_id=entity[1],
        text=text,
        occurred_at=row.occurred_at,
        metadata={"activity_type": activity_type, "direction": direction},
    )


def note_document(row: NoteSource) -> SourceDocument | None:
    account_id = resolve_account(row.account_id, row.opportunity_account_id)
    text = (row.body or "").strip()
    if account_id is None or not text:
        return None
    entity = (
        (DocumentEntityType.OPPORTUNITY, row.opportunity_id)
        if row.opportunity_id is not None
        else (DocumentEntityType.ACCOUNT, account_id)
    )
    return SourceDocument(
        source_type=DocumentSourceType.NOTE,
        source_id=row.id,
        account_id=account_id,
        entity_type=entity[0],
        entity_id=entity[1],
        text=text,
        occurred_at=row.created_at,
    )


def opportunity_document(row: OpportunitySource) -> SourceDocument:
    stage = _value(row.stage)
    lines = [f"Opportunity: {row.name}", f"Stage: {_label(stage)}"]
    if row.amount is not None:
        lines.append(f"Amount: {row.amount}")
    if row.expected_close_date is not None:
        lines.append(f"Expected close: {row.expected_close_date.isoformat()}")
    if row.description and row.description.strip():
        lines.append(row.description.strip())
    return SourceDocument(
        source_type=DocumentSourceType.OPPORTUNITY,
        source_id=row.id,
        account_id=row.account_id,
        entity_type=DocumentEntityType.OPPORTUNITY,
        entity_id=row.id,
        text="\n".join(lines),
        occurred_at=row.updated_at,
        metadata={"stage": stage},
    )


def account_document(row: AccountSource) -> SourceDocument:
    lines = [f"Account: {row.name}"]
    if row.industry:
        lines.append(f"Industry: {row.industry}")
    if row.employee_count is not None:
        lines.append(f"Employees: {row.employee_count}")
    if row.website:
        lines.append(f"Website: {row.website}")
    return SourceDocument(
        source_type=DocumentSourceType.ACCOUNT,
        source_id=row.id,
        account_id=row.id,
        entity_type=DocumentEntityType.ACCOUNT,
        entity_id=row.id,
        text="\n".join(lines),
        occurred_at=row.updated_at,
    )
