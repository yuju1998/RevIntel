"""Validation of the CRM parents an activity or note refers to.

A SQL CHECK cannot compare rows across tables, so the service layer enforces that all
account-scoped references agree on one account (docs/DATABASE.md, service rule 3), and
derives the canonical account_id from them. Leads have no account in V1: lead_id is
checked for existence only and never implies an account.
"""

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

from app.core.exceptions import BusinessRuleError, RelatedResourceNotFoundError
from app.repositories.account import AccountRepository
from app.repositories.contact import ContactRepository
from app.repositories.lead import LeadRepository
from app.repositories.opportunity import OpportunityRepository


@dataclass(frozen=True, slots=True)
class CrmRepositories:
    accounts: AccountRepository
    leads: LeadRepository
    opportunities: OpportunityRepository
    contacts: ContactRepository


def resolve_account_id(
    repos: CrmRepositories,
    *,
    account_id: uuid.UUID | None,
    lead_id: uuid.UUID | None,
    opportunity_id: uuid.UUID | None,
    contact_id: uuid.UUID | None = None,
) -> uuid.UUID | None:
    """Check that every reference exists and that they agree on one account.

    Returns the canonical account_id: the explicit one, or the account derived from the
    opportunity/contact. Returns None for lead-only records.
    """
    # (field, account the field implies) for each account-scoped reference.
    implied: list[tuple[str, uuid.UUID]] = []

    if account_id is not None:
        if repos.accounts.get(account_id) is None:
            raise not_found("account_id", "account")
        implied.append(("account_id", account_id))
    if lead_id is not None and repos.leads.get(lead_id) is None:
        raise not_found("lead_id", "lead")
    if opportunity_id is not None:
        opportunity = repos.opportunities.get(opportunity_id)
        if opportunity is None:
            raise not_found("opportunity_id", "opportunity")
        implied.append(("opportunity_id", opportunity.account_id))
    if contact_id is not None:
        contact = repos.contacts.get(contact_id)
        if contact is None:
            raise not_found("contact_id", "contact")
        implied.append(("contact_id", contact.account_id))

    accounts = {account for _, account in implied}
    if len(accounts) > 1:
        fields = ", ".join(field for field, _ in implied)
        raise BusinessRuleError(f"{fields} refer to different accounts")
    return accounts.pop() if accounts else None


def not_found(field: str, entity: str) -> RelatedResourceNotFoundError:
    return RelatedResourceNotFoundError(f"{field} does not refer to an existing {entity}")


def not_found_for(field: str, entity: str) -> Callable[[], RelatedResourceNotFoundError]:
    """Error factory for a foreign-key violation, for `write_transaction` mappings."""
    return partial(not_found, field, entity)


def parent_required(fields: Sequence[str]) -> BusinessRuleError:
    return BusinessRuleError(f"at least one of {', '.join(fields)} is required")


def parent_required_for(fields: Sequence[str]) -> Callable[[], BusinessRuleError]:
    """Error factory for an at-least-one-parent CHECK, for `write_transaction` mappings."""
    return partial(parent_required, tuple(fields))
