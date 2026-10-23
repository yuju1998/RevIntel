"""Account/contact activities that fill the remaining activity budget exactly.

Each references an account and a contact OF THAT ACCOUNT (the cross-account rule), and
happens after the contact was created.
"""

from dataclasses import dataclass

import numpy as np

from app.models.enums import ActivityDirection, ActivityType
from ml.synthetic import config as c
from ml.synthetic.entities import Accounts, Contacts


@dataclass(frozen=True)
class AccountActivityDraft:
    account_index: np.ndarray
    contact_index: np.ndarray
    occurred_offsets: np.ndarray  # seconds relative to as_of (negative)
    types: list[ActivityType]
    directions: list[ActivityDirection]


def draft_account_activities(
    rng: np.random.Generator, count: int, accounts: Accounts, contacts: Contacts
) -> AccountActivityDraft:
    if count and not contacts.rows:
        raise ValueError("account activities need at least one account with contacts")
    if count == 0:
        empty = np.array([], dtype=np.int64)
        return AccountActivityDraft(empty, empty, empty, [], [])

    # Busier (larger) accounts get more activity; contacts are picked within the account.
    weights = np.log1p(accounts.employee_counts.astype(float))
    account_index = rng.choice(weights.size, size=count, p=weights / weights.sum())
    order = np.argsort(contacts.account_index, kind="stable")
    first = np.searchsorted(contacts.account_index[order], np.arange(weights.size))
    per_account = np.bincount(contacts.account_index, minlength=weights.size)
    contact_index = order[first[account_index] + rng.integers(0, per_account[account_index])]

    contact_created = contacts.created_offsets[contact_index]
    occurred = contact_created + np.floor(rng.random(count) * (-contact_created)).astype(np.int64)

    type_values = [t for t, _ in c.ACCOUNT_ACTIVITY_TYPE_WEIGHTS]
    type_weights = np.array([w for _, w in c.ACCOUNT_ACTIVITY_TYPE_WEIGHTS])
    direction_values = [d for d, _ in c.ACCOUNT_ACTIVITY_DIRECTION_WEIGHTS]
    direction_weights = np.array([w for _, w in c.ACCOUNT_ACTIVITY_DIRECTION_WEIGHTS])
    types = rng.choice(len(type_values), size=count, p=type_weights / type_weights.sum())
    directions = rng.choice(
        len(direction_values), size=count, p=direction_weights / direction_weights.sum()
    )
    return AccountActivityDraft(
        account_index=account_index,
        contact_index=contact_index,
        occurred_offsets=occurred,
        types=[type_values[t] for t in types],
        directions=[direction_values[d] for d in directions],
    )
