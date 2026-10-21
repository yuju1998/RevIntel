"""Account timeline: activities and notes linked to an account, newest first."""

import uuid

from app.core.exceptions import NotFoundError
from app.models.activity import Activity
from app.models.note import Note
from app.repositories.account import AccountRepository
from app.repositories.timeline import TimelineRepository
from app.services.pagination import PageResult


class TimelineService:
    def __init__(self, accounts: AccountRepository, timeline: TimelineRepository) -> None:
        self._accounts = accounts
        self._timeline = timeline

    def get_account_timeline(
        self, account_id: uuid.UUID, *, limit: int, offset: int
    ) -> PageResult[Activity | Note]:
        """Records whose canonical account_id is this account (lead-only rows never appear)."""
        if self._accounts.get(account_id) is None:
            raise NotFoundError(f"Account {account_id} not found")
        items = self._timeline.find_account_timeline(account_id, limit=limit, offset=offset)
        total = self._timeline.count_account_timeline(account_id)
        return PageResult(items=items, total=total, limit=limit, offset=offset)
