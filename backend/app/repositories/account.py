"""Account data access. Repositories never commit or roll back; services own transactions."""

import uuid
from collections.abc import Sequence

from sqlalchemy import ColumnElement, func, select
from sqlalchemy.orm import Session

from app.models.account import Account


class AccountRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, account: Account) -> Account:
        self._session.add(account)
        self._session.flush()
        return account

    def get(self, account_id: uuid.UUID) -> Account | None:
        return self._session.get(Account, account_id)

    def find(self, *, industry: str | None, limit: int, offset: int) -> Sequence[Account]:
        stmt = (
            select(Account)
            .where(*self._filters(industry=industry))
            .order_by(Account.created_at.desc(), Account.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return self._session.scalars(stmt).all()

    def count(self, *, industry: str | None) -> int:
        stmt = select(func.count()).select_from(Account).where(*self._filters(industry=industry))
        return self._session.scalar(stmt) or 0

    @staticmethod
    def _filters(*, industry: str | None) -> list[ColumnElement[bool]]:
        filters: list[ColumnElement[bool]] = []
        if industry is not None:
            filters.append(Account.industry == industry)
        return filters
