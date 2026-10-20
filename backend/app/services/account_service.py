"""Account business logic."""

import uuid

from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.models.account import Account
from app.repositories.account import AccountRepository
from app.repositories.user import UserRepository
from app.schemas.account import AccountCreate
from app.services.ownership import ensure_active_owner, owner_not_found
from app.services.pagination import PageResult
from app.services.transaction import write_transaction

CONSTRAINT_ERRORS = {"fk_accounts_owner_id_users": owner_not_found}


class AccountService:
    def __init__(
        self, session: Session, accounts: AccountRepository, users: UserRepository
    ) -> None:
        self._session = session
        self._accounts = accounts
        self._users = users

    def create_account(self, data: AccountCreate) -> Account:
        with write_transaction(self._session, CONSTRAINT_ERRORS):
            ensure_active_owner(self._users, data.owner_id)
            account = self._accounts.add(Account(**data.model_dump()))
        return account

    def get_account(self, account_id: uuid.UUID) -> Account:
        account = self._accounts.get(account_id)
        if account is None:
            raise NotFoundError(f"Account {account_id} not found")
        return account

    def list_accounts(
        self, *, industry: str | None, limit: int, offset: int
    ) -> PageResult[Account]:
        items = self._accounts.find(industry=industry, limit=limit, offset=offset)
        total = self._accounts.count(industry=industry)
        return PageResult(items=items, total=total, limit=limit, offset=offset)
