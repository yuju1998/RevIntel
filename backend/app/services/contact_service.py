"""Contact business logic."""

from sqlalchemy.orm import Session

from app.core.exceptions import RelatedResourceNotFoundError
from app.models.contact import Contact
from app.repositories.account import AccountRepository
from app.repositories.contact import ContactRepository
from app.schemas.contact import ContactCreate
from app.services.transaction import write_transaction


def _account_not_found() -> RelatedResourceNotFoundError:
    return RelatedResourceNotFoundError("account_id does not refer to an existing account")


CONSTRAINT_ERRORS = {"fk_contacts_account_id_accounts": _account_not_found}


class ContactService:
    def __init__(
        self, session: Session, contacts: ContactRepository, accounts: AccountRepository
    ) -> None:
        self._session = session
        self._contacts = contacts
        self._accounts = accounts

    def create_contact(self, data: ContactCreate) -> Contact:
        with write_transaction(self._session, CONSTRAINT_ERRORS):
            if self._accounts.get(data.account_id) is None:
                raise _account_not_found()
            contact = self._contacts.add(Contact(**data.model_dump()))
        return contact
