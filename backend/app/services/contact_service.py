"""Contact business logic."""

from sqlalchemy.orm import Session

from app.models.contact import Contact
from app.repositories.account import AccountRepository
from app.repositories.contact import ContactRepository
from app.schemas.contact import ContactCreate
from app.services.crm_references import not_found, not_found_for
from app.services.transaction import write_transaction

CONSTRAINT_ERRORS = {"fk_contacts_account_id_accounts": not_found_for("account_id", "account")}


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
                raise not_found("account_id", "account")
            contact = self._contacts.add(Contact(**data.model_dump()))
        return contact
