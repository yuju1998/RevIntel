"""Contact data access. Repositories never commit or roll back; services own transactions."""

import uuid

from sqlalchemy.orm import Session

from app.models.contact import Contact


class ContactRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, contact: Contact) -> Contact:
        self._session.add(contact)
        self._session.flush()
        return contact

    def get(self, contact_id: uuid.UUID) -> Contact | None:
        return self._session.get(Contact, contact_id)
