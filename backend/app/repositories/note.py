"""Note data access. Repositories never commit or roll back; services own transactions."""

from sqlalchemy.orm import Session

from app.models.note import Note


class NoteRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, note: Note) -> Note:
        self._session.add(note)
        self._session.flush()
        return note
