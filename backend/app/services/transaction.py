"""Transaction boundary for service write operations."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import DomainError

# Maps a constraint name to the domain error raised when that constraint is violated.
ConstraintErrors = Mapping[str, Callable[[], DomainError]]


@contextmanager
def write_transaction(
    session: Session, constraint_errors: ConstraintErrors | None = None
) -> Iterator[None]:
    """Commit the block's work, or roll it back on any error.

    Services validate related records up front. An IntegrityError can still occur if a
    related row disappears between validation and commit; when the violated constraint
    is listed in `constraint_errors` it becomes that domain error. Any other
    IntegrityError is re-raised unchanged rather than mislabeled.
    """
    try:
        yield
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        error_factory = (constraint_errors or {}).get(violated_constraint(exc) or "")
        if error_factory is None:
            raise
        raise error_factory() from exc
    except Exception:
        session.rollback()
        raise


def violated_constraint(exc: IntegrityError) -> str | None:
    """Name of the violated constraint as reported by PostgreSQL (psycopg diagnostics)."""
    diagnostics = getattr(exc.orig, "diag", None)
    return getattr(diagnostics, "constraint_name", None)
