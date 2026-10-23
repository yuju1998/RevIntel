from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import RelatedResourceNotFoundError
from app.services.transaction import violated_constraint, write_transaction
from tests.unit.helpers import integrity_error


@pytest.fixture
def session() -> MagicMock:
    return MagicMock(spec=Session)


def related_error() -> RelatedResourceNotFoundError:
    return RelatedResourceNotFoundError("account_id does not refer to an existing account")


def test_commits_on_success(session: MagicMock) -> None:
    with write_transaction(session):
        pass

    session.commit.assert_called_once()
    session.rollback.assert_not_called()


def test_rolls_back_and_reraises_domain_errors(session: MagicMock) -> None:
    with pytest.raises(RelatedResourceNotFoundError), write_transaction(session):
        raise related_error()

    session.rollback.assert_called_once()
    session.commit.assert_not_called()


def test_maps_known_constraint_violation(session: MagicMock) -> None:
    session.commit.side_effect = integrity_error("fk_contacts_account_id_accounts")

    with (
        pytest.raises(RelatedResourceNotFoundError),
        write_transaction(session, {"fk_contacts_account_id_accounts": related_error}),
    ):
        pass

    session.rollback.assert_called_once()


@pytest.mark.parametrize("constraint", ["uq_users_email", None])
def test_reraises_unknown_integrity_errors_unchanged(
    session: MagicMock, constraint: str | None
) -> None:
    error = integrity_error(constraint)
    session.commit.side_effect = error

    with (
        pytest.raises(IntegrityError) as exc_info,
        write_transaction(session, {"fk_contacts_account_id_accounts": related_error}),
    ):
        pass

    assert exc_info.value is error
    session.rollback.assert_called_once()


def test_violated_constraint_handles_errors_without_diagnostics() -> None:
    assert violated_constraint(IntegrityError("x", {}, Exception())) is None
