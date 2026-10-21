import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.schemas.activity import ActivityCreate
from app.schemas.note import NoteCreate


def _activity(**fields: object) -> ActivityCreate:
    values: dict[str, object] = {
        "account_id": uuid.uuid4(),
        "activity_type": "CALL",
        "direction": "OUTBOUND",
        "occurred_at": datetime.now(UTC) - timedelta(hours=1),
    }
    return ActivityCreate(**{**values, **fields})


@pytest.mark.parametrize("parent", ["account_id", "lead_id", "opportunity_id", "contact_id"])
def test_activity_accepts_any_single_parent(parent: str) -> None:
    activity = _activity(**{"account_id": None, parent: uuid.uuid4()})

    assert getattr(activity, parent) is not None


def test_activity_accepts_several_parents() -> None:
    _activity(lead_id=uuid.uuid4(), opportunity_id=uuid.uuid4(), contact_id=uuid.uuid4())


def test_activity_requires_at_least_one_parent() -> None:
    with pytest.raises(ValidationError, match="at least one of"):
        _activity(account_id=None)


def test_activity_rejects_naive_occurred_at() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        _activity(occurred_at=datetime(2026, 1, 1, 12, 0))


def test_activity_accepts_any_utc_offset() -> None:
    occurred = datetime(2026, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=-5)))

    assert _activity(occurred_at=occurred).occurred_at == occurred


def test_activity_allows_small_clock_skew_into_the_future() -> None:
    _activity(occurred_at=datetime.now(UTC) + timedelta(minutes=4))


def test_activity_rejects_occurred_at_more_than_5_minutes_ahead() -> None:
    with pytest.raises(ValidationError, match="5 minutes in the future"):
        _activity(occurred_at=datetime.now(UTC) + timedelta(minutes=6))


@pytest.mark.parametrize(
    "fields",
    [
        {"activity_type": "LUNCH"},
        {"direction": "SIDEWAYS"},
        {"subject": "x" * 256},
        {"subject": "   "},
        {"body": "   \n\t"},
        {"account_id": "not-a-uuid"},
        {"created_by_id": "not-a-uuid"},
        {"updated_at": "2026-01-01T00:00:00Z"},
    ],
)
def test_activity_rejects_invalid_input(fields: dict) -> None:
    with pytest.raises(ValidationError):
        _activity(**fields)


def test_activity_body_is_optional_and_keeps_formatting() -> None:
    assert _activity().body is None
    body = "  Agenda:\n  - pricing\n  - timeline\n"

    assert _activity(body=body).body == body


def test_activity_subject_is_trimmed() -> None:
    assert _activity(subject="  QBR  ").subject == "QBR"


@pytest.mark.parametrize("parent", ["account_id", "lead_id", "opportunity_id"])
def test_note_accepts_any_single_parent(parent: str) -> None:
    NoteCreate(body="Hello", **{parent: uuid.uuid4()})


def test_note_requires_at_least_one_parent() -> None:
    with pytest.raises(ValidationError, match="at least one of"):
        NoteCreate(body="Hello")


def test_note_has_no_contact_parent() -> None:
    with pytest.raises(ValidationError):
        NoteCreate(body="Hello", contact_id=uuid.uuid4())


@pytest.mark.parametrize("body", ["", "   ", "\n\t "])
def test_note_body_must_contain_non_whitespace(body: str) -> None:
    with pytest.raises(ValidationError):
        NoteCreate(body=body, account_id=uuid.uuid4())


def test_note_body_is_required() -> None:
    with pytest.raises(ValidationError):
        NoteCreate(account_id=uuid.uuid4())  # type: ignore[call-arg]


def test_note_body_keeps_original_formatting() -> None:
    body = "\n  Indented first line\n\nSecond paragraph  \n"

    assert NoteCreate(body=body, account_id=uuid.uuid4()).body == body
