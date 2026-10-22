"""CLI argument/password handling with a fake session (no database)."""

import io
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from app.cli import create_user
from app.core.exceptions import BusinessRuleError
from app.models import User
from app.models.enums import UserRole
from app.schemas.user import UserCreate

PASSWORD = "correct horse battery staple"
ARGS = ["--email", "Ada@Example.com", "--full-name", "Ada Admin", "--role", "ADMIN"]


@pytest.fixture
def created(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[UserCreate]]:
    calls: list[UserCreate] = []

    class FakeUserService:
        def __init__(self, session: object, users: object) -> None:
            pass

        def create_user(self, data: UserCreate) -> User:
            calls.append(data)
            return User(email=data.email.lower(), role=UserRole(data.role))

    monkeypatch.setattr(create_user, "UserService", FakeUserService)
    yield calls


def _run(argv: list[str], *, prompt_answers: list[str] | None = None, stdin: str = "") -> int:
    answers = iter(prompt_answers or [])
    return create_user.main(
        argv,
        stdin=io.StringIO(stdin),
        prompt=lambda _: next(answers),
        session_factory=MagicMock(),
    )


def test_prompts_for_password_twice(
    created: list[UserCreate], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(ARGS, prompt_answers=[PASSWORD, PASSWORD]) == 0

    assert created[0].password.get_secret_value() == PASSWORD
    output = capsys.readouterr()
    assert "Created ADMIN user ada@example.com" in output.out
    assert PASSWORD not in output.out + output.err


def test_mismatched_passwords_are_rejected(created: list[UserCreate]) -> None:
    assert _run(ARGS, prompt_answers=[PASSWORD, PASSWORD + "!"]) == 2
    assert created == []


def test_password_from_stdin(created: list[UserCreate]) -> None:
    assert _run([*ARGS, "--password-stdin"], stdin=PASSWORD + "\n") == 0
    assert created[0].password.get_secret_value() == PASSWORD


def test_invalid_input_reports_fields_without_echoing_password(
    created: list[UserCreate], capsys: pytest.CaptureFixture[str]
) -> None:
    short = "tiny-pass"

    assert _run(["--email", "bad", "--full-name", "A", "--role", "ADMIN", "--password-stdin"],
                stdin=short) == 2

    err = capsys.readouterr().err
    assert "email" in err and "password" in err
    assert short not in err
    assert created == []


def test_duplicate_email_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class DuplicateUserService:
        def __init__(self, session: object, users: object) -> None:
            pass

        def create_user(self, data: UserCreate) -> User:
            raise BusinessRuleError("A user with this email already exists")

    monkeypatch.setattr(create_user, "UserService", DuplicateUserService)

    assert _run(ARGS, prompt_answers=[PASSWORD, PASSWORD]) == 1
    assert "already exists" in capsys.readouterr().err


def test_role_must_be_known() -> None:
    with pytest.raises(SystemExit):
        _run(["--email", "a@b.co", "--full-name", "A", "--role", "ROOT"])
