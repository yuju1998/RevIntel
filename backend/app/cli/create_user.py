"""Create a RevIntel user (there is no signup API).

Usage, from the repository root:

    PYTHONPATH=backend python -m app.cli.create_user \\
        --email admin@example.com --full-name "Ada Admin" --role ADMIN

The password is prompted for twice (not echoed). For automation, pass
--password-stdin and pipe the password in; never put it on the command line.

Uses the same UserService as the application (email normalization, Argon2 hashing,
duplicate handling) and only needs DATABASE_URL.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from getpass import getpass
from typing import TextIO

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import get_database_settings
from app.core.exceptions import DomainError
from app.db.session import create_db_engine, create_session_factory
from app.models.enums import UserRole
from app.repositories.user import UserRepository
from app.schemas.user import UserCreate
from app.services.user_service import UserService

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INVALID_INPUT = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="create_user", description="Create a RevIntel user.")
    parser.add_argument("--email", required=True)
    parser.add_argument("--full-name", required=True)
    parser.add_argument("--role", required=True, choices=[role.value for role in UserRole])
    parser.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read the password from the first line of stdin instead of prompting.",
    )
    return parser


def _read_password(
    args: argparse.Namespace, stdin: TextIO, prompt: Callable[[str], str]
) -> str | None:
    if args.password_stdin:
        return stdin.readline().rstrip("\r\n")
    password = prompt("Password: ")
    if prompt("Repeat password: ") != password:
        return None
    return password


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO = sys.stdin,
    prompt: Callable[[str], str] = getpass,
    session_factory: Callable[[], Session] | None = None,
) -> int:
    args = _parser().parse_args(argv)

    password = _read_password(args, stdin, prompt)
    if password is None:
        print("Error: passwords do not match.", file=sys.stderr)
        return EXIT_INVALID_INPUT

    try:
        data = UserCreate(
            email=args.email, full_name=args.full_name, role=args.role, password=password
        )
    except ValidationError as exc:
        # Field and message only: never echo the input (it may be the password).
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            print(f"Error: {field}: {error['msg']}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    engine = None
    if session_factory is None:
        engine = create_db_engine(get_database_settings())
        session_factory = create_session_factory(engine)
    try:
        with session_factory() as session:
            user = UserService(session, UserRepository(session)).create_user(data)
            print(f"Created {user.role} user {user.email} ({user.id}).")
    except DomainError as exc:
        print(f"Error: {exc.message}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if engine is not None:
            engine.dispose()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
