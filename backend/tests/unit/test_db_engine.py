from unittest.mock import MagicMock

import pytest

from app.core.config import DatabaseSettings
from app.db import session as session_module
from tests.unit.helpers import UNREACHABLE_DATABASE_URL


def test_engine_pins_utc_and_keeps_connect_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    create_engine = MagicMock()
    monkeypatch.setattr(session_module, "create_engine", create_engine)

    session_module.create_db_engine(
        DatabaseSettings(
            _env_file=None, database_url=UNREACHABLE_DATABASE_URL, db_connect_timeout_seconds=7
        )
    )

    kwargs = create_engine.call_args.kwargs
    assert kwargs["connect_args"] == {"connect_timeout": 7, "options": "-c timezone=UTC"}
    # Bound values (e.g. password hashes) never appear in SQLAlchemy error messages.
    assert kwargs["hide_parameters"] is True
