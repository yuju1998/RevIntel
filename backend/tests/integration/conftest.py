import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Engine, make_url, text
from sqlalchemy.orm import Session

from app.core.config import DatabaseSettings, Settings, get_database_settings
from app.db.base import Base
from app.db.session import create_db_engine
from app.main import create_app

# Defaults match docker-compose.yml; override with env vars to target other infrastructure.
DEFAULT_TEST_DATABASE_URL = "postgresql+psycopg://revintel:revintel@localhost:5432/revintel_test"
DEFAULT_TEST_REDIS_URL = "redis://localhost:6379/15"

ALEMBIC_INI = Path(__file__).resolve().parents[3] / "alembic.ini"


def get_test_database_url() -> str:
    return os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL)


def require_test_database(url: str) -> None:
    """Hard guard: destructive setup only ever runs against a database named `*_test`."""
    database = make_url(url).database or ""
    if not database.endswith("_test"):
        pytest.fail(
            f"Refusing to modify database {database!r}: integration tests only run "
            "destructive setup against databases whose name ends with '_test'.",
            pytrace=False,
        )


@pytest.fixture
def integration_settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=get_test_database_url(),
        redis_url=os.environ.get("TEST_REDIS_URL", DEFAULT_TEST_REDIS_URL),
    )


@pytest.fixture(scope="session")
def migrated_database() -> str:
    """Rebuild the test schema from the real Alembic migrations once per session.

    Never uses Base.metadata.create_all(): downgrade to base, upgrade to head, then
    `alembic check` to prove the models and migrations agree.
    """
    url = get_test_database_url()
    # Must run before any downgrade, schema change or TRUNCATE.
    require_test_database(url)

    with pytest.MonkeyPatch.context() as env:
        # env.py reads DatabaseSettings; environment variables take precedence over `.env`.
        env.setenv("DATABASE_URL", url)
        get_database_settings.cache_clear()
        config = Config(str(ALEMBIC_INI))
        command.downgrade(config, "base")
        command.upgrade(config, "head")
        command.check(config)
    get_database_settings.cache_clear()
    return url


@pytest.fixture
def db_engine(migrated_database: str) -> Iterator[Engine]:
    engine = create_db_engine(DatabaseSettings(_env_file=None, database_url=migrated_database))
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def clean_tables(migrated_database: str, db_engine: Engine) -> Iterator[None]:
    """Empty every model table after the test. alembic_version is left to Alembic."""
    yield
    require_test_database(migrated_database)
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    with db_engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {tables} CASCADE"))


@pytest.fixture
def db_session(db_engine: Engine, clean_tables: None) -> Iterator[Session]:
    with Session(db_engine, expire_on_commit=False) as session:
        yield session


@pytest.fixture
def api_client(
    integration_settings: Settings, migrated_database: str, clean_tables: None
) -> Iterator[TestClient]:
    with TestClient(create_app(integration_settings)) as client:
        yield client
