"""Alembic migration environment for RevIntel.

The database URL comes from DatabaseSettings (environment variables or `.env`), so
migrations need no Redis or other application settings. Autogenerate compares the
database against `Base.metadata`.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

import app.models  # noqa: F401  # registers every model on Base.metadata
from app.core.config import get_database_settings
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    # Keep application loggers enabled when migrations run in-process (e.g. from tests).
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    # Passed straight to SQLAlchemy rather than through config.set_main_option(), whose
    # configparser interpolation breaks on `%` in URL-encoded passwords.
    return str(get_database_settings().database_url)


def run_migrations_offline() -> None:
    """Emit SQL to stdout (`--sql`) without connecting to the database."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the database with a single, unpooled connection."""
    settings = get_database_settings()
    engine = create_engine(
        _database_url(),
        poolclass=pool.NullPool,
        connect_args={"connect_timeout": settings.db_connect_timeout_seconds},
    )
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)

            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
