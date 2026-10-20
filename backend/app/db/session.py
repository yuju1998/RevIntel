"""SQLAlchemy engine and session factory construction.

Nothing here runs at import time: the application lifespan builds these objects
and stores them on `app.state`.
"""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import DatabaseSettings


def create_db_engine(settings: DatabaseSettings) -> Engine:
    # create_engine is lazy: no connection is opened until the first checkout.
    return create_engine(
        str(settings.database_url),
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_seconds,
        connect_args={
            "connect_timeout": settings.db_connect_timeout_seconds,
            # Sessions report and interpret timestamps in UTC regardless of server config.
            "options": "-c timezone=UTC",
        },
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
