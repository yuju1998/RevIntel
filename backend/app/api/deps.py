"""FastAPI dependencies that expose infrastructure resources held on `app.state`."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from redis import Redis
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


def get_db(request: Request) -> Iterator[Session]:
    """Provide one Session per request.

    Successful requests are not committed here; service code owns transactions.
    Closing a session with an open transaction discards uncommitted work.
    """
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    session = session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


EngineDep = Annotated[Engine, Depends(get_engine)]
RedisDep = Annotated[Redis, Depends(get_redis)]
SessionDep = Annotated[Session, Depends(get_db)]
