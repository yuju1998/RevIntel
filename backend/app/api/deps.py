"""FastAPI dependencies: infrastructure resources held on `app.state`, and services."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from redis import Redis
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.repositories.account import AccountRepository
from app.repositories.contact import ContactRepository
from app.repositories.lead import LeadRepository
from app.repositories.opportunity import OpportunityRepository
from app.repositories.user import UserRepository
from app.services.account_service import AccountService
from app.services.contact_service import ContactService
from app.services.lead_service import LeadService
from app.services.opportunity_service import OpportunityService


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


def get_account_service(session: SessionDep) -> AccountService:
    return AccountService(session, AccountRepository(session), UserRepository(session))


def get_contact_service(session: SessionDep) -> ContactService:
    return ContactService(session, ContactRepository(session), AccountRepository(session))


def get_lead_service(session: SessionDep) -> LeadService:
    return LeadService(session, LeadRepository(session), UserRepository(session))


def get_opportunity_service(session: SessionDep) -> OpportunityService:
    return OpportunityService(
        session,
        OpportunityRepository(session),
        AccountRepository(session),
        UserRepository(session),
    )


AccountServiceDep = Annotated[AccountService, Depends(get_account_service)]
ContactServiceDep = Annotated[ContactService, Depends(get_contact_service)]
LeadServiceDep = Annotated[LeadService, Depends(get_lead_service)]
OpportunityServiceDep = Annotated[OpportunityService, Depends(get_opportunity_service)]
