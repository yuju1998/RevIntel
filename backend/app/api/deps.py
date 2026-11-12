"""FastAPI dependencies: infrastructure resources held on `app.state`, and services."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from redis import Redis
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.ai.embeddings import EmbeddingProvider
from app.core.cache import Cache, NullCache, RedisCache
from app.core.config import Settings
from app.ml_inference.champion import ChampionModelProvider
from app.repositories.account import AccountRepository
from app.repositories.activity import ActivityRepository
from app.repositories.contact import ContactRepository
from app.repositories.lead import LeadRepository
from app.repositories.lead_features import LeadFeatureRepository
from app.repositories.lead_prediction import LeadPredictionRepository
from app.repositories.note import NoteRepository
from app.repositories.opportunity import OpportunityRepository
from app.repositories.semantic_search import SemanticSearchRepository
from app.repositories.timeline import TimelineRepository
from app.repositories.user import UserRepository
from app.services.account_cache import AccountCache
from app.services.account_service import AccountService
from app.services.activity_service import ActivityService
from app.services.auth_service import AuthService
from app.services.contact_service import ContactService
from app.services.crm_references import CrmRepositories
from app.services.lead_scoring_service import LeadScoringService
from app.services.lead_service import LeadService
from app.services.note_service import NoteService
from app.services.opportunity_service import OpportunityService
from app.services.semantic_search_service import SemanticSearchService
from app.services.timeline_service import TimelineService


def get_app_settings(request: Request) -> Settings:
    """Settings resolved at startup (the lifespan stores them on app.state)."""
    return request.app.state.settings


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


def get_cache_redis(request: Request) -> Redis:
    """Separate short-timeout client used only by the cache."""
    return request.app.state.cache_redis


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


def get_champion_models(request: Request) -> ChampionModelProvider:
    """Process-wide champion model cache (created at startup, loaded on first use)."""
    return request.app.state.champion_models


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
EngineDep = Annotated[Engine, Depends(get_engine)]
RedisDep = Annotated[Redis, Depends(get_redis)]
CacheRedisDep = Annotated[Redis, Depends(get_cache_redis)]
SessionDep = Annotated[Session, Depends(get_db)]


def get_account_cache(settings: SettingsDep, cache_redis: CacheRedisDep) -> AccountCache:
    ttl_seconds = settings.account_cache_ttl_seconds
    cache: Cache = RedisCache(cache_redis) if ttl_seconds > 0 else NullCache()
    return AccountCache(cache, ttl_seconds=ttl_seconds)


def get_auth_service(session: SessionDep, settings: SettingsDep) -> AuthService:
    return AuthService(
        session,
        UserRepository(session),
        jwt_secret=settings.jwt_secret,
        access_token_expire_minutes=settings.access_token_expire_minutes,
    )


def get_account_service(
    session: SessionDep, cache: Annotated[AccountCache, Depends(get_account_cache)]
) -> AccountService:
    return AccountService(
        session, AccountRepository(session), UserRepository(session), cache=cache
    )


def get_contact_service(session: SessionDep) -> ContactService:
    return ContactService(session, ContactRepository(session), AccountRepository(session))


def get_lead_service(session: SessionDep) -> LeadService:
    return LeadService(session, LeadRepository(session), UserRepository(session))


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    """Process-wide embedding model (created at startup, loaded on first use)."""
    return request.app.state.embedding_provider


def get_semantic_search_service(
    session: SessionDep,
    embeddings: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
) -> SemanticSearchService:
    return SemanticSearchService(
        AccountRepository(session), SemanticSearchRepository(session), embeddings
    )


def get_lead_scoring_service(
    session: SessionDep,
    models: Annotated[ChampionModelProvider, Depends(get_champion_models)],
) -> LeadScoringService:
    return LeadScoringService(
        session,
        LeadRepository(session),
        LeadPredictionRepository(session),
        LeadFeatureRepository(session),
        models,
    )


def get_opportunity_service(session: SessionDep) -> OpportunityService:
    return OpportunityService(
        session,
        OpportunityRepository(session),
        AccountRepository(session),
        UserRepository(session),
    )


def _crm_repositories(session: Session) -> CrmRepositories:
    return CrmRepositories(
        accounts=AccountRepository(session),
        leads=LeadRepository(session),
        opportunities=OpportunityRepository(session),
        contacts=ContactRepository(session),
    )


def get_activity_service(session: SessionDep) -> ActivityService:
    return ActivityService(session, ActivityRepository(session), _crm_repositories(session))


def get_note_service(session: SessionDep) -> NoteService:
    return NoteService(session, NoteRepository(session), _crm_repositories(session))


def get_timeline_service(session: SessionDep) -> TimelineService:
    return TimelineService(AccountRepository(session), TimelineRepository(session))


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
AccountServiceDep = Annotated[AccountService, Depends(get_account_service)]
ContactServiceDep = Annotated[ContactService, Depends(get_contact_service)]
LeadServiceDep = Annotated[LeadService, Depends(get_lead_service)]
LeadScoringServiceDep = Annotated[LeadScoringService, Depends(get_lead_scoring_service)]
SemanticSearchServiceDep = Annotated[
    SemanticSearchService, Depends(get_semantic_search_service)
]
OpportunityServiceDep = Annotated[OpportunityService, Depends(get_opportunity_service)]
ActivityServiceDep = Annotated[ActivityService, Depends(get_activity_service)]
NoteServiceDep = Annotated[NoteService, Depends(get_note_service)]
TimelineServiceDep = Annotated[TimelineService, Depends(get_timeline_service)]
