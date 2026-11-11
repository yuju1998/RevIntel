"""FastAPI application factory.

Importing this module reads no settings and opens no connections. Settings are
resolved and infrastructure clients are created in the lifespan, at startup.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.redis import create_redis_client
from app.db.session import create_db_engine, create_session_factory
from app.ml_inference.champion import ChampionModelProvider


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_settings = settings if settings is not None else get_settings()
        configure_logging(
            environment=resolved_settings.environment, log_level=resolved_settings.log_level
        )
        engine = create_db_engine(resolved_settings)
        redis_client = create_redis_client(resolved_settings)
        # Separate short-timeout client for the cache; /ready keeps the default client.
        cache_redis_client = create_redis_client(
            resolved_settings, timeout_seconds=resolved_settings.cache_redis_timeout_seconds
        )

        app.state.settings = resolved_settings
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)
        app.state.redis = redis_client
        app.state.cache_redis = cache_redis_client
        # No MLflow call here: the champion model is resolved and loaded on first use.
        app.state.champion_models = ChampionModelProvider(
            resolved_settings.mlflow_tracking_uri, resolved_settings.mlflow_model_name
        )
        try:
            yield
        finally:
            cache_redis_client.close()
            redis_client.close()
            engine.dispose()

    app = FastAPI(title="RevIntel API", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(app)
    # Request IDs, access log and safe 500s (see app.api.middleware for its position).
    app.add_middleware(RequestContextMiddleware)
    app.include_router(api_router)
    return app


app = create_app()
