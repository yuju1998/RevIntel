"""FastAPI application factory.

Importing this module reads no settings and opens no connections. Settings are
resolved and infrastructure clients are created in the lifespan, at startup.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.redis import create_redis_client
from app.db.session import create_db_engine, create_session_factory


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_settings = settings if settings is not None else get_settings()
        engine = create_db_engine(resolved_settings)
        redis_client = create_redis_client(resolved_settings)

        app.state.settings = resolved_settings
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)
        app.state.redis = redis_client
        try:
            yield
        finally:
            redis_client.close()
            engine.dispose()

    app = FastAPI(title="RevIntel API", version="0.1.0", lifespan=lifespan)
    app.include_router(api_router)
    return app


app = create_app()
