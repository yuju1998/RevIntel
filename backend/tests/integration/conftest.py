import os

import pytest

from app.core.config import Settings

# Defaults match docker-compose.yml; override with env vars to target other infrastructure.
DEFAULT_TEST_DATABASE_URL = "postgresql+psycopg://revintel:revintel@localhost:5432/revintel_test"
DEFAULT_TEST_REDIS_URL = "redis://localhost:6379/15"


@pytest.fixture
def integration_settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL),
        redis_url=os.environ.get("TEST_REDIS_URL", DEFAULT_TEST_REDIS_URL),
    )
