from collections.abc import Iterator

import pytest

from app.core.config import get_database_settings, get_embedding_settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    get_database_settings.cache_clear()
    get_embedding_settings.cache_clear()
    yield
    get_settings.cache_clear()
    get_database_settings.cache_clear()
    get_embedding_settings.cache_clear()
