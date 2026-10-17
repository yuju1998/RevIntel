from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from pydantic import ValidationError

from app.core.config import get_database_settings, get_settings
from tests.unit.conftest import UNREACHABLE_DATABASE_URL

ALEMBIC_INI = Path(__file__).resolve().parents[3] / "alembic.ini"

pytestmark = pytest.mark.usefixtures("clean_env")


@pytest.fixture
def alembic_config() -> Config:
    return Config(str(ALEMBIC_INI))


def test_alembic_ini_does_not_hardcode_a_database_url(alembic_config: Config) -> None:
    assert alembic_config.get_main_option("sqlalchemy.url") is None


def test_script_directory_loads(alembic_config: Config) -> None:
    script = ScriptDirectory.from_config(alembic_config)

    assert Path(script.dir) == ALEMBIC_INI.parent / "backend" / "alembic"


def test_offline_mode_uses_settings_without_connecting(
    alembic_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Closed port: offline (--sql) mode must succeed without opening a connection.
    monkeypatch.setenv("DATABASE_URL", UNREACHABLE_DATABASE_URL)

    command.upgrade(alembic_config, "head", sql=True)


@pytest.mark.parametrize("redis_url", [None, "not-a-redis-url"])
def test_env_uses_database_only_settings(
    alembic_config: Config, monkeypatch: pytest.MonkeyPatch, redis_url: str | None
) -> None:
    # Missing or invalid REDIS_URL must not matter to migrations.
    monkeypatch.setenv("DATABASE_URL", UNREACHABLE_DATABASE_URL)
    if redis_url is not None:
        monkeypatch.setenv("REDIS_URL", redis_url)

    command.upgrade(alembic_config, "head", sql=True)

    assert get_database_settings.cache_info().currsize == 1
    assert get_settings.cache_info().currsize == 0


def test_env_requires_database_url(alembic_config: Config) -> None:
    with pytest.raises(ValidationError):
        command.upgrade(alembic_config, "head", sql=True)
