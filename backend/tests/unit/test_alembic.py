import io
import re
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from pydantic import ValidationError

from app.core.config import get_database_settings, get_settings
from tests.unit.helpers import UNREACHABLE_DATABASE_URL

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


def _offline_sql(monkeypatch: pytest.MonkeyPatch, upgrade: bool) -> str:
    monkeypatch.setenv("DATABASE_URL", UNREACHABLE_DATABASE_URL)
    buffer = io.StringIO()
    config = Config(str(ALEMBIC_INI), output_buffer=buffer)
    if upgrade:
        command.upgrade(config, "head", sql=True)
    else:
        command.downgrade(config, "head:base", sql=True)
    return buffer.getvalue()


def test_migrations_have_a_single_head(alembic_config: Config) -> None:
    assert len(ScriptDirectory.from_config(alembic_config).get_heads()) == 1


def test_migration_chain_is_linear(alembic_config: Config) -> None:
    script = ScriptDirectory.from_config(alembic_config)
    chain = [rev.revision for rev in reversed(list(script.walk_revisions()))]

    assert chain == ["d9abf148e9ed", "9d4d176a3488"]
    assert script.get_revision("9d4d176a3488").down_revision == "d9abf148e9ed"


def test_upgrade_sql_creates_only_implemented_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    sql = _offline_sql(monkeypatch, upgrade=True)

    created = set(re.findall(r"CREATE TABLE (\w+)", sql)) - {"alembic_version"}
    assert created == {
        "users",
        "accounts",
        "contacts",
        "leads",
        "opportunities",
        "activities",
        "notes",
    }
    assert "vector" not in sql.lower()


def test_second_migration_only_adds_activities_and_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", UNREACHABLE_DATABASE_URL)
    buffer = io.StringIO()
    command.upgrade(Config(str(ALEMBIC_INI), output_buffer=buffer), "d9abf148e9ed:head", sql=True)
    sql = buffer.getvalue()

    assert set(re.findall(r"CREATE TABLE (\w+)", sql)) == {"activities", "notes"}
    assert not re.search(r"(ALTER|DROP) TABLE", sql)
    for name in ("ck_activities_activity_type", "ck_activities_activity_direction"):
        assert len(re.findall(rf"CONSTRAINT {name} ", sql)) == 1


def test_upgrade_sql_names_each_constraint_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    sql = _offline_sql(monkeypatch, upgrade=True)
    names = re.findall(r"CONSTRAINT (\w+)", sql)

    assert len(names) == len(set(names))
    # Enum CHECKs are named only through the naming convention, never bare enum names.
    assert not {
        "user_role",
        "lead_source",
        "lead_status",
        "opportunity_stage",
        "activity_type",
        "activity_direction",
    } & set(names)
    for name in (
        "uq_users_email",
        "ck_leads_converted_requires_converted_at",
        "ck_opportunities_opportunity_stage",
        "fk_contacts_account_id_accounts",
    ):
        assert name in names


def test_downgrade_sql_drops_tables_in_reverse_dependency_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sql = _offline_sql(monkeypatch, upgrade=False)

    dropped = [name for name in re.findall(r"DROP TABLE (\w+)", sql) if name != "alembic_version"]
    assert dropped == [
        "notes",
        "activities",
        "opportunities",
        "leads",
        "contacts",
        "accounts",
        "users",
    ]
