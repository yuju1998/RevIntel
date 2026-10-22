from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import DatabaseSettings, Settings, get_database_settings, get_settings

pytestmark = pytest.mark.usefixtures("clean_env")

TEST_JWT_SECRET = "config-test-jwt-secret-0123456789abcdef"


@pytest.fixture(autouse=True)
def _jwt_secret(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # Settings requires JWT_SECRET; provide it so other assertions stay focused.
    monkeypatch.setenv("JWT_SECRET", TEST_JWT_SECRET)

VALID_DATABASE_URL = "postgresql+psycopg://revintel:s3cret@localhost:5432/revintel"
VALID_REDIS_URL = "redis://localhost:6379/0"


def test_loads_settings_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "APP_NAME=RevIntel Test\n"
        "ENVIRONMENT=test\n"
        f"DATABASE_URL={VALID_DATABASE_URL}\n"
        f"REDIS_URL={VALID_REDIS_URL}\n"
        "DB_POOL_SIZE=7\n"
        # Keys for features that are not built yet must not break loading.
        "MLFLOW_TRACKING_URI=http://localhost:5001\n"
    )

    settings = Settings(_env_file=env_file)

    assert settings.app_name == "RevIntel Test"
    assert settings.environment == "test"
    assert str(settings.database_url) == VALID_DATABASE_URL
    assert str(settings.redis_url) == VALID_REDIS_URL
    assert settings.db_pool_size == 7
    assert not hasattr(settings, "mlflow_tracking_uri")


def test_environment_variables_override_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"DATABASE_URL={VALID_DATABASE_URL}\nREDIS_URL={VALID_REDIS_URL}\n")
    monkeypatch.setenv("REDIS_URL", "redis://cache.internal:6380/2")

    settings = Settings(_env_file=env_file)

    assert str(settings.redis_url) == "redis://cache.internal:6380/2"


def test_defaults_apply_when_optional_values_are_missing() -> None:
    settings = Settings(_env_file=None, database_url=VALID_DATABASE_URL, redis_url=VALID_REDIS_URL)

    assert settings.app_name == "RevIntel"
    assert settings.environment == "local"
    assert settings.db_connect_timeout_seconds > 0
    assert settings.redis_socket_timeout_seconds > 0


@pytest.mark.parametrize("missing", ["database_url", "redis_url"])
def test_missing_required_url_raises(missing: str) -> None:
    values = {"database_url": VALID_DATABASE_URL, "redis_url": VALID_REDIS_URL}
    del values[missing]

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, **values)

    assert exc_info.value.errors()[0]["loc"] == (missing,)


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://revintel:s3cret@localhost:5432/revintel",
        "postgresql+psycopg2://revintel:s3cret@localhost:5432/revintel",
        "postgresql+asyncpg://revintel:s3cret@localhost:5432/revintel",
        "mysql://revintel:s3cret@localhost:3306/revintel",
        "not-a-url",
    ],
)
def test_rejects_database_url_without_psycopg_driver(database_url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url=database_url, redis_url=VALID_REDIS_URL)


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql+psycopg://revintel:s3cret@localhost:5432",
        "postgresql+psycopg://revintel:s3cret@localhost:5432/",
    ],
)
def test_rejects_database_url_without_database_name(database_url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url=database_url, redis_url=VALID_REDIS_URL)


@pytest.mark.parametrize(
    "redis_url", ["http://localhost:6379/0", "localhost:6379", "postgresql://localhost/0"]
)
def test_rejects_invalid_redis_url(redis_url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url=VALID_DATABASE_URL, redis_url=redis_url)


def test_rejects_unknown_environment() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            environment="prod-ish",
            database_url=VALID_DATABASE_URL,
            redis_url=VALID_REDIS_URL,
        )


def test_validation_errors_do_not_echo_credentials() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            database_url="postgresql://revintel:s3cret@localhost:5432/revintel",
            redis_url=VALID_REDIS_URL,
        )

    assert "s3cret" not in str(exc_info.value)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    monkeypatch.setenv("REDIS_URL", VALID_REDIS_URL)

    assert get_settings() is get_settings()


def test_database_settings_load_without_redis_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    monkeypatch.setenv("DB_CONNECT_TIMEOUT_SECONDS", "7")

    settings = DatabaseSettings()

    assert str(settings.database_url) == VALID_DATABASE_URL
    assert settings.db_connect_timeout_seconds == 7
    assert not hasattr(settings, "redis_url")


def test_database_settings_ignore_application_settings_in_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"DATABASE_URL={VALID_DATABASE_URL}\nREDIS_URL=not-a-redis-url\nAPP_NAME=Ignored\n"
    )

    settings = DatabaseSettings(_env_file=env_file)

    assert str(settings.database_url) == VALID_DATABASE_URL
    assert not hasattr(settings, "app_name")


def test_database_settings_require_database_url() -> None:
    with pytest.raises(ValidationError) as exc_info:
        DatabaseSettings(_env_file=None)

    assert exc_info.value.errors()[0]["loc"] == ("database_url",)


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://revintel:s3cret@localhost:5432/revintel",
        "postgresql+psycopg://revintel:s3cret@localhost:5432",
    ],
)
def test_database_settings_share_database_url_validation(database_url: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        DatabaseSettings(_env_file=None, database_url=database_url)

    assert exc_info.value.errors()[0]["loc"] == ("database_url",)
    assert "s3cret" not in str(exc_info.value)


def test_application_settings_still_require_redis_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert [error["loc"] for error in exc_info.value.errors()] == [("redis_url",)]


def test_application_settings_include_database_settings() -> None:
    settings = Settings(_env_file=None, database_url=VALID_DATABASE_URL, redis_url=VALID_REDIS_URL)

    assert isinstance(settings, DatabaseSettings)
    assert str(settings.database_url) == VALID_DATABASE_URL


def test_get_database_settings_is_cached_and_independent_of_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)

    assert get_database_settings() is get_database_settings()
    with pytest.raises(ValidationError):
        get_settings()


def _app_settings(**fields: object) -> Settings:
    values = {"database_url": VALID_DATABASE_URL, "redis_url": VALID_REDIS_URL}
    return Settings(_env_file=None, **{**values, **fields})


def test_jwt_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _app_settings()

    assert settings.jwt_secret.get_secret_value() == TEST_JWT_SECRET
    assert settings.jwt_algorithm == "HS256"
    assert settings.access_token_expire_minutes == 60


def test_jwt_secret_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JWT_SECRET")

    with pytest.raises(ValidationError) as exc_info:
        _app_settings()

    assert [error["loc"] for error in exc_info.value.errors()] == [("jwt_secret",)]


def test_jwt_secret_must_be_at_least_32_characters() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _app_settings(jwt_secret="too-short-secret")

    assert "too-short-secret" not in str(exc_info.value)


@pytest.mark.parametrize("algorithm", ["HS512", "RS256", "none"])
def test_only_hs256_is_accepted(algorithm: str) -> None:
    with pytest.raises(ValidationError):
        _app_settings(jwt_algorithm=algorithm)


@pytest.mark.parametrize("minutes", [0, 1441])
def test_token_lifetime_is_bounded(minutes: int) -> None:
    with pytest.raises(ValidationError):
        _app_settings(access_token_expire_minutes=minutes)


def test_jwt_secret_is_hidden_in_repr() -> None:
    settings = _app_settings()

    assert TEST_JWT_SECRET not in repr(settings)
    assert TEST_JWT_SECRET not in str(settings.model_dump())


def test_database_settings_do_not_require_jwt_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JWT_SECRET")
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)

    assert DatabaseSettings().database_url is not None
