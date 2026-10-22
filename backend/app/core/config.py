"""Centralized application configuration loaded from environment variables and `.env`."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Name the driver explicitly instead of relying on SQLAlchemy's default for `postgresql://`.
REQUIRED_DATABASE_SCHEME = "postgresql+psycopg"


class DatabaseSettings(BaseSettings):
    """PostgreSQL settings only, for consumers such as Alembic that need nothing else."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # `.env` also holds settings for other components and features not built yet.
        extra="ignore",
        # Keep connection strings (and their credentials) out of validation error messages.
        hide_input_in_errors=True,
    )

    database_url: PostgresDsn
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=10, ge=0)
    db_pool_timeout_seconds: float = Field(default=10.0, gt=0)
    db_connect_timeout_seconds: int = Field(default=3, ge=1)

    @field_validator("database_url")
    @classmethod
    def _require_psycopg_database_url(cls, value: PostgresDsn) -> PostgresDsn:
        if value.scheme != REQUIRED_DATABASE_SCHEME:
            raise ValueError(f"DATABASE_URL must use the '{REQUIRED_DATABASE_SCHEME}' scheme")
        if not (value.path or "").strip("/"):
            raise ValueError("DATABASE_URL must include a database name")
        return value


class Settings(DatabaseSettings):
    """Full configuration for the FastAPI application."""

    app_name: str = "RevIntel"
    environment: Literal["local", "test", "staging", "production"] = "local"

    redis_url: RedisDsn
    redis_socket_timeout_seconds: float = Field(default=2.0, gt=0)

    # Required; never logged or shown (SecretStr). HS256 is fixed for V1.
    jwt_secret: SecretStr = Field(min_length=32)
    jwt_algorithm: Literal["HS256"] = "HS256"
    access_token_expire_minutes: int = Field(default=60, ge=1, le=1440)


@lru_cache
def get_database_settings() -> DatabaseSettings:
    # DATABASE_URL comes from the environment or `.env`; other services' settings are ignored.
    return DatabaseSettings()  # type: ignore[call-arg]


@lru_cache
def get_settings() -> Settings:
    # Required fields (DATABASE_URL, REDIS_URL, JWT_SECRET) come from the environment or `.env`.
    return Settings()  # type: ignore[call-arg]
