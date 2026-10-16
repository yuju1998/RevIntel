"""Centralized application configuration loaded from environment variables and `.env`."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Name the driver explicitly instead of relying on SQLAlchemy's default for `postgresql://`.
REQUIRED_DATABASE_SCHEME = "postgresql+psycopg"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # `.env` also holds settings for features that are not built yet (JWT, MLflow, ...).
        extra="ignore",
        # Keep connection strings (and their credentials) out of validation error messages.
        hide_input_in_errors=True,
    )

    app_name: str = "RevIntel"
    environment: Literal["local", "test", "staging", "production"] = "local"

    database_url: PostgresDsn
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=10, ge=0)
    db_pool_timeout_seconds: float = Field(default=10.0, gt=0)
    db_connect_timeout_seconds: int = Field(default=3, ge=1)

    redis_url: RedisDsn
    redis_socket_timeout_seconds: float = Field(default=2.0, gt=0)

    @field_validator("database_url")
    @classmethod
    def _require_psycopg_database_url(cls, value: PostgresDsn) -> PostgresDsn:
        if value.scheme != REQUIRED_DATABASE_SCHEME:
            raise ValueError(f"DATABASE_URL must use the '{REQUIRED_DATABASE_SCHEME}' scheme")
        if not (value.path or "").strip("/"):
            raise ValueError("DATABASE_URL must include a database name")
        return value


@lru_cache
def get_settings() -> Settings:
    # Required fields (DATABASE_URL, REDIS_URL) come from the environment or `.env`.
    return Settings()  # type: ignore[call-arg]
