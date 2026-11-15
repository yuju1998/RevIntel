"""Centralized application configuration loaded from environment variables and `.env`."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Name the driver explicitly instead of relying on SQLAlchemy's default for `postgresql://`.
REQUIRED_DATABASE_SCHEME = "postgresql+psycopg"
# MLflow defaults; kept equal to ml.training.mlflow_config by a unit test. Not imported
# from `ml` so that settings (and Alembic, which loads them) stay free of the ML package.
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
DEFAULT_MLFLOW_MODEL_NAME = "revintel-lead-conversion"
# Embedding defaults; EMBEDDING_DIMENSION must equal the VECTOR(384) column width.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


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


class EmbeddingSettings(BaseSettings):
    """Embedding settings, shared by the API and the CRM indexing CLI."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Only local sentence-transformers embeddings exist in V1; the dimension is fixed by
    # document_chunks.embedding (VECTOR(384)).
    embedding_provider: Literal["local"] = "local"
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, min_length=1)
    embedding_dimension: int = EMBEDDING_DIMENSION
    embedding_batch_size: int = Field(default=64, ge=1, le=1024)

    @field_validator("embedding_dimension")
    @classmethod
    def _require_schema_dimension(cls, value: int) -> int:
        if value != EMBEDDING_DIMENSION:
            raise ValueError(
                f"EMBEDDING_DIMENSION must be {EMBEDDING_DIMENSION} (document_chunks.embedding)"
            )
        return value


class LLMSettings(BaseSettings):
    """LLM provider settings, shared by the API and the RAG evaluation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Keep the API key out of validation error messages.
        hide_input_in_errors=True,
    )

    # CRM Copilot answer generation. `mock` is deterministic and offline (local/tests);
    # `openai` is the V1 production provider and then requires a key and a model.
    llm_provider: Literal["mock", "openai"] = "mock"
    # Never logged or shown (SecretStr).
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    # 0 = low-variance answers. Empty to omit (for models that reject `temperature`).
    openai_temperature: float | None = Field(default=0.0, ge=0, le=2)
    openai_max_output_tokens: int = Field(default=800, ge=1, le=16384)
    openai_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    openai_max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("openai_api_key", "openai_model", "openai_temperature", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _require_openai_settings(self) -> Self:
        if self.llm_provider == "openai":
            if self.openai_api_key is None or not self.openai_api_key.get_secret_value():
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            if not self.openai_model:
                raise ValueError("OPENAI_MODEL is required when LLM_PROVIDER=openai")
        return self


class Settings(DatabaseSettings, EmbeddingSettings, LLMSettings):
    """Full configuration for the FastAPI application."""

    app_name: str = "RevIntel"
    # "local" logs human-readable lines; every other environment logs JSON.
    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    redis_url: RedisDsn
    redis_socket_timeout_seconds: float = Field(default=2.0, gt=0)
    # The cache uses its own client with a short timeout: a slow Redis must not stall
    # requests (worst case one read + one write timeout before PostgreSQL answers).
    cache_redis_timeout_seconds: float = Field(default=0.2, gt=0, le=5)
    # GET /accounts/{id} cache lifetime; 0 disables caching.
    account_cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)

    # Required; never logged or shown (SecretStr). HS256 is fixed for V1.
    jwt_secret: SecretStr = Field(min_length=32)
    jwt_algorithm: Literal["HS256"] = "HS256"
    access_token_expire_minutes: int = Field(default=60, ge=1, le=1440)

    # Online lead scoring. The `champion` alias of this registered model is resolved and
    # loaded once per process; restart the API after promoting a new champion.
    mlflow_tracking_uri: str = Field(default=DEFAULT_MLFLOW_TRACKING_URI, min_length=1)
    mlflow_model_name: str = Field(default=DEFAULT_MLFLOW_MODEL_NAME, min_length=1)


@lru_cache
def get_database_settings() -> DatabaseSettings:
    # DATABASE_URL comes from the environment or `.env`; other services' settings are ignored.
    return DatabaseSettings()  # type: ignore[call-arg]


@lru_cache
def get_embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings()


@lru_cache
def get_llm_settings() -> LLMSettings:
    return LLMSettings()


@lru_cache
def get_settings() -> Settings:
    # Required fields (DATABASE_URL, REDIS_URL, JWT_SECRET) come from the environment or `.env`.
    return Settings()  # type: ignore[call-arg]
