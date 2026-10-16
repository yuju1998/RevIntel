# RevIntel

RevIntel is a Revenue Intelligence Platform demonstrating production-style
backend engineering, data engineering, machine learning, MLOps, and RAG.

# Architecture

Backend:
- Python 3.14
- FastAPI
- SQLAlchemy 2.x
- Alembic
- PostgreSQL
- pgvector
- Redis

ML:
- pandas
- scikit-learn
- XGBoost
- MLflow

Data orchestration:
- Airflow TaskFlow

AI:
- pgvector retrieval
- local embeddings during development
- Amazon Bedrock for production LLM generation

Infrastructure:
- Docker
- AWS ECS Fargate
- RDS PostgreSQL
- ECR
- S3
- Secrets Manager
- CloudWatch
- GitHub Actions

# Application Structure

Use this layering:

API Route
    ↓
Service
    ↓
Repository
    ↓
SQLAlchemy
    ↓
PostgreSQL

Business logic must never live directly inside route handlers.

# Coding Rules

- Python code must use type hints.
- Use SQLAlchemy 2.x patterns.
- Use Pydantic models for API validation.
- Use UUID primary keys unless there is a strong reason not to.
- Use timezone-aware UTC timestamps.
- Use Decimal/Numeric for currency values.
- Do not use float for money.
- API endpoints use `/api/v1`.
- Never hardcode secrets.
- Never commit `.env`.
- Never modify an existing Alembic migration after it has been committed.
- Schema changes require a new migration.
- Prefer small modules.
- Avoid circular imports.
- Add database indexes deliberately.
- Use centralized exception handling.
- Use structured logging.
- Add request IDs to logs.

# Testing Rules

Every feature must have tests.

Before completing a task, run:

ruff check .
pytest

Do not claim the task is complete if tests fail.

# ML Rules

The prediction task is:

Predict whether a lead converts within 30 days after the prediction timestamp.

Features must only contain data available at or before prediction time.

Never use:
- converted_at
- future activities
- future notes
- final lead status

as features.

Offline and online feature generation should share feature definitions whenever possible.

Use time-based train/validation/test splitting.

Package preprocessing with the model using sklearn Pipeline.

# MLflow Rules

Log:
- parameters
- metrics
- feature list
- dataset version/hash
- git commit
- model artifact
- model signature

Models are registered as candidates.

Never automatically promote every candidate to champion.

Use:
- champion
- challenger

aliases.

# RAG Rules

RAG retrieval must be scoped to the requested CRM entity/account.

Never retrieve unrelated customer information.

Each retrieved record must retain:
- source type
- source ID
- account ID
- timestamp
- text

Generated answers must cite source CRM IDs.

If evidence is insufficient, the assistant must say so.

# Security

- Hash passwords.
- Never log passwords, access tokens, API keys, or secrets.
- Validate authentication on protected endpoints.
- Only two roles exist in V1:
  - admin
  - sales_rep

Do not build complex RBAC.

# Definition of Done

A feature is complete only if:

1. implementation works
2. tests exist
3. tests pass
4. Ruff passes
5. database migrations exist when needed
6. API documentation is updated
7. failure scenarios have been considered

# Scope

Do NOT add:

- Kafka
- Spark
- Kubernetes
- microservices
- Gmail integration
- calendar integration
- multi-agent systems
- complex frontend
- additional ML models

unless explicitly requested.