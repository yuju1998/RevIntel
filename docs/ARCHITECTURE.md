# RevIntel Architecture

This describes the backend as it exists today. Related documents: `docs/DATABASE.md`
(authoritative schema), `docs/API.md` (endpoints and behavior), `docs/TASKS.md` (progress).

## Overview

A single FastAPI application (Python 3.14, synchronous SQLAlchemy 2.x + psycopg 3)
backed by PostgreSQL (pgvector image) and Redis, run locally with `docker compose`.

```
HTTP request
  -> RequestContextMiddleware     request_id, access log, safe 500
  -> route (app/api/routes)       validation, auth dependency, HTTP mapping
  -> service (app/services)       business rules, transactions
  -> repository (app/repositories) queries; add/flush only
  -> SQLAlchemy -> PostgreSQL
                 \-> Redis (account cache only, optional)
```

Routes contain no business logic; services never build HTTP responses; repositories never
commit.

## Module boundaries

| Package | Responsibility |
|---|---|
| `app/main.py` | `create_app()`; the lifespan resolves settings, configures logging and creates the engine, session factory and Redis clients on `app.state`. Importing the app opens no connections. |
| `app/api/` | `routes/` (one module per resource), `deps.py` (all providers: settings, engine, sessions, Redis clients, services), `auth.py` (current user, role helper), `errors.py` (domain error -> HTTP), `middleware.py`, `pagination.py` (query params), `sensitive_route.py` (login validation errors without input). |
| `app/services/` | One service per resource plus shared helpers: `transaction.py` (`write_transaction`), `crm_references.py` (parent validation and canonical `account_id`), `user_validation.py` (owner checks), `pagination.py` (`PageResult`), `account_cache.py`, `auth_service.py`, `user_service.py`. |
| `app/repositories/` | Data access per table, plus `timeline.py` for the combined activity/note query. |
| `app/models/` | SQLAlchemy models; `enums.py` holds the domain enums shared with schemas. |
| `app/schemas/` | Pydantic request (`extra="forbid"`) and response models; `common.py` holds shared field types and `Page`. |
| `app/db/` | Declarative `Base` (naming convention), mixins, enum column helper, engine/session factories. |
| `app/core/` | Settings, exceptions, security (Argon2, JWT), logging, cache abstraction, Redis client factory, email normalization. |
| `app/cli/` | `create_user` (there is no signup API). |
| `backend/alembic/` | Migrations (`d9abf148e9ed` CRM core, `9d4d176a3488` activities/notes). |

`ml/synthetic/` generates deterministic synthetic CRM data (raw rows only, no ML features
or labels); `scripts/generate_crm_data.py` is its CLI. It writes only to guarded `*_ml` /
`*_test` databases (see `docs/DATABASE.md`, "Synthetic CRM data").

Placeholder packages `app/ai/` and `app/ml_inference/` exist for future layers and contain
no code yet.

## Transactions

- One SQLAlchemy `Session` per request (`get_db`). It never commits; it rolls back on an
  exception and always closes.
- Services own transactions: every write runs inside `write_transaction(session,
  constraint_errors)`, which commits once or rolls back.
- Services validate related records before writing. If a constraint is still violated (a
  race), only constraint names listed in the service's `CONSTRAINT_ERRORS` become domain
  errors; any other `IntegrityError` is re-raised unchanged.
- Reads never commit.

## Error handling

- Services raise domain errors from `app/core/exceptions.py`; `app/api/errors.py` maps
  them: `NotFoundError` -> 404, `RelatedResourceNotFoundError` / `BusinessRuleError` ->
  422, `AuthenticationError` -> 401 (with `WWW-Authenticate: Bearer`),
  `PermissionDeniedError` -> 403. Bodies are `{"detail": "<message>"}`.
- Request validation errors use FastAPI's standard 422 body, except on the login route,
  whose errors omit the submitted input.
- Unhandled exceptions are caught by the middleware: logged once (type and stack frames
  only) and returned as `500 {"detail": "Internal Server Error"}`.

## Authentication

- `POST /api/v1/auth/login` (JSON) verifies an Argon2 hash (pwdlib) and issues an HS256
  JWT with `sub`, `iat`, `exp`, `iss`, `aud`, `type`. Stateless; no refresh tokens.
- `get_current_user` decodes the token and loads the user from PostgreSQL on every
  request; unknown, deleted or inactive users get 401. Role and status are never trusted
  from the token.
- All CRM mutations require a token; CRM reads are public. `require_roles(...)` exists but
  ADMIN and SALES_REP currently have the same permissions.
- Activity `created_by_id` and note `author_id` always come from the current user.
- Users are created with the CLI through `UserService` (email normalized to lowercase).

## Request logging and request IDs

- structlog over stdlib logging: one root handler formats application and library logs
  alike. `ENVIRONMENT=local` writes human-readable lines; other environments write JSON.
- `RequestContextMiddleware` (pure ASGI) generates a UUID4 `request_id` per request
  (incoming `X-Request-ID` is ignored), binds it to contextvars, returns it in
  `X-Request-ID`, and writes one `http_request` access event (method, path, canonical
  route, status, duration, user_id when authenticated).
- Redaction by key and value pattern keeps passwords, hashes, tokens, the JWT secret and
  URL credentials out of logs. SQLAlchemy statement and httpx request logging are capped
  at WARNING.

## Redis cache-aside

- `app/core/cache.py` defines a small `Cache` protocol with a fail-open `RedisCache`
  (errors become misses or no-ops) and a `NullCache`.
- Only `GET /api/v1/accounts/{account_id}` is cached: `AccountService.get_account` checks
  `AccountCache` (key `revintel:v1:account:{id}`, `AccountRead` JSON, TTL
  `ACCOUNT_CACHE_TTL_SECONDS`), falls back to PostgreSQL on a miss, and never caches
  404s.
- The cache uses its own short-timeout Redis client; `/ready` uses the default client and
  still reports Redis as a dependency.
- `AccountCache.invalidate()` exists for future account mutations (call after commit);
  nothing invokes it today because accounts have no update/delete API.

## Activities, notes and the account timeline

- Activities and notes reference at least one CRM parent (enforced by a CHECK and by
  request validation). `crm_references.resolve_account_id` checks that the referenced
  account, opportunity and contact belong to one account and stores that account as the
  canonical `account_id`; lead-only records keep `account_id` NULL.
- `GET /api/v1/accounts/{id}/timeline` merges both record types with a `UNION ALL` in
  `TimelineRepository`, ordered and paginated in PostgreSQL (`event_timestamp DESC,
  record_type ASC, id DESC`), then loads the page's rows. `event_timestamp` is API-only
  (activity `occurred_at`, note `created_at`).

## Testing

- Unit tests (`pytest`) need no Docker: fakes and dependency overrides.
- Integration tests (`pytest -m integration`) use `revintel_test` (rebuilt from the
  Alembic migrations, guarded to `*_test` databases) and Redis database 15 (guarded
  against database 0).
- `ml/tests` are in-memory unit tests of the synthetic generator (determinism,
  relationships, constraints, point-in-time invariants, broad realism bands).

## Future layers (not implemented)

- The lead-conversion ML model (snapshot labels, feature builder, training,
  evaluation, MLflow registry) and online scoring into `lead_predictions`.
- MLOps orchestration with Airflow.
- RAG over `document_chunks` with pgvector, account-scoped retrieval and Bedrock
  generation.

Their schema is designed in `docs/DATABASE.md`; none of their code exists yet.
