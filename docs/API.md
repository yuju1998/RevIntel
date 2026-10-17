# RevIntel API

Interactive documentation is served by FastAPI at `/docs` (Swagger UI) and `/redoc`;
the OpenAPI schema is at `/openapi.json`.

## Conventions

- Business endpoints are versioned under `/api/v1`.
- Infrastructure probes (`/health`, `/ready`) are served at the root so load balancers
  and ECS health checks do not depend on API versioning.

## Running locally

```bash
source ~/.virtualenvs/revintel/bin/activate
docker compose up -d                       # PostgreSQL + Redis
cp .env.example .env                       # first time only
fastapi dev backend/app/main.py
```

## Configuration

Settings are loaded from environment variables, falling back to `.env`
(see `backend/app/core/config.py`). Unknown keys are ignored.

The API needs every required variable below. Alembic only needs `DATABASE_URL` and the
`DB_*` settings (see `docs/DATABASE.md`).

| Variable | Required | Default | Notes |
|---|---|---|---|
| `APP_NAME` | no | `RevIntel` | |
| `ENVIRONMENT` | no | `local` | One of `local`, `test`, `staging`, `production`. |
| `DATABASE_URL` | yes | — | Must use `postgresql+psycopg://` (explicit psycopg 3 driver) and include a database name. |
| `DB_POOL_SIZE` | no | `5` | |
| `DB_MAX_OVERFLOW` | no | `10` | |
| `DB_POOL_TIMEOUT_SECONDS` | no | `10` | Max wait for a pooled connection. |
| `DB_CONNECT_TIMEOUT_SECONDS` | no | `3` | libpq connect timeout (integer seconds). |
| `REDIS_URL` | yes | — | `redis://` or `rediss://`. |
| `REDIS_SOCKET_TIMEOUT_SECONDS` | no | `2` | Applies to connect and to commands. |

Invalid or missing required settings stop the application at startup. Validation errors
do not echo the offending values, so connection strings do not end up in logs.

## Endpoints

### `GET /health`

Liveness probe. Returns `200` while the API process can serve requests. It never
contacts PostgreSQL or Redis, so a dependency outage does not cause the container
to be restarted.

**200 OK**

```json
{"status": "ok"}
```

### `GET /ready`

Readiness probe. Runs `SELECT 1` against PostgreSQL and `PING` against Redis.

- `200 OK` when both checks succeed.
- `503 Service Unavailable` when either check fails or times out.

The body always has the same shape. Failures are reported only as `"error"`;
connection strings, credentials and exception messages are never returned.

**200 OK**

```json
{
  "status": "ready",
  "checks": {
    "postgres": {"status": "ok", "latency_ms": 2.4},
    "redis": {"status": "ok", "latency_ms": 0.8}
  }
}
```

**503 Service Unavailable**

```json
{
  "status": "not_ready",
  "checks": {
    "postgres": {"status": "error", "latency_ms": 3001.7},
    "redis": {"status": "ok", "latency_ms": 0.9}
  }
}
```

| Field | Type | Description |
|---|---|---|
| `status` | `"ready"` \| `"not_ready"` | Overall readiness. |
| `checks.<name>.status` | `"ok"` \| `"error"` | Result of an individual dependency check. |
| `checks.<name>.latency_ms` | number | Time spent on the check, in milliseconds. |

## Tests

```bash
pytest                  # unit/application tests; no Docker required
pytest -m integration   # real PostgreSQL/Redis tests; requires `docker compose up -d`
```

Integration tests use `revintel_test` by default; override with `TEST_DATABASE_URL`
and `TEST_REDIS_URL`.
