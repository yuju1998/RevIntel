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

## CRM API conventions (`/api/v1`)

Authentication is not implemented yet; all endpoints below are currently open.

### Errors

| Situation | Status | Body |
|---|---|---|
| Malformed JSON, wrong types, failed field validation, unknown fields, malformed UUID | `422` | FastAPI validation error: `{"detail": [{"loc": [...], "msg": "...", ...}]}` |
| Valid UUID in the **path** that does not exist | `404` | `{"detail": "Account <id> not found"}` |
| Valid UUID in the **body** referring to a nonexistent record (e.g. `account_id`, `owner_id`) | `422` | `{"detail": "account_id does not refer to an existing account"}` |
| Business rule violated (e.g. inactive owner, creating a `CONVERTED` lead) | `422` | `{"detail": "owner_id refers to an inactive user"}` |

Request bodies reject unknown fields (`422`), so a misspelled or unsupported field is never
silently ignored.

### Data representation

- **IDs** are UUIDs.
- **Money** (`annual_revenue`, `estimated_value`, `amount`) is USD with two decimal places.
  Requests accept a JSON number or string (`1234.5` or `"1234.50"`); more than two decimal
  places, negative values, `NaN` and infinities are rejected. **Responses always return
  money as a string** (`"1234.50"`) so decimal precision is never lost to floating point.
  Clients should parse it with a decimal type.
- **Timestamps** (`created_at`, `updated_at`, `converted_at`) are ISO 8601 in UTC, e.g.
  `"2026-10-20T23:15:27.785992Z"`. `expected_close_date` is a date (`"2026-12-31"`).
- **Text** fields are trimmed; required text must be non-blank. Maximum lengths match the
  database (names 255, email 320, phone 50, website 2048).
- **Contact and lead emails** are trimmed and syntax-checked, but otherwise stored exactly
  as entered (casing preserved). They are not unique.
- **Counts** (`employee_count`, `company_size`) are integers `>= 0`.
- **`owner_id`** is optional. If given, it must refer to an existing, **active** user,
  otherwise `422`. Owners are never assigned automatically.

### Pagination

List endpoints use limit/offset pagination:

| Query parameter | Default | Constraint |
|---|---|---|
| `limit` | `50` | `1`–`100` |
| `offset` | `0` | `>= 0` |

```json
{"items": [...], "total": 3, "limit": 50, "offset": 0}
```

`total` is the number of rows **matching the filters**, not the size of the table. Items
are ordered newest first (`created_at` descending, then `id`). An offset past the end
returns an empty `items` list with the correct `total`.

### Filters

Filters are optional, use exact matching, and combine with pagination.

| Endpoint | Filters |
|---|---|
| `GET /api/v1/accounts` | `industry` (exact, case-sensitive) |
| `GET /api/v1/leads` | `status`, `lead_source` (enum values; unknown values return `422`) |

Examples: `GET /api/v1/leads?status=QUALIFIED&lead_source=WEB&limit=20&offset=20`,
`GET /api/v1/accounts?industry=Software`.

## CRM endpoints

### Accounts

#### `POST /api/v1/accounts` → `201`

| Field | Type | Required |
|---|---|---|
| `name` | string (≤255) | yes |
| `industry` | string (≤255) | no |
| `employee_count` | integer `>= 0` | no |
| `annual_revenue` | money, up to 14 digits before the decimal point | no |
| `website` | string (≤2048) | no |
| `owner_id` | UUID of an active user | no |

Account names are not unique. `422` for validation errors or an unknown/inactive owner.

```json
{
  "id": "222342a4-6a20-4ddf-ac41-e37935f2e065",
  "name": "Acme Corp",
  "industry": "Software",
  "employee_count": 250,
  "annual_revenue": "1234567.50",
  "website": null,
  "owner_id": null,
  "created_at": "2026-10-20T23:15:27.785992Z",
  "updated_at": "2026-10-20T23:15:27.785992Z"
}
```

#### `GET /api/v1/accounts` → `200`

Paginated list of accounts. Optional filter: `industry`.

#### `GET /api/v1/accounts/{account_id}` → `200`

`404` if the account does not exist; `422` for a malformed UUID.

### Contacts

#### `POST /api/v1/contacts` → `201`

| Field | Type | Required |
|---|---|---|
| `account_id` | UUID of an existing account | yes |
| `first_name`, `last_name` | string (≤255) | yes |
| `email` | email (≤320), stored as entered | no |
| `phone` | string (≤50) | no |
| `job_title` | string (≤255) | no |

`422` if `account_id` does not exist. The response mirrors the stored contact, including
`id`, `created_at` and `updated_at`.

### Leads

#### `POST /api/v1/leads` → `201`

| Field | Type | Required |
|---|---|---|
| `company_name`, `contact_name` | string (≤255) | yes |
| `lead_source` | `WEB`, `REFERRAL`, `EVENT`, `OUTBOUND`, `PARTNER`, `OTHER` | yes |
| `email` | email (≤320), stored as entered | no |
| `industry` | string (≤255) | no |
| `company_size` | integer `>= 0` | no |
| `estimated_value` | money, up to 12 digits before the decimal point | no |
| `status` | `NEW` (default), `WORKING`, `QUALIFIED`, `DISQUALIFIED` | no |
| `owner_id` | UUID of an active user | no |

`status: "CONVERTED"` returns `422`: lead conversion will be a separate workflow.
`converted_at` is not accepted (unknown field, `422`); it is returned as `null`.

#### `GET /api/v1/leads` → `200`

Paginated list of leads. Optional filters: `status`, `lead_source`.

#### `GET /api/v1/leads/{lead_id}` → `200`

`404` if the lead does not exist; `422` for a malformed UUID.

### Opportunities

#### `POST /api/v1/opportunities` → `201`

| Field | Type | Required |
|---|---|---|
| `account_id` | UUID of an existing account | yes |
| `name` | string (≤255) | yes |
| `stage` | `PROSPECTING`, `QUALIFICATION`, `PROPOSAL`, `NEGOTIATION`, `CLOSED_WON`, `CLOSED_LOST` | yes |
| `amount` | money, up to 12 digits before the decimal point | no |
| `expected_close_date` | date | no |
| `description` | string | no |
| `owner_id` | UUID of an active user | no |

`422` if `account_id` does not exist or the owner is unknown/inactive.

#### `PATCH /api/v1/opportunities/{opportunity_id}` → `200`

Partial update. Updatable fields: `name`, `stage`, `amount`, `expected_close_date`,
`description`, `owner_id`.

- **Omitted fields are left unchanged.**
- `null` clears a nullable field (`amount`, `expected_close_date`, `description`,
  `owner_id`); `null` for `name` or `stage` returns `422`.
- `account_id` cannot be changed: opportunities do not move between accounts (`422`).
- Any stage may change to any other stage.
- `{}` is a no-op: `200` with the unchanged opportunity; `updated_at` does not change.
- `updated_at` advances only when a value actually changes.
- `404` if the opportunity does not exist.

## Tests

```bash
pytest                  # unit/application tests; no Docker required
pytest -m integration   # real PostgreSQL/Redis tests; requires `docker compose up -d`
```

Integration tests use `revintel_test` by default; override with `TEST_DATABASE_URL`
and `TEST_REDIS_URL`. The test database is rebuilt from the Alembic migrations at the start
of each run, and setup refuses to run unless the database name ends in `_test`
(see `docs/DATABASE.md`).
