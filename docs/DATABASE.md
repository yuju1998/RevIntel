# RevIntel Database

PostgreSQL 16 with pgvector, accessed through synchronous SQLAlchemy 2.x and psycopg 3.
Schema changes are managed with Alembic.

This document is the authoritative schema specification. ORM models and migrations must
match it. Change this document (with approval) before changing the schema.

**Status:** schema phases 1–2 are implemented: phase 1 (`users`, `accounts`, `contacts`,
`leads`, `opportunities`) in Alembic revision `d9abf148e9ed`, phase 2 (`activities`, `notes`)
in `9d4d176a3488`. Phases 3–4 are designed, not implemented. The
decisions behind this design are recorded in [Design decisions](#design-decisions-q1q14);
intentionally postponed work is in [Deferred / future considerations](#deferred--future-considerations).

## Contents

1. [Conventions](#conventions)
2. [Entity relationships](#entity-relationships)
3. [Tables](#tables)
4. [Enum values](#enum-values)
5. [Foreign keys and delete behavior](#foreign-keys-and-delete-behavior)
6. [Check constraints](#check-constraints)
7. [Index summary](#index-summary)
8. [Service-layer rules](#service-layer-rules)
9. [RAG isolation rule](#rag-isolation-rule)
10. [ML point-in-time rules](#ml-point-in-time-rules)
11. [Synthetic CRM data](#synthetic-crm-data)
12. [Lead conversion features](#lead-conversion-features)
13. [Implementation order](#implementation-order)
14. [Design decisions (Q1–Q14)](#design-decisions-q1q14)
15. [Deferred / future considerations](#deferred--future-considerations)
16. [Tooling: layout, Alembic, commands](#layout)

---

## Conventions

### Keys

- Every table has `id UUID PRIMARY KEY`.
- IDs are generated application-side with `uuid.uuid4()` (SQLAlchemy `default=uuid.uuid4`), so
  IDs are known before flush and the synthetic data generator can build related rows in
  memory. No server-side default.
- Foreign keys are `UUID` and always named `<role>_id` (`owner_id`, `author_id`,
  `created_by_id`, `account_id`, ...).

### Timestamps

- All timestamps are `TIMESTAMPTZ` (`DateTime(timezone=True)`) and represent UTC instants.
  Python code only creates timezone-aware UTC datetimes (`datetime.now(UTC)`); naive
  datetimes are a bug.
- `created_at`: `NOT NULL`, server default `now()`. Callers may set it explicitly (the
  synthetic data generator writes historical timestamps).
- `updated_at`: `NOT NULL`, server default `now()`, refreshed on every ORM/Core update via
  SQLAlchemy `onupdate=func.now()`. No database triggers. Raw SQL updates must set it
  themselves. Every mutable primary entity has `updated_at` (`users`, `accounts`,
  `contacts`, `leads`, `opportunities`, `notes`).
- Append-only tables have only `created_at`: `activities`, `lead_predictions`,
  `document_chunks`.
- Business-time columns (`occurred_at`, `converted_at`, `prediction_timestamp`) are set
  by the application and never defaulted to `now()`.
- `DATE` is used only for calendar dates without a time (`expected_close_date`).

### Money and numbers

- Currency values are `NUMERIC(p,2)` mapped to `Decimal`. Never `float`.
- **All monetary values are USD.** There are no currency columns and no conversion logic
  in V1.
- Money and count columns that cannot be negative have `>= 0` CHECK constraints (see
  [Check constraints](#check-constraints)).
- `lead_predictions.conversion_probability` is a probability, not money, so
  `DOUBLE PRECISION` (`float`) is correct there.

### Strings

All short text is bounded `VARCHAR`; long free-form content is `TEXT`.

| Kind | Type | Columns |
|---|---|---|
| Email | `VARCHAR(320)` | `users.email`, `contacts.email`, `leads.email` |
| Person / account / company / contact names | `VARCHAR(255)` | `users.full_name`, `accounts.name`, `contacts.first_name`, `contacts.last_name`, `leads.company_name`, `leads.contact_name`, `opportunities.name` |
| Industry | `VARCHAR(255)` | `accounts.industry`, `leads.industry` |
| Job title | `VARCHAR(255)` | `contacts.job_title` |
| Activity subject | `VARCHAR(255)` | `activities.subject` |
| Phone | `VARCHAR(50)` | `contacts.phone` |
| Website | `VARCHAR(2048)` | `accounts.website` |
| Password hash | `VARCHAR(255)` | `users.password_hash` |
| Model name | `VARCHAR(128)` | `lead_predictions.model_name` |
| Model version | `VARCHAR(64)` | `lead_predictions.model_version` |
| Embedding model | `VARCHAR(255)` | `document_chunks.embedding_model` |
| Enum-backed | `VARCHAR(32)` | every enum column |
| Long free-form | `TEXT` | `opportunities.description`, `activities.body`, `notes.body`, `document_chunks.content` |

Pydantic request models must apply the same maximum lengths.

### Emails

- `users.email` is globally unique and stored in **canonical lowercase**. The service layer
  lowercases the address before every insert/update and before every lookup (e.g. login).
  The database has a plain `UNIQUE (email)` only: no `CITEXT`, no `lower(email)` index.
- `contacts.email` and `leads.email` are not unique and are stored as entered.
- Account names are not unique.

### Enums

- Python `StrEnum` classes whose member values equal their names (e.g. `NEW = "NEW"`).
  The database stores the uppercase value.
- Mapped as **non-native** enums: `VARCHAR(32)` plus a `CHECK` constraint, using
  `sa.Enum(PyEnum, name="<enum_name>", native_enum=False, create_constraint=True, length=32,
  values_callable=lambda e: [m.value for m in e])`.
  - `create_constraint=True` is required; SQLAlchemy 2.x does not create the CHECK by default.
  - `name=` is required; the naming convention turns it into the CHECK name
    (`ck_<table>_<enum_name>`).
  - A fixed `length=32` lets new values be added without widening the column.
- Adding or removing an enum value requires a hand-written migration that drops and
  recreates the CHECK constraint. Alembic autogenerate does **not** detect CHECK changes.
  `ck_leads_converted_requires_converted_at` also references the literal `'CONVERTED'`.

### JSON

- `JSONB`. Python defaults use `default=dict` (never a shared `{}` literal); server
  default `'{}'::jsonb` where a default is specified.

### Constraint and index names

`Base.metadata` (in `backend/app/db/base.py`) applies this naming convention. It is **not**
changed for composite constraints; those get explicit names instead.

| Kind | Pattern | Example |
|---|---|---|
| Primary key | `pk_<table>` | `pk_users` |
| Foreign key | `fk_<table>_<column>_<referred_table>` | `fk_accounts_owner_id_users` |
| Unique (single column) | `uq_<table>_<column>` | `uq_users_email` |
| Check | `ck_<table>_<name>` | `ck_leads_lead_status` |
| Index (single column) | `ix_<table>_<column>` | `ix_leads_status` |

Rules:

- CHECK constraints are declared with a short `name=` token, which the convention prefixes:
  `CheckConstraint("amount >= 0", name="amount_non_negative")` →
  `ck_opportunities_amount_non_negative`.
- Composite unique constraints and composite indexes are declared with their **full
  explicit name** (the convention would otherwise name them after the first column only).
  Explicit `uq_`/`ix_` names are used verbatim.
- PostgreSQL identifiers are limited to 63 characters; SQLAlchemy raises `IdentifierError`
  for longer names. All names in this document fit.

### Scope

- No soft deletes (`deleted_at`, `is_deleted`) yet.
- No tenant / organization columns. RevIntel V1 is single-tenant.
- Users are deactivated (`is_active = false`), not deleted, in normal operation.
- V1 has no DELETE endpoints for CRM records; FK delete policies are still defined.
- Do not over-index: only the indexes listed in this document.

---

## Entity relationships

```mermaid
erDiagram
    users |o--o{ accounts : "owns (owner_id)"
    users |o--o{ leads : "owns (owner_id)"
    users |o--o{ opportunities : "owns (owner_id)"
    users |o--o{ activities : "created (created_by_id)"
    users |o--o{ notes : "authored (author_id)"
    accounts ||--o{ contacts : has
    accounts ||--o{ opportunities : has
    accounts |o--o{ activities : has
    accounts |o--o{ notes : has
    accounts ||--o{ document_chunks : has
    contacts |o--o{ activities : has
    leads |o--o{ activities : has
    leads |o--o{ notes : has
    leads ||--o{ lead_predictions : has
    opportunities |o--o{ activities : has
    opportunities |o--o{ notes : has
```

`||` = required parent (`NOT NULL` FK), `|o` = optional parent (nullable FK).

| Parent | Child | FK column | Required |
|---|---|---|---|
| User | Account (owner) | `accounts.owner_id` | no |
| User | Lead (owner) | `leads.owner_id` | no |
| User | Opportunity (owner) | `opportunities.owner_id` | no |
| User | Activity (creator) | `activities.created_by_id` | no |
| User | Note (author) | `notes.author_id` | no |
| Account | Contact | `contacts.account_id` | yes |
| Account | Opportunity | `opportunities.account_id` | yes |
| Account | Activity | `activities.account_id` | no |
| Account | Note | `notes.account_id` | no |
| Account | DocumentChunk | `document_chunks.account_id` | yes |
| Contact | Activity | `activities.contact_id` | no |
| Lead | Activity | `activities.lead_id` | no |
| Lead | Note | `notes.lead_id` | no |
| Lead | LeadPrediction | `lead_predictions.lead_id` | yes |
| Opportunity | Activity | `activities.opportunity_id` | no |
| Opportunity | Note | `notes.opportunity_id` | no |

Notes:

- Ownership FKs are nullable so CRM records survive user deactivation or removal.
- An activity may reference several CRM entities at once (e.g. a meeting with an account,
  opportunity and contact). There is no "exactly one parent" rule, but there must be
  **at least one** parent, and the parents must agree on one account
  ([Service-layer rules](#service-layer-rules)).
- Lead-only activities and notes are allowed.
- `document_chunks.source_id` is a polymorphic reference (no FK); see
  [document_chunks](#document_chunks).
- A converted lead is not linked to the account/opportunity it became (deferred).

---

## Tables

"—" means no default. `now()` is a server default.

### users

Application users. Defined now; authentication is implemented later.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `UUID` | no | app `uuid4` | PK |
| `email` | `VARCHAR(320)` | no | — | Unique. Canonical lowercase, normalized by the service layer. |
| `password_hash` | `VARCHAR(255)` | no | — | Argon2id hash via pwdlib (97 characters). Never log or return. Deferred in the ORM: loaded only when login verifies a password. |
| `full_name` | `VARCHAR(255)` | no | — | |
| `role` | `VARCHAR(32)` enum `user_role` | no | — | `ADMIN`, `SALES_REP` |
| `is_active` | `BOOLEAN` | no | `true` | Deactivate instead of delete. |
| `created_at` | `TIMESTAMPTZ` | no | `now()` | |
| `updated_at` | `TIMESTAMPTZ` | no | `now()` | `onupdate` (name, role, activation changes) |

Constraints: `pk_users`, `uq_users_email`, `ck_users_user_role`.

Indexes: `uq_users_email` is the index; no separate index.

### accounts

Customer companies.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `UUID` | no | app `uuid4` | PK |
| `name` | `VARCHAR(255)` | no | — | **Not** unique. |
| `industry` | `VARCHAR(255)` | yes | — | |
| `employee_count` | `INTEGER` | yes | — | `>= 0` |
| `annual_revenue` | `NUMERIC(16,2)` | yes | — | USD, `Decimal`, `>= 0` |
| `website` | `VARCHAR(2048)` | yes | — | |
| `owner_id` | `UUID` → `users.id` | yes | — | `ON DELETE SET NULL` |
| `created_at` | `TIMESTAMPTZ` | no | `now()` | |
| `updated_at` | `TIMESTAMPTZ` | no | `now()` | `onupdate` |

Constraints: `pk_accounts`, `fk_accounts_owner_id_users`,
`ck_accounts_employee_count_non_negative`, `ck_accounts_annual_revenue_non_negative`.

Indexes: `ix_accounts_owner_id`.

### contacts

People at an account.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `UUID` | no | app `uuid4` | PK |
| `account_id` | `UUID` → `accounts.id` | no | — | `ON DELETE RESTRICT` |
| `first_name` | `VARCHAR(255)` | no | — | |
| `last_name` | `VARCHAR(255)` | no | — | |
| `email` | `VARCHAR(320)` | yes | — | **Not** unique. |
| `phone` | `VARCHAR(50)` | yes | — | Stored as text, not a number. |
| `job_title` | `VARCHAR(255)` | yes | — | |
| `created_at` | `TIMESTAMPTZ` | no | `now()` | |
| `updated_at` | `TIMESTAMPTZ` | no | `now()` | `onupdate` |

Constraints: `pk_contacts`, `fk_contacts_account_id_accounts`.

Indexes: `ix_contacts_account_id`.

### leads

Prospects not yet tied to an account. The ML model predicts lead conversion.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `UUID` | no | app `uuid4` | PK |
| `company_name` | `VARCHAR(255)` | no | — | |
| `contact_name` | `VARCHAR(255)` | no | — | |
| `email` | `VARCHAR(320)` | yes | — | **Not** unique. |
| `lead_source` | `VARCHAR(32)` enum `lead_source` | no | — | `WEB`, `REFERRAL`, `EVENT`, `OUTBOUND`, `PARTNER`, `OTHER` |
| `industry` | `VARCHAR(255)` | yes | — | |
| `company_size` | `INTEGER` | yes | — | `>= 0` |
| `estimated_value` | `NUMERIC(14,2)` | yes | — | USD, `Decimal`, `>= 0` |
| `status` | `VARCHAR(32)` enum `lead_status` | no | — | `NEW`, `WORKING`, `QUALIFIED`, `CONVERTED`, `DISQUALIFIED`. **Never an ML feature.** |
| `owner_id` | `UUID` → `users.id` | yes | — | `ON DELETE SET NULL`. Not an ML feature in V1. |
| `created_at` | `TIMESTAMPTZ` | no | `now()` | |
| `updated_at` | `TIMESTAMPTZ` | no | `now()` | `onupdate` |
| `converted_at` | `TIMESTAMPTZ` | yes | — | ML label source. **Never an ML feature.** |

Constraints: `pk_leads`, `fk_leads_owner_id_users`, `ck_leads_lead_source`,
`ck_leads_lead_status`, `ck_leads_company_size_non_negative`,
`ck_leads_estimated_value_non_negative`, and

```sql
CONSTRAINT ck_leads_converted_requires_converted_at
    CHECK (status <> 'CONVERTED' OR converted_at IS NOT NULL)
```

Conversion rule:

- `status = CONVERTED` ⇒ `converted_at` is populated (enforced by the CHECK above).
- The reverse is intentionally **not** enforced: other statuses normally have
  `converted_at = NULL`, but the database allows a value so future workflows stay flexible.
- The API cannot create converted leads or set `converted_at` today (see service rule 2);
  when a conversion workflow exists it must also keep `converted_at >= created_at`.

Indexes: `ix_leads_owner_id`, `ix_leads_status`, `ix_leads_lead_source`.

### opportunities

Deals against an account.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `UUID` | no | app `uuid4` | PK |
| `account_id` | `UUID` → `accounts.id` | no | — | `ON DELETE RESTRICT` |
| `name` | `VARCHAR(255)` | no | — | |
| `stage` | `VARCHAR(32)` enum `opportunity_stage` | no | — | `PROSPECTING`, `QUALIFICATION`, `PROPOSAL`, `NEGOTIATION`, `CLOSED_WON`, `CLOSED_LOST` |
| `amount` | `NUMERIC(14,2)` | yes | — | USD, `Decimal`, `>= 0` |
| `expected_close_date` | `DATE` | yes | — | |
| `description` | `TEXT` | yes | — | |
| `owner_id` | `UUID` → `users.id` | yes | — | `ON DELETE SET NULL` |
| `created_at` | `TIMESTAMPTZ` | no | `now()` | |
| `updated_at` | `TIMESTAMPTZ` | no | `now()` | `onupdate` |

Constraints: `pk_opportunities`, `fk_opportunities_account_id_accounts`,
`fk_opportunities_owner_id_users`, `ck_opportunities_opportunity_stage`,
`ck_opportunities_amount_non_negative`.

Indexes: `ix_opportunities_account_id`, `ix_opportunities_owner_id`,
`ix_opportunities_stage`.

### activities

Calls, emails, meetings, demos and follow-ups. May reference several CRM entities.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `UUID` | no | app `uuid4` | PK |
| `account_id` | `UUID` → `accounts.id` | yes | — | `ON DELETE RESTRICT` |
| `lead_id` | `UUID` → `leads.id` | yes | — | `ON DELETE RESTRICT` |
| `opportunity_id` | `UUID` → `opportunities.id` | yes | — | `ON DELETE RESTRICT` |
| `contact_id` | `UUID` → `contacts.id` | yes | — | `ON DELETE RESTRICT` |
| `created_by_id` | `UUID` → `users.id` | yes | — | `ON DELETE SET NULL` |
| `activity_type` | `VARCHAR(32)` enum `activity_type` | no | — | `CALL`, `EMAIL`, `MEETING`, `DEMO`, `FOLLOW_UP` |
| `direction` | `VARCHAR(32)` enum `activity_direction` | no | — | `INBOUND`, `OUTBOUND`, `INTERNAL` |
| `subject` | `VARCHAR(255)` | yes | — | |
| `body` | `TEXT` | yes | — | |
| `occurred_at` | `TIMESTAMPTZ` | no | — | When it happened (business time). ML time filter. |
| `created_at` | `TIMESTAMPTZ` | no | `now()` | When it was recorded. |

Constraints: `pk_activities`, `fk_activities_account_id_accounts`,
`fk_activities_lead_id_leads`, `fk_activities_opportunity_id_opportunities`,
`fk_activities_contact_id_contacts`, `fk_activities_created_by_id_users`,
`ck_activities_activity_type`, `ck_activities_activity_direction`, and

```sql
CONSTRAINT ck_activities_at_least_one_parent
    CHECK (account_id IS NOT NULL OR lead_id IS NOT NULL
           OR opportunity_id IS NOT NULL OR contact_id IS NOT NULL)
```

Cross-account consistency of the parents is enforced by the service layer
([Service-layer rules](#service-layer-rules)).

Indexes: `ix_activities_account_id`, `ix_activities_lead_id`,
`ix_activities_opportunity_id`, `ix_activities_contact_id`,
`ix_activities_created_by_id`, `ix_activities_occurred_at`.

### notes

Free-text notes on accounts, leads and opportunities.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `UUID` | no | app `uuid4` | PK |
| `account_id` | `UUID` → `accounts.id` | yes | — | `ON DELETE RESTRICT` |
| `lead_id` | `UUID` → `leads.id` | yes | — | `ON DELETE RESTRICT` |
| `opportunity_id` | `UUID` → `opportunities.id` | yes | — | `ON DELETE RESTRICT` |
| `author_id` | `UUID` → `users.id` | yes | — | `ON DELETE SET NULL` |
| `body` | `TEXT` | no | — | |
| `created_at` | `TIMESTAMPTZ` | no | `now()` | ML time filter. |
| `updated_at` | `TIMESTAMPTZ` | no | `now()` | `onupdate` |

Constraints: `pk_notes`, `fk_notes_account_id_accounts`, `fk_notes_lead_id_leads`,
`fk_notes_opportunity_id_opportunities`, `fk_notes_author_id_users`, and

```sql
CONSTRAINT ck_notes_at_least_one_parent
    CHECK (account_id IS NOT NULL OR lead_id IS NOT NULL OR opportunity_id IS NOT NULL)
```

Indexes: `ix_notes_account_id`, `ix_notes_lead_id`, `ix_notes_opportunity_id`,
`ix_notes_author_id`.

### lead_predictions

Scores produced by the lead conversion model. Append-only.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `UUID` | no | app `uuid4` | PK |
| `lead_id` | `UUID` → `leads.id` | no | — | `ON DELETE CASCADE` (derived data) |
| `prediction_timestamp` | `TIMESTAMPTZ` | no | — | Feature cut-off time `T`. Label: converted during `T` through `T + 30 days`. |
| `conversion_probability` | `DOUBLE PRECISION` | no | — | Probability, not money. `0..1`. |
| `lead_tier` | `VARCHAR(32)` enum `lead_tier` | no | — | `HOT`, `WARM`, `COLD`. Named `lead_tier`, **not** `risk_tier`: high conversion probability is not high risk. |
| `model_name` | `VARCHAR(128)` | no | — | MLflow registered model name. |
| `model_version` | `VARCHAR(64)` | no | — | MLflow model version. |
| `features` | `JSONB` | no | — | The exact model input for this score (wire form: strings, integers, floats, `null` for missing), for audit and debugging. |
| `inference_ms` | `DOUBLE PRECISION` | yes | — | Model inference time in milliseconds (`>= 0`), excluding feature computation and storage. |
| `created_at` | `TIMESTAMPTZ` | no | `now()` | |

Constraints: `pk_lead_predictions`, `fk_lead_predictions_lead_id_leads`,
`ck_lead_predictions_lead_tier`, and

```sql
CONSTRAINT ck_lead_predictions_conversion_probability_range
    CHECK (conversion_probability >= 0 AND conversion_probability <= 1),
CONSTRAINT uq_lead_predictions_lead_id_timestamp_model
    UNIQUE (lead_id, prediction_timestamp, model_name, model_version)
```

The unique constraint makes batch/Airflow retries idempotent: the same model version
cannot score the same lead twice for the same `prediction_timestamp`. (The descriptive
name `uq_lead_predictions_lead_id_prediction_timestamp_model_name_model_version` is 73
characters, over PostgreSQL's 63-character limit, hence the shorter name.)

Indexes:

- `uq_lead_predictions_lead_id_timestamp_model` creates a B-tree index whose leading
  columns are `(lead_id, prediction_timestamp)`. It serves "latest prediction for this
  lead" (`WHERE lead_id = :id ORDER BY prediction_timestamp DESC`) and `lead_id`-only
  lookups (the FK), so there is no separate `lead_id` or `(lead_id, prediction_timestamp)`
  index.
- `ix_lead_predictions_prediction_timestamp` for batch/time-window queries.

### document_chunks

Embedded CRM text for account-scoped semantic search. Rebuilt by the indexer
(`python -m app.cli.index_crm`); never edited through the API.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `UUID` | no | app `uuid4` | PK |
| `account_id` | `UUID` → `accounts.id` | no | — | `ON DELETE CASCADE`. Isolation key; see [RAG isolation rule](#rag-isolation-rule). |
| `source_type` | `VARCHAR(32)` enum `document_source_type` | no | — | `ACCOUNT`, `ACTIVITY`, `NOTE`, `OPPORTUNITY` |
| `source_id` | `UUID` | no | — | ID of the source row in the table named by `source_type`. **No FK** (polymorphic). |
| `entity_type` | `VARCHAR(32)` enum `document_entity_type` | no | — | CRM record the text is about: `ACCOUNT`, `CONTACT`, `OPPORTUNITY`. |
| `entity_id` | `UUID` | no | — | ID of that record (no FK). |
| `chunk_index` | `INTEGER` | no | — | 0-based position within the source document. `>= 0` |
| `content` | `TEXT` | no | — | Chunk text returned by search (and later cited by answers). |
| `embedding` | `VECTOR(384)` | no | — | Unit-length embedding of `content`; chunks are embedded when indexed. |
| `embedding_model` | `VARCHAR(255)` | no | — | Model that produced `embedding`, e.g. `sentence-transformers/all-MiniLM-L6-v2`. Search only uses chunks of the configured model. |
| `source_sha256` | `VARCHAR(64)` | no | — | Hash of the indexed input (account, entity, timestamp, metadata, chunk texts, model); unchanged sources are not re-embedded. |
| `metadata` | `JSONB` | no | `'{}'::jsonb` | e.g. `activity_type`, `direction`, `stage`, `chunk_count`. |
| `occurred_at` | `TIMESTAMPTZ` | no | — | Business timestamp of the source, returned with every search result. |
| `created_at` | `TIMESTAMPTZ` | no | `now()` | When the chunk was indexed. |

`occurred_at` comes from the source:

| `source_type` | `occurred_at` | `entity_type` / `entity_id` |
|---|---|---|
| `ACTIVITY` | `activities.occurred_at` | its opportunity, else its contact, else its account |
| `NOTE` | `notes.created_at` | its opportunity, else its account |
| `OPPORTUNITY` | `opportunities.updated_at` | the opportunity |
| `ACCOUNT` | `accounts.updated_at` | the account |

Indexed text per source (V1):

| `source_type` | Text |
|---|---|
| `ACTIVITY` | `"{Type} ({direction}): {subject}"` and the body on the next line |
| `NOTE` | the body |
| `OPPORTUNITY` | name, stage, amount, expected close date and description, one per line |
| `ACCOUNT` | name, industry, employee count and website, one per line |

**Chunking (V1):** a source is one chunk unless its text exceeds the embedding model's
input limit (`max_seq_length` minus the two special tokens: 254 tokens for MiniLM),
measured with the model's own tokenizer. Longer text is cut at whitespace into windows of
at most that many tokens, overlapping by about 32 tokens, so the model never silently
truncates a chunk. Current CRM text (tens of characters) is always a single chunk.

Constraints: `pk_document_chunks`, `fk_document_chunks_account_id_accounts`,
`ck_document_chunks_document_source_type`, `ck_document_chunks_document_entity_type`,
`ck_document_chunks_chunk_index_non_negative`, and

```sql
CONSTRAINT uq_document_chunks_source_type_source_id_chunk_index
    UNIQUE (source_type, source_id, chunk_index)
```

(explicit name; the global naming convention is not changed). Its index also serves "find
all chunks for a source" during re-indexing, so `source_id` has no separate index.

Embedding dimension:

- `VECTOR(384)` is fixed for V1 and matches `EMBEDDING_DIMENSION=384`
  (`sentence-transformers/all-MiniLM-L6-v2`); the settings reject any other value and the
  provider checks every vector.
- Switching to a model with a different dimension requires a schema migration and
  re-embedding all existing chunks.

Implementation notes:

- `metadata` is reserved for the `MetaData` instance on Declarative classes, so the column
  `metadata` is mapped to the attribute `chunk_metadata`.
- The migration runs `CREATE EXTENSION IF NOT EXISTS vector` (RDS needs it; locally
  `docker/postgres/init/001-init.sql` already creates it) and imports `pgvector.sqlalchemy`.

Indexes: `ix_document_chunks_account_id`. **No vector similarity (ANN) index in V1**:
search is an exact cosine scan restricted to one account, which is correct and fast at the
current chunk counts per account. An HNSW index with a filter can return fewer than
`limit` rows; add it with pgvector ≥ 0.8 iterative scans if per-account volumes grow.

### Indexing

`PYTHONPATH=backend python -m app.cli.index_crm [--account-id UUID]` (uses `DATABASE_URL`
and the `EMBEDDING_*` settings). It is rerunnable and idempotent:

- Sources are read in pages of 500; each page is one transaction and one batched
  embedding call (identical texts are embedded once).
- Unchanged sources (same `source_sha256`) are skipped without embedding.
- A new or changed source has all its chunks replaced, whatever account they were under.
- Chunks of deleted sources, or of sources that no longer resolve to an account, are
  removed. With `--account-id`, only that account's chunks are considered.
- The unique `(source_type, source_id, chunk_index)` constraint rules out duplicates.

Indexing is not triggered by CRM writes in V1: search reflects the last indexing run.

---

## Enum values

| Python class | DB name | Column(s) | Values |
|---|---|---|---|
| `UserRole` | `user_role` | `users.role` | `ADMIN`, `SALES_REP` |
| `LeadSource` | `lead_source` | `leads.lead_source` | `WEB`, `REFERRAL`, `EVENT`, `OUTBOUND`, `PARTNER`, `OTHER` |
| `LeadStatus` | `lead_status` | `leads.status` | `NEW`, `WORKING`, `QUALIFIED`, `CONVERTED`, `DISQUALIFIED` |
| `OpportunityStage` | `opportunity_stage` | `opportunities.stage` | `PROSPECTING`, `QUALIFICATION`, `PROPOSAL`, `NEGOTIATION`, `CLOSED_WON`, `CLOSED_LOST` |
| `ActivityType` | `activity_type` | `activities.activity_type` | `CALL`, `EMAIL`, `MEETING`, `DEMO`, `FOLLOW_UP` |
| `ActivityDirection` | `activity_direction` | `activities.direction` | `INBOUND`, `OUTBOUND`, `INTERNAL` |
| `LeadTier` | `lead_tier` | `lead_predictions.lead_tier` | `HOT`, `WARM`, `COLD` |
| `DocumentSourceType` | `document_source_type` | `document_chunks.source_type` | `ACCOUNT`, `ACTIVITY`, `NOTE`, `OPPORTUNITY` |

All are non-native (`VARCHAR(32)` + CHECK); see [Conventions](#enums).

---

## Foreign keys and delete behavior

Principle: references to users are cleared, core CRM history is protected, derived data
follows its parent. Delete actions are declared explicitly (`ondelete=...`) so they are
visible in migrations.

| FK | Null | On delete | Reason |
|---|---|---|---|
| `accounts.owner_id` → `users.id` | yes | `SET NULL` | CRM records survive user removal. |
| `leads.owner_id` → `users.id` | yes | `SET NULL` | Same. |
| `opportunities.owner_id` → `users.id` | yes | `SET NULL` | Same. |
| `activities.created_by_id` → `users.id` | yes | `SET NULL` | Same. |
| `notes.author_id` → `users.id` | yes | `SET NULL` | Same. |
| `contacts.account_id` → `accounts.id` | no | `RESTRICT` | Protect CRM history. |
| `opportunities.account_id` → `accounts.id` | no | `RESTRICT` | Same. |
| `activities.account_id` → `accounts.id` | yes | `RESTRICT` | Same; `SET NULL` could also violate `ck_activities_at_least_one_parent`. |
| `activities.lead_id` → `leads.id` | yes | `RESTRICT` | Same. |
| `activities.opportunity_id` → `opportunities.id` | yes | `RESTRICT` | Same. |
| `activities.contact_id` → `contacts.id` | yes | `RESTRICT` | Same. |
| `notes.account_id` → `accounts.id` | yes | `RESTRICT` | Same. |
| `notes.lead_id` → `leads.id` | yes | `RESTRICT` | Same. |
| `notes.opportunity_id` → `opportunities.id` | yes | `RESTRICT` | Same. |
| `lead_predictions.lead_id` → `leads.id` | no | `CASCADE` | Derived data. |
| `document_chunks.account_id` → `accounts.id` | no | `CASCADE` | Derived data; stale chunks are a retrieval leak risk. |

Consequence: deleting an account or lead that has CRM history fails until that history is
removed explicitly. This is intended while V1 has no soft deletes and no delete endpoints.

---

## Check constraints

| Constraint | Table | Expression | Phase |
|---|---|---|---|
| `ck_users_user_role` | users | enum values | 1 |
| `ck_accounts_employee_count_non_negative` | accounts | `employee_count >= 0` | 1 |
| `ck_accounts_annual_revenue_non_negative` | accounts | `annual_revenue >= 0` | 1 |
| `ck_leads_lead_source` | leads | enum values | 1 |
| `ck_leads_lead_status` | leads | enum values | 1 |
| `ck_leads_company_size_non_negative` | leads | `company_size >= 0` | 1 |
| `ck_leads_estimated_value_non_negative` | leads | `estimated_value >= 0` | 1 |
| `ck_leads_converted_requires_converted_at` | leads | `status <> 'CONVERTED' OR converted_at IS NOT NULL` | 1 |
| `ck_opportunities_opportunity_stage` | opportunities | enum values | 1 |
| `ck_opportunities_amount_non_negative` | opportunities | `amount >= 0` | 1 |
| `ck_activities_activity_type` | activities | enum values | 2 |
| `ck_activities_activity_direction` | activities | enum values | 2 |
| `ck_activities_at_least_one_parent` | activities | at least one of `account_id`, `lead_id`, `opportunity_id`, `contact_id` | 2 |
| `ck_notes_at_least_one_parent` | notes | at least one of `account_id`, `lead_id`, `opportunity_id` | 2 |
| `ck_lead_predictions_lead_tier` | lead_predictions | enum values | 3 |
| `ck_lead_predictions_conversion_probability_range` | lead_predictions | `conversion_probability >= 0 AND conversion_probability <= 1` | 3 |
| `ck_lead_predictions_inference_ms_non_negative` | lead_predictions | `inference_ms >= 0` | 3 |
| `ck_document_chunks_document_source_type` | document_chunks | enum values | 4 |
| `ck_document_chunks_document_entity_type` | document_chunks | enum values | 4 |
| `ck_document_chunks_chunk_index_non_negative` | document_chunks | `chunk_index >= 0` | 4 |

Non-negative checks on nullable columns pass for `NULL` (SQL CHECK semantics), which is
intended.

---

## Index summary

Every foreign key is indexed (directly or as the leading column of a composite index or
unique constraint). Primary keys are indexed implicitly.

| Table | Index / unique constraint | Columns | Purpose |
|---|---|---|---|
| users | `uq_users_email` (unique) | `email` | Login lookup, uniqueness |
| accounts | `ix_accounts_owner_id` | `owner_id` | "My accounts", FK |
| contacts | `ix_contacts_account_id` | `account_id` | Contacts of an account, FK |
| leads | `ix_leads_owner_id` | `owner_id` | "My leads", FK |
| leads | `ix_leads_status` | `status` | Filter by status |
| leads | `ix_leads_lead_source` | `lead_source` | Filter by source |
| opportunities | `ix_opportunities_account_id` | `account_id` | Opportunities of an account, FK |
| opportunities | `ix_opportunities_owner_id` | `owner_id` | "My opportunities", FK |
| opportunities | `ix_opportunities_stage` | `stage` | Pipeline by stage |
| activities | `ix_activities_account_id` | `account_id` | Account timeline, FK |
| activities | `ix_activities_lead_id` | `lead_id` | Lead timeline, ML features, FK |
| activities | `ix_activities_opportunity_id` | `opportunity_id` | Opportunity timeline, FK |
| activities | `ix_activities_contact_id` | `contact_id` | Contact timeline, FK |
| activities | `ix_activities_created_by_id` | `created_by_id` | FK |
| activities | `ix_activities_occurred_at` | `occurred_at` | Time windows, ML cut-offs |
| notes | `ix_notes_account_id` | `account_id` | Account notes, FK |
| notes | `ix_notes_lead_id` | `lead_id` | Lead notes, FK |
| notes | `ix_notes_opportunity_id` | `opportunity_id` | Opportunity notes, FK |
| notes | `ix_notes_author_id` | `author_id` | FK |
| lead_predictions | `ix_lead_predictions_prediction_timestamp` | `prediction_timestamp` | Batch/time queries |
| lead_predictions | `uq_lead_predictions_lead_id_timestamp_model` (unique) | `lead_id, prediction_timestamp, model_name, model_version` | Idempotent scoring; latest prediction for a lead; FK |
| document_chunks | `ix_document_chunks_account_id` | `account_id` | RAG pre-filter, FK |
| document_chunks | `uq_document_chunks_source_type_source_id_chunk_index` (unique) | `source_type, source_id, chunk_index` | Idempotent indexing, re-index lookups |

Not indexed on purpose: `contacts.email`, `leads.email`, `accounts.name`, `embedding`
(no vector index yet).

---

## Service-layer rules

Rules the database does not (or cannot) enforce. Services must implement them, with tests,
when the corresponding phase is built.

1. **User email normalization** (user creation/authentication): trim and lowercase
   `users.email` before every insert and lowercase the input before every lookup
   (`app.core.emails.normalize_user_email`, used by `UserService` and login). No database
   CHECK; the plain `UNIQUE (email)` stays.
2. **Lead status** (current behavior): there is no lead update or conversion workflow
   yet. `POST /leads` accepts every status except `CONVERTED` (rejected with `422`) and
   never accepts `converted_at`, so leads created through the API always have
   `converted_at = NULL`. `ck_leads_converted_requires_converted_at` is the database
   backstop for any other write path (e.g. synthetic data). A future conversion workflow
   must set `converted_at` (with `converted_at >= created_at`) in the same write that
   sets `status = CONVERTED`.
3. **Cross-account consistency for activities and notes** (Phase 2). A SQL CHECK cannot
   enforce cross-table rules, so before insert/update the service resolves the account of
   every referenced account-scoped entity and rejects the write if they disagree:
   - `account_id` + `opportunity_id`: `opportunity.account_id == account_id`
   - `account_id` + `contact_id`: `contact.account_id == account_id`
   - `opportunity_id` + `contact_id`: `opportunity.account_id == contact.account_id`
   - Notes: `account_id` + `opportunity_id` as above.
   - `lead_id` has no account and is not part of this check. A lead may be combined with
     account-scoped references; lead-only activities/notes are allowed.
   - **Canonical `account_id`:** when `account_id` is omitted but an account-scoped
     reference is given, the service stores the derived account (activity: from
     `opportunity_id` or `contact_id`; note: from `opportunity_id`). So `account_id` is
     NULL only for lead-only records, and account timelines and RAG can filter on
     `account_id` alone. `lead_id` never implies an account.
   - Implemented in `backend/app/services/crm_references.py`.
4. **Activity and note authors**: `activities.created_by_id` and `notes.author_id` are
   set to the authenticated user by the API and are never taken from request input. The
   columns stay nullable (`ON DELETE SET NULL`) so records survive user deletion.
5. **`activities.occurred_at`** (Phase 2) is when the activity actually happened: it must
   be timezone-aware and not more than 5 minutes in the future (clock-skew allowance).
6. **Document chunk maintenance** (Phase 4): the indexer rebuilds or deletes a source's
   chunks when the source changes, moves account, or is deleted, and populates
   `occurred_at`, `embedding_model` and `source_sha256`
   (`backend/app/services/crm_indexing_service.py`).

---

## RAG isolation rule

RevIntel V1 RAG is **account-scoped**. Retrieval must never return another customer's data.

`document_chunks.account_id` is stored on every chunk **even though** the source record
may already reference an account. Retrieval filters on it **before** vector similarity
ranking, so chunks from other accounts are never candidates.

The query (`backend/app/repositories/semantic_search.py`), one statement:

```sql
SELECT account_id, source_type, source_id, content, occurred_at,
       1 - (embedding <=> :query_embedding) AS similarity   -- cosine similarity
FROM document_chunks
WHERE account_id = :account_id
  AND embedding_model = :embedding_model
ORDER BY embedding <=> :query_embedding, id
LIMIT :k;
```

The account predicate is in the same statement that ranks and limits; results are never
retrieved across accounts and filtered afterwards.

Rules:

- `account_id` is a required argument of the retrieval repository; there is no unscoped
  search.
- Each retrieved record keeps `source_type`, `source_id`, `account_id`, a timestamp
  (`occurred_at`, never null) and the text (`content`). Answers cite `source_type` +
  `source_id`.
- `account_id` is derived from the source when indexing:

  | `source_type` | `account_id` comes from |
  |---|---|
  | `ACCOUNT` | the account itself (`source_id = account_id`) |
  | `OPPORTUNITY` | `opportunities.account_id` |
  | `ACTIVITY` | `activities.account_id`, else its opportunity's or contact's account |
  | `NOTE` | `notes.account_id`, else its opportunity's account |

  All of a source's account references must agree on one account; a source whose
  references conflict is skipped. The API already stores the canonical account (service
  rule 3); resolving through the opportunity/contact also covers rows written without it.
- Every indexed document has a resolved `account_id`. Activities and notes that resolve to
  no account are **not indexed** in V1; leads have no account in the schema, so lead-only
  activities and notes are always skipped. `document_chunks.account_id` stays `NOT NULL`.
- Only chunks whose `embedding_model` matches the configured `EMBEDDING_MODEL` should be
  queried; mismatches are stale and must be re-embedded.
- V1 has no vector index (exact per-account scan). If one is added, filtered approximate
  search needs care (pgvector ≥ 0.8 iterative index scans).

---

## ML point-in-time rules

Prediction definition: at timestamp `T` (`prediction_timestamp`), predict whether the lead
converts in `(T, T + 30 days]`. The label comes from `leads.converted_at`.

Features may only use data that existed in the CRM at or before `T`.

| Data | Rule |
|---|---|
| `leads.status` | **Never** a feature. |
| `leads.converted_at` | **Never** a feature (label source only). |
| `leads.owner_id` | Not a feature in V1. |
| `leads.lead_source`, `industry`, `company_size`, `estimated_value` | Allowed in V1. The V1 synthetic generator keeps them unchanged after lead creation, so the current value equals the value at `T`. |
| `activities` | Only rows known at `T`: `created_at <= T AND occurred_at <= T`. Feature windows are then placed by `occurred_at`. |
| `activities.created_at` | Availability only (was the row in the CRM at `T`?); **never** a feature value. |
| `notes` | Only rows with `created_at <= T`. |
| Any other event | Only if its timestamp is `<= T`. |
| Any other mutable field | Only if its point-in-time value is guaranteed. |
| `lead_predictions.features` | Records what the model actually saw at scoring time. |

V1 has no lead history/snapshot table. If lead attributes become mutable in real use, the
allowed lead-level features above are no longer point-in-time safe (see
[Deferred / future considerations](#deferred--future-considerations)).

---

## Synthetic CRM data

`scripts/generate_crm_data.py` (logic in `ml/synthetic/`) generates deterministic, raw CRM
data for model development: users, accounts, contacts, leads and activities. It creates
**no features, labels or prediction timestamps** and needs no schema change. Notes and
opportunities are not generated yet, so the first feature set must not use notes (no
`notes_count_30d`).

### Target database

The dataset lives in a dedicated `revintel_ml` database (integration tests truncate
`revintel_test`). Create it once:

```bash
docker exec revintel-postgres createdb -U revintel revintel_ml
docker exec revintel-postgres psql -U revintel -d revintel_ml -c "CREATE EXTENSION IF NOT EXISTS vector"
DATABASE_URL=postgresql+psycopg://revintel:revintel@localhost:5432/revintel_ml alembic upgrade head
```

The target comes **only** from `--database-url` or `SYNTHETIC_DATABASE_URL`; `DATABASE_URL`
and `.env` are deliberately ignored. Before writing, the generator refuses:

- database names that don't end in `_test` or `_ml` (the dev `revintel` is always refused);
- a schema that is not at the Alembic head;
- a database containing `admin@example.com` or the account `RevIntel Co`;
- a non-empty database, unless `--reset` is given. `--reset` truncates the CRM tables of
  that guarded database in the same transaction as the insert.

Everything is written with batched Core inserts in a single transaction, so a failure
leaves nothing behind.

### Usage

```bash
export SYNTHETIC_DATABASE_URL=postgresql+psycopg://revintel:revintel@localhost:5432/revintel_ml
python scripts/generate_crm_data.py --seed 42 --accounts 5000 --contacts 20000 \
  --leads 30000 --activities 100000 [--reset]
python scripts/generate_crm_data.py --seed 42 --dry-run   # generate and summarise only
```

Other options: `--reps`, `--admins`, `--as-of` (default `2026-10-01T00:00:00Z`),
`--lead-history-days` (540), `--max-lead-age-days` (180), `--batch-size` (5000),
`--manifest` (default `data/raw/synthetic_crm_manifest.json`, gitignored).

The same seed and arguments produce byte-identical rows (deterministic UUIDs; every
timestamp derives from `--as-of`, never the wall clock) for the pinned numpy/Faker
versions. The manifest records the configuration, library versions, row counts, summary
statistics, sanity counters and `dataset_sha256`, the dataset version to log in MLflow.

### How the data is generated

- **Leads** get a hidden quality score from `lead_source`, `company_size`,
  `estimated_value` and noise. The score is never stored.
- A **forward, day-by-day simulation** creates lead activities (`lead_id` only) whose rate,
  type and direction depend on that score. A recency-decayed engagement level (inbound
  replies, meetings and demos weigh most) and the score drive daily conversion and
  disqualification hazards. Outcomes depend only on the past, and **no lead activity
  occurs after conversion or disqualification**.
- **Account/contact activities** fill the rest of `--activities` exactly (about 30%). If the
  lead simulation alone exceeds `--activities`, generation fails; it is never re-run or
  recalibrated to hit a count.
- Activity text depends only on type and direction and never mentions the outcome.
- Synthetic users (`*@synthetic.example.com`) share a valid Argon2id hash of a discarded
  random secret, so nobody can log in as them.

| Final status | `converted_at` | `updated_at` |
|---|---|---|
| CONVERTED | conversion time | `converted_at` |
| DISQUALIFIED | NULL | disqualification time (after every lead activity) |
| NEW / WORKING / QUALIFIED | NULL | last simulated status change, or `created_at` |

### Terminal-time convention (synthetic data only)

The schema has no lead status history. In this dataset a disqualified lead's exit time is
its `updated_at`. This is a **synthetic-dataset convention, not a production
assumption**, and is recorded in the manifest (`conventions.terminal_time`). All code must
go through `ml/synthetic/conventions.py::lead_terminal_time(status, converted_at,
updated_at)` so it can later be replaced by a real status-history source.

### Observation-end convention (synthetic data only)

The generator simulates each lead for at most `max_lead_age_days` (180) after
`created_at`. A lead still open at that point is censored, not a negative: nothing after
`created_at + max_lead_age_days` is observable. Code uses
`ml/synthetic/conventions.py::lead_observation_end(created_at, max_lead_age_days)`.

Prediction timestamps are not stored; the feature pipeline derives them (see
[Lead conversion features](#lead-conversion-features)).

### Limitations

- Leads are not linked to accounts or opportunities (no conversion linkage in the schema).
- No notes, opportunities or lead predictions are generated.
- Lead attributes never change after creation; real data would need history tables.
- Leads still open at `as_of` are right-censored; the simulation stops 180 days after
  creation.
- Distributions are plausible, not fitted to real CRM data.

---

## Lead conversion features

`scripts/build_lead_features.py` (logic in `ml/features/`) turns the synthetic CRM dataset
into one row per lead snapshot in `data/processed/lead_features.parquet` (gitignored),
plus `lead_features_summary.json`. It trains nothing and never writes to the database.

```bash
export SYNTHETIC_DATABASE_URL=postgresql+psycopg://revintel:revintel@localhost:5432/revintel_ml
python scripts/build_lead_features.py [--manifest PATH] [--database-url URL] [--output PATH]
```

The source comes only from `--database-url` or `SYNTHETIC_DATABASE_URL` (name must end in
`_ml` or `_test`; `DATABASE_URL` is ignored). Both reads run in one `REPEATABLE READ`,
`READ ONLY` transaction. The build refuses a database whose lead, activity and
lead-activity counts differ from the manifest, which also supplies `as_of`,
`max_lead_age_days` and the `dataset_sha256` recorded in the output.

### Modules

| Module | Role |
|---|---|
| `feature_spec.py` | Single source of truth: feature definitions, windows, input categories, `compute_features()` (shared by offline and online), spec fingerprint. |
| `sources.py` | Read-only SQL extraction shared by both paths. |
| `snapshots.py` | Snapshot policy and label; the only reader of `status`, `converted_at`, `updated_at`. |
| `build_features.py` | Offline build: manifest check, extraction, snapshots, features, validation, parquet. |
| `validation.py` | Hard invariants (nothing is written on failure) and build statistics. |
| `online_features.py` | Features for one lead at time `at` via the same SQL and `compute_features()`. |

### Snapshot policy

For each lead, `T = created_at + k × 7 days` (`k ≥ 1`). A snapshot exists only if:

1. `T + 30d <= as_of` (the label window is fully observed);
2. the lead is active at `T`: `T < converted_at` for converted leads,
   `T < lead_terminal_time(...)` for disqualified leads, always for open leads. A
   disqualified lead is never a negative after its disqualification;
3. `T + 30d <= lead_observation_end(...)` (the label window is within the observation
   horizon). Past that the snapshot is dropped (censored), not labelled negative.

These rules apply to open, converted and disqualified leads alike, so inclusion never
depends on a conversion or disqualification after `T`. With the 180-day horizon and the
7-day cadence, the oldest possible snapshot is at age 147 days (`k = 21`).

Label `converted_within_30d`: `converted_at` in `(T, T + 30d]`.

### Features

An activity is **known at T** when `created_at <= T AND occurred_at <= T`. Only known
activities count; windows then use `occurred_at` and are left-exclusive,
right-inclusive: 7d = `(T − 7d, T]`, 30d = `(T − 30d, T]`. A day is exactly 24 hours.

| Feature | Definition | Type |
|---|---|---|
| `lead_source` | Lead source (fixed categories) | category |
| `industry` | Lead industry | string, nullable |
| `company_size` | Employees | Int64, nullable |
| `estimated_value` | Exact decimal; cast to float inside the model pipeline | decimal(14,2), nullable |
| `lead_age_days` | `(T − created_at)` in days | float64 |
| `activity_count_7d`, `activity_count_30d` | Known activities in the window | int32 |
| `email_count_30d` / `meeting_count_30d` | Known EMAIL / MEETING activities in 30d (DEMO is not a meeting) | int32 |
| `inbound_response_count_30d` | Known INBOUND activities in 30d | int32 |
| `active_days_30d` | Distinct UTC calendar dates with a known activity in 30d | int32 |
| `days_since_last_activity` | Days since the latest known activity; NULL if none (imputed inside the model pipeline) | float64, nullable |

**Never features:** `status`, `converted_at`, `updated_at` (snapshot/label control only),
`owner_id`, and `activities.created_at` (availability only).

**Deferred Notes feature:** `notes_count_30d` (notes with `created_at` in `(T − 30d, T]`) is
defined in the spec but disabled, because the synthetic dataset currently contains no
Notes. It is absent from the parquet and the model feature list; enabling it changes the
spec fingerprint and requires a spec version bump.

### Output

Columns: `lead_id`, `prediction_timestamp`, the active features (spec order),
`converted_within_30d`; rows sorted by `(prediction_timestamp, lead_id)`. The parquet
schema metadata (`revintel.lead_features`) records the spec version and fingerprint,
active and deferred features, window semantics, availability rule, snapshot policy,
source `dataset_sha256`, `features_sha256` (content hash of the rows) and git commit.
Reproducibility is judged by `features_sha256`, not by file bytes.

Online scoring (`online_features.fetch_online_features(connection, lead_id, at)`) loads
the lead and its activities known at `at` with the same queries and calls the same
`compute_features()`, so a lead scored at `T` gets exactly its training-row features.

---

## Implementation order

All tables are designed now and implemented incrementally. Each phase adds its models,
tests and exactly one Alembic migration, and must include every constraint and index
listed for its tables.

| Schema phase | Tables | Also in the migration | TASKS.md phase |
|---|---|---|---|
| 1 ✅ `d9abf148e9ed` | `users`, `accounts`, `contacts`, `leads`, `opportunities` | Enum checks, non-negative checks, `ck_leads_converted_requires_converted_at` | Phase 1 — Backend |
| 2 ✅ `9d4d176a3488` | `activities`, `notes` | `*_at_least_one_parent` checks, FK indexes incl. `created_by_id` / `author_id` | Phase 1 — Backend |
| 3 ✅ `8d37bd7e6458` | `lead_predictions` | Idempotency unique constraint (also the `lead_id` index), probability range check, nullable `inference_ms` | Phase 2 — Data + ML |
| 4 ✅ `cbc0356c1ce0` | `document_chunks` | `CREATE EXTENSION IF NOT EXISTS vector`, explicit composite unique constraint, `chunk_index` check, `entity_type`/`entity_id`, `source_sha256` | Phase 4 — AI |

---

## Design decisions (Q1–Q14)

All open questions from the initial design review are resolved. The rest of this document
already reflects these decisions.

| # | Topic | Decision |
|---|---|---|
| Q1 | Delete behavior | User refs `SET NULL`; core CRM relationships `RESTRICT`; derived data (`lead_predictions`, `document_chunks`) `CASCADE`. Documented now; no DELETE endpoints in V1. |
| Q2 | User email case | Service layer stores canonical lowercase; plain `UNIQUE (email)`. No `CITEXT`, no `lower(email)` index in V1. |
| Q3 | VARCHAR lengths | Bounded lengths per [Strings](#strings); long free-form content stays `TEXT`. |
| Q4 | ML leakage | No history table in V1. `status`, `converted_at` never features; `owner_id` not a feature in V1; nothing after `T`. `lead_source`, `industry`, `company_size`, `estimated_value` allowed because the V1 generator keeps them immutable. |
| Q5 | Activity/note parents | Service-layer cross-account validation. DB CHECK requires at least one parent (activities: 4 FKs; notes: 3 FKs). Lead-only rows allowed. |
| Q6 | Conversion check | `CHECK (status <> 'CONVERTED' OR converted_at IS NOT NULL)`, one-directional by design. |
| Q7 | User FK indexes | Index `activities.created_by_id` and `notes.author_id`; every FK is indexed. |
| Q8 | Prediction indexing | Unique `uq_lead_predictions_lead_id_timestamp_model` on `(lead_id, prediction_timestamp, model_name, model_version)` for idempotent retries (shortened name: the full descriptive name exceeds 63 characters). Its index, led by `(lead_id, prediction_timestamp)`, serves "latest prediction for this lead" and the FK, so there is no separate `lead_id` or composite index. |
| Q9 | Document chunks | `occurred_at` → `source_timestamp NOT NULL`; add `embedding_model VARCHAR(255) NOT NULL`; keep `VECTOR(384)`; account-scoped RAG only, lead-only content not indexed. Implemented with the business timestamp named `occurred_at` and `embedding` NOT NULL (see [document_chunks](#document_chunks)). |
| Q10 | Lead conversion links | Deferred. V1 uses `status` + `converted_at` only. |
| Q11 | Naming / audit | `activities.created_by` → `created_by_id`; add `users.updated_at`. |
| Q12 | Non-negative checks | `>= 0` on `employee_count`, `annual_revenue`, `company_size`, `estimated_value`, `amount`, and later `chunk_index`. |
| Q13 | Composite names | Global naming convention unchanged; composite constraints/indexes get explicit names. |
| Q14 | Currency | All money is USD in V1; no currency columns. |

---

## Deferred / future considerations

Intentionally postponed. Each needs a design update here and a new migration.

- **Lead conversion linkage.** If a real "convert lead" workflow is built, add explicit links
  from the lead to the account/contact/opportunity created during conversion (e.g.
  `leads.converted_account_id`, `converted_contact_id`, `converted_opportunity_id`).
- **Mutable lead history / snapshots.** Needed if lead attributes change after creation
  and must remain point-in-time correct for ML (e.g. `lead_snapshots` or
  `lead_status_history`).
- **Lead-scoped RAG.** Indexing lead-only activities/notes requires a separate,
  lead-scoped corpus or retrieval path; `document_chunks.account_id` stays `NOT NULL`.
- **Multi-currency.** Add ISO 4217 currency codes (e.g. `currency CHAR(3)`) and conversion
  rules if multi-currency CRM support is needed.
- **Embedding model / dimension changes.** A different dimension requires a migration of
  `VECTOR(384)` and full re-embedding; a same-dimension model change requires re-embedding
  chunks whose `embedding_model` differs.
- **Vector similarity index.** Designed with the RAG feature, including filtered search by
  `account_id`.
- **Soft deletes and multi-tenancy.** Out of scope for V1.

---

## Layout

| Path | Purpose |
|---|---|
| `backend/app/db/base.py` | Declarative `Base` and its `MetaData` naming convention. |
| `backend/app/db/session.py` | Engine and session factory used by the API. |
| `backend/app/models/` | ORM models. Every model module must be imported in `app/models/__init__.py`. |
| `alembic.ini` | Alembic settings. Contains no database URL. |
| `backend/alembic/env.py` | Migration environment. |
| `backend/alembic/versions/` | Migration revisions. |

## How Alembic is wired

- **Database URL:** `env.py` calls `app.core.config.get_database_settings()`, which loads
  only `DatabaseSettings` (`DATABASE_URL` plus the `DB_*` pool/timeout settings). Migrations
  therefore do not need `REDIS_URL` or other application settings. The full application
  `Settings` extends `DatabaseSettings`, so `DATABASE_URL` validation is shared. Values
  come from environment variables first, then `.env` in the current directory.
  `alembic.ini` has no `sqlalchemy.url`, so no credentials live in it.
- **Metadata:** `env.py` imports `app.models` (which registers models) and sets
  `target_metadata = Base.metadata`, so autogenerate compares the models against PostgreSQL.
- **Driver:** online migrations use a `postgresql+psycopg` engine with `NullPool` and the
  configured connect timeout. The engine is created only inside `run_migrations_online()`,
  so importing `env.py` or the app never opens a connection.
- **Constraint names:** `Base.metadata` uses a naming convention (`pk_`, `fk_`, `uq_`,
  `ix_`, `ck_`) so generated migrations have deterministic, reversible constraint names.
- **Generated files:** a Ruff post-write hook (`ruff check --fix`) runs on every new
  revision so it passes `ruff check .`.

## Commands

Run from the repository root with the virtualenv active and PostgreSQL running
(`docker compose up -d`):

```bash
alembic current                                     # revision applied to the database
alembic heads                                       # latest revision(s) in the codebase
alembic history                                     # all revisions
alembic revision --autogenerate -m "add crm models" # generate a revision from model changes
alembic upgrade head                                # apply migrations
alembic downgrade -1                                # roll back one revision
alembic check                                       # fail if models and database differ
```

To target another database (for example the test database), override the URL for one
command, since environment variables take precedence over `.env`:

```bash
DATABASE_URL=postgresql+psycopg://revintel:revintel@localhost:5432/revintel_test alembic upgrade head
```

## Rules

- Never edit a migration after it has been committed; add a new one.
- Every schema change needs a migration. Review autogenerated revisions against this
  document before committing: autogenerate does not detect everything (e.g. CHECK
  constraint changes, some server defaults, renames, and extension-specific types).
- **Enum columns in migrations:** autogenerate renders each non-native enum's CHECK
  several times (inside `sa.Enum(..., create_constraint=True)`, once with the bare enum
  name, and once via the naming convention). Write enum columns as
  `sa.String(length=32)` plus exactly one `CheckConstraint(..., name=op.f("ck_<table>_<enum>"))`,
  as in `d9abf148e9ed`. `alembic check` treats this as matching the models.
- Add indexes deliberately; only those listed in this document.

## Verification

Automated, without Docker (`pytest`):

- `backend/tests/unit/test_models_metadata.py`: tables, column types/lengths/nullability,
  constraint and index names, and FK delete actions match this document.
- `backend/tests/unit/test_alembic.py`: no hardcoded URL, single head, offline upgrade SQL
  creates only the implemented tables with each constraint named exactly once, and the
  downgrade drops them in reverse dependency order.

Against PostgreSQL (`pytest -m integration`, requires `docker compose up -d`):

- `test_migrations_integration.py`: downgrade to base and upgrade to head; each migration
  applied and reverted on its own (`d9abf148e9ed` → head → `-1`); documented indexes; enum
  CHECKs present exactly once.
- `test_alembic_integration.py`: `alembic check` reports no drift; autogenerate sees
  `Base.metadata`.
- `test_constraints_integration.py`: unique, CHECK, FK, `RESTRICT` and `SET NULL` behavior,
  UTC timestamps, `updated_at` changing only on real updates.
- `test_activity_note_constraints_integration.py`: at-least-one-parent and enum CHECKs, activity/note
  foreign keys, `RESTRICT` on CRM parent deletion (SQL and ORM), `SET NULL` of
  `created_by_id` / `author_id` on user deletion.

Manual:

```bash
alembic upgrade head
alembic current -v   # cbc0356c1ce0 (head); URL printed with the password masked
alembic check        # "No new upgrade operations detected."
```

## Test database

Integration tests never build the schema with `Base.metadata.create_all()`. A
session-scoped fixture (`backend/tests/integration/conftest.py`) runs, in order:

1. **Safety guard:** fail unless the database name ends in `_test`
   (`TEST_DATABASE_URL`, default `revintel_test`). Nothing destructive runs before this.
2. `alembic downgrade base`, `alembic upgrade head`, `alembic check`.

After each test that touches CRM data, every model table is `TRUNCATE`d (again behind the
guard). `alembic_version` is left to Alembic.

The application engine pins sessions to UTC (`options=-c timezone=UTC`), so timestamps
are returned as UTC regardless of server configuration.
