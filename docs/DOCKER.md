# Running the API in Docker

The root `Dockerfile` builds the API image; `docker-compose.yml` runs it with PostgreSQL
and Redis.

```bash
cp .env.example .env              # first time: set JWT_SECRET (and OPENAI_* for openai)
docker compose up --build
curl http://localhost:8000/health
```

If port 8000 is taken (for example by `fastapi dev`), publish elsewhere:
`API_PORT=8001 docker compose up --build`. Stop the API with `docker compose rm -sf api`.

## Configuration

- **`.env` is optional** (`env_file` with `required: false`): when present, the API reads
  its settings and secrets from it at runtime (`JWT_SECRET`, `LLM_PROVIDER`, `OPENAI_*`,
  …). Without it, set those variables in the shell. `JWT_SECRET` is required either way,
  so the API does not start without it.
- **Set by Compose** (they override `.env`, which points at `localhost` for host runs):

  | Variable | Value |
  |---|---|
  | `DATABASE_URL` | `postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}` |
  | `REDIS_URL` | `redis://redis:6379/0` |
  | `MLFLOW_TRACKING_URI` | `${API_MLFLOW_TRACKING_URI:-http://host.docker.internal:5001}` |
  | `EMBEDDING_MODEL` | `${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}` (see below) |

- `POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB` default to `revintel` and are
  used by both the `postgres` service and the API. They only take effect when the
  database volume is first created.
- Nothing from `.env`, no keys, no `artifacts/` or `data/` and no tests go into the image
  (`.dockerignore`).

## MLflow (lead scoring)

MLflow runs on the host, not in Compose. Start it so containers can reach it:

```bash
MLFLOW_HOST=0.0.0.0 \
MLFLOW_SERVER_ALLOWED_HOSTS="localhost:*,127.0.0.1:*,host.docker.internal:*" \
scripts/start_mlflow.sh
```

The API reaches it at `http://host.docker.internal:5001` (Compose maps that name on Linux
too) and loads the `champion` version lazily on the first score; model files are served
by MLflow over HTTP. If MLflow is not running, scoring returns `503` and everything else
works, as outside Docker. Restart the API container after promoting a new champion.

## Embedding model

The embedding model is downloaded **when the image is built** and the container runs with
`HF_HUB_OFFLINE=1`, so nothing is downloaded at startup or on the first search.

- The same `EMBEDDING_MODEL` variable sets the baked model (build arg) and the runtime
  setting, so in Compose they always match.
- Changing the model requires a rebuild: `docker compose up --build`.
- A runtime model that is not baked into the image fails clearly on the first search:
  `503 Semantic search is currently unavailable.`, with the reason logged
  (`cannot load <model>: … couldn't find them in the cached files`).

## Database migrations

Migrations never run on API startup. Apply them explicitly (non-destructive; never
downgrade from here):

```bash
docker compose run --rm api alembic upgrade head
docker compose run --rm api alembic current
```

## Image

- `python:3.14.8-slim-trixie`, non-root user, `fastapi run` on `0.0.0.0:8000` (one
  worker, `--proxy-headers`).
- Dependencies: `requirements/api.txt` installed with `requirements/lock.txt` as
  constraints (exact locked versions, no dev tools, no MLflow server). torch and xgboost
  are their CPU-only builds at the locked versions (`torch+cpu`, `xgboost-cpu`): no CUDA
  or NCCL.
- Healthcheck: `GET /health`. `/ready` checks PostgreSQL and Redis.
- The champion model, embedding model and OpenAI client still load lazily on first use.

## Smoke test

```bash
API_PORT=8001 scripts/docker_smoke.sh
```

It builds and starts the stack and checks `/health`, `/ready`, `alembic current`, that
every installed package matches the lock, that the image runs as non-root with CPU torch
and nothing local inside, and that the embedding model loads with networking disabled.
Scoring with the champion, semantic search and the Copilot need MLflow, indexed data and
(for `LLM_PROVIDER=openai`) an API key; check them against the running API as described
in `docs/API.md`.
