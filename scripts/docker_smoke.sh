#!/usr/bin/env bash
# Build and start the Compose stack, then check the API container end to end.
#
#   scripts/docker_smoke.sh            # API on localhost:${API_PORT:-8000}
#   API_PORT=8001 scripts/docker_smoke.sh
#
# Non-destructive: no migrations are applied and no data is written. Champion scoring,
# semantic search and the Copilot need MLflow, indexed data and keys; see docs/DOCKER.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PORT="${API_PORT:-8000}"
export API_PORT="$PORT"

echo "==> docker compose up -d --build"
docker compose up -d --build

echo "==> waiting for the api container to be healthy"
for _ in $(seq 1 60); do
    status="$(docker inspect -f '{{.State.Health.Status}}' revintel-api 2>/dev/null || true)"
    [ "$status" = "healthy" ] && break
    sleep 2
done
[ "$status" = "healthy" ] || { echo "api is not healthy ($status)"; docker logs --tail 50 revintel-api; exit 1; }

echo "==> GET /health and /ready"
curl -fsS "http://localhost:${PORT}/health" && echo
curl -fsS "http://localhost:${PORT}/ready" && echo

echo "==> alembic current (from the container; read-only)"
docker compose run --rm -T api alembic current

echo "==> image checks (no network: the embedding model must be baked in)"
docker run -i --rm --network none -v "$ROOT/requirements/lock.txt:/tmp/lock.txt:ro" \
    --entrypoint python revintel-api:local - <<'EOF'
import importlib.metadata as md
import os
import pathlib
import re

norm = lambda name: re.sub(r"[-_.]+", "-", name).lower()
lock = {norm(l.split("==")[0]): l.split("==")[1].strip() for l in open("/tmp/lock.txt") if "==" in l}
alias = {"xgboost-cpu": "xgboost"}  # CPU-only build of the locked xgboost
installed = {norm(d.metadata["Name"]): d.version for d in md.distributions()}
unpinned = sorted(n for n in installed if alias.get(n, n) not in lock and n != "pip")
drift = sorted(
    n for n, v in installed.items()
    if alias.get(n, n) in lock and v.split("+")[0] != lock[alias.get(n, n)]
)
assert not unpinned and not drift, f"not in lock: {unpinned}; version drift: {drift}"
assert os.getuid() != 0, "the image must not run as root"
for path in ("/app/.env", "/app/artifacts", "/app/data", "/app/backend/tests", "/app/ml/tests"):
    assert not pathlib.Path(path).exists(), f"{path} must not be in the image"

import torch

assert torch.version.cuda is None, "torch must be the CPU build"
from app.ai.embeddings import SentenceTransformerProvider

vector = SentenceTransformerProvider(os.environ["EMBEDDING_MODEL"], 384).embed_query("pricing")
assert len(vector) == 384
print(f"ok: {len(installed)} packages match the lock, non-root, CPU torch, "
      f"offline embedding with {os.environ['EMBEDDING_MODEL']}")
EOF

echo "==> PASS (stack left running; stop the API with: docker compose rm -sf api)"
