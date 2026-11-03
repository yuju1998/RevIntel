#!/usr/bin/env bash
# Start the local MLflow tracking server and Model Registry.
#
#   export MLFLOW_BACKEND_STORE_URI=postgresql+psycopg://revintel:revintel@localhost:5432/mlflow
#   scripts/start_mlflow.sh
#
# - Backend store: PostgreSQL database `mlflow` (MLFLOW_BACKEND_STORE_URI, or that key in
#   the repo-root .env). Only MLflow's own database is ever touched.
# - Artifacts: <repo>/artifacts/mlflow, served through the tracking server.
# - Listens on MLFLOW_HOST:MLFLOW_PORT (default 127.0.0.1:5001).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="${ROOT}/artifacts/mlflow"
HOST="${MLFLOW_HOST:-127.0.0.1}"
PORT="${MLFLOW_PORT:-5001}"
EXAMPLE_URI="postgresql+psycopg://revintel:revintel@localhost:5432/mlflow"

fail() {
    echo "Error: $*" >&2
    exit 1
}

# Read only this one key from .env (the file is never sourced).
if [[ -z "${MLFLOW_BACKEND_STORE_URI:-}" && -f "${ROOT}/.env" ]]; then
    MLFLOW_BACKEND_STORE_URI="$(
        sed -n 's/^MLFLOW_BACKEND_STORE_URI=//p' "${ROOT}/.env" | tail -n 1 | tr -d '"'"'"
    )"
fi
[[ -n "${MLFLOW_BACKEND_STORE_URI:-}" ]] ||
    fail "MLFLOW_BACKEND_STORE_URI is not set (e.g. ${EXAMPLE_URI})."

# Guard: the backend must be the dedicated `mlflow` database, never an application one.
database="${MLFLOW_BACKEND_STORE_URI##*/}"
database="${database%%\?*}"
[[ "${MLFLOW_BACKEND_STORE_URI}" == postgresql* ]] ||
    fail "MLFLOW_BACKEND_STORE_URI must be a PostgreSQL URI (e.g. ${EXAMPLE_URI})."
[[ "${database}" == "mlflow" ]] ||
    fail "refusing backend database '${database}': MLflow must use the 'mlflow' database."

command -v mlflow >/dev/null 2>&1 ||
    fail "mlflow is not on PATH; activate the project virtualenv first."

export MLFLOW_DISABLE_AGENT_HINT=1
mkdir -p "${ARTIFACTS_DIR}"

echo "Upgrading the MLflow schema in database '${database}'..."
mlflow db upgrade "${MLFLOW_BACKEND_STORE_URI}"

echo "Starting MLflow on http://${HOST}:${PORT} (artifacts: ${ARTIFACTS_DIR})"
exec mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --artifacts-destination "${ARTIFACTS_DIR}" \
    --host "${HOST}" \
    --port "${PORT}"
