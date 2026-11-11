# Airflow: weekly lead conversion training (local development)

Airflow 3.4 runs in Docker (`airflow/docker-compose.yaml`) and schedules
`lead_conversion_training` every week:

```
extract_crm_snapshot → build_features → train_candidate_models → evaluate_models
→ register_best_candidate → assign_challenger_alias
```

| | |
|---|---|
| UI | http://localhost:8080 (user `admin`, generated password) |
| Schedule | `0 3 * * 1`: Mondays 03:00 UTC, `catchup=False`, one active run at a time |
| Metadata database | Its own `airflow-postgres` container, separate from `revintel`, `revintel_test`, `revintel_ml` and `mlflow` |
| Pipeline code | `ml/pipeline/lead_conversion.py`, called by `airflow/dags/lead_conversion_training.py` |
| Run artifacts | `artifacts/pipeline_runs/<run key>/` on the host (gitignored) |

Airflow isn't installed in the RevIntel virtualenv. The image (`airflow/Dockerfile`) is
the official `apache/airflow:3.4.0-python3.14` plus a separate virtualenv,
`/opt/revintel-venv`, built from `requirements/lock.txt`. Pipeline tasks use
`@task.external_python` to run in that virtualenv, so Airflow's dependency constraints
and RevIntel's pinned versions never mix. RevIntel's code (`ml/`, `backend/`) is mounted
read-only, so tasks always run the current checkout.

## Start and stop

1. **MLflow on the host, reachable from containers.** The normal `scripts/start_mlflow.sh`
   defaults (127.0.0.1 only) are unchanged; for Airflow, listen on all interfaces and
   allow the Docker host name:

   ```bash
   export MLFLOW_BACKEND_STORE_URI=postgresql+psycopg://revintel:revintel@localhost:5432/mlflow
   MLFLOW_HOST=0.0.0.0 \
   MLFLOW_SERVER_ALLOWED_HOSTS="localhost:*,127.0.0.1:*,host.docker.internal:*" \
   scripts/start_mlflow.sh
   ```

   `MLFLOW_SERVER_ALLOWED_HOSTS` replaces MLflow's default host allow-list, so it must
   also list `localhost` and `127.0.0.1` for the browser on the host.

2. **RevIntel PostgreSQL** (`docker compose up -d postgres` in the repo root), with
   `revintel_ml` holding the synthetic dataset from `data/raw/synthetic_crm_manifest.json`.

3. **Airflow:**

   ```bash
   cp airflow/.env.example airflow/.env        # first time; set AIRFLOW_UID=$(id -u)
   docker compose -f airflow/docker-compose.yaml up -d --build
   ```

4. **Stop:** `docker compose -f airflow/docker-compose.yaml down`. Add `-v` to also delete
   Airflow's metadata volume (DAG run history). That doesn't affect RevIntel's databases
   or MLflow.

## Open the UI

Go to http://localhost:8080 and log in as `admin`. SimpleAuthManager generates the password
on first start and stores it in `airflow/config/simple_auth_manager_passwords.json.generated`
(gitignored), and the API server logs it at startup:

```bash
cat airflow/config/simple_auth_manager_passwords.json.generated
```

## Trigger the DAG manually

The DAG starts paused. In the UI, unpause `lead_conversion_training` and select **Trigger**.
From the command line:

```bash
docker compose -f airflow/docker-compose.yaml exec airflow-scheduler \
    airflow dags unpause lead_conversion_training
docker compose -f airflow/docker-compose.yaml exec airflow-scheduler \
    airflow dags trigger lead_conversion_training
```

Once unpaused, it also runs weekly on Mondays at 03:00 UTC. With `catchup=False`, unpausing
still schedules the most recent weekly interval immediately (one run, not a backfill).
Each DAG run registers its own version.

## Environment variables (`airflow/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `AIRFLOW_UID` | `50000` | Your host user ID (`id -u`), so files in mounted directories belong to you. |
| `SYNTHETIC_DATABASE_URL` | **required** | Source CRM dataset to extract from (read-only), e.g. `revintel_ml` through `host.docker.internal`. |
| `MLFLOW_TRACKING_URI` | `http://host.docker.internal:5001` | MLflow server on the host. |
| `MLFLOW_MODEL_NAME` | `revintel-lead-conversion` | Registered model that receives the new version and `challenger`. |
| `FERNET_KEY`, `AIRFLOW__API_AUTH__JWT_SECRET` | local defaults | Set real values beyond local development. |

The compose file sets `REVINTEL_PYTHON`, `REVINTEL_PIPELINE_ROOT` and
`REVINTEL_SOURCE_MANIFEST` for the containers.

## Shared artifact paths

Each DAG run has its own directory, named after its run key
(`lead_conversion_training/<run_id>`):

```
artifacts/pipeline_runs/lead_conversion_training_<run_id>/
  snapshot/     leads.parquet, lead_activities.parquet, snapshot.json (row counts,
                content hash, source database name, manifest fields)
  features/     lead_features.parquet, lead_features_summary.json
  models/       logistic_regression.joblib, xgboost.joblib, fit.json
  evaluation/   evaluation_report.json, <model>/model.joblib, <model>/metadata.json
```

Each step writes into a temporary directory and publishes it atomically, along with a
`_SUCCESS.json` holding the step's result. On the host this is `artifacts/pipeline_runs/`;
in the containers it's `/opt/revintel/pipeline_runs` (`REVINTEL_PIPELINE_ROOT`). Results
refer to files by paths relative to the run directory, so object storage such as S3 can
replace the local directory later without changing the DAG or XCom contents.

## Why XCom holds metadata only

XCom values are stored in Airflow's metadata database and moved between processes. They
must stay small, safe to deserialize and independent of where the data lives. So tasks
only pass:
- relative paths
- row counts and hashes (`features_sha256`, snapshot content hash, `dataset_sha256`)
- MLflow run IDs, model URIs and registered versions
- validation metrics

DataFrames and models stay in the run directory and in MLflow.

## What each task does

| Task | Reuses | Result (XCom) |
|---|---|---|
| `extract_crm_snapshot` | `extract_sources()`: the feature pipeline's read-only extraction, checked against the manifest | snapshot path, row counts, content hash |
| `build_features` | `build_from_sources()`: the current snapshot policy and feature contract | features path, `features_sha256`, rows |
| `train_candidate_models` | `fit_candidates()`: Logistic Regression, plus XGBoost depth and early stopping on validation | models path, XGBoost depth and iteration |
| `evaluate_models` | `evaluate_candidates()` (validation thresholds, then one test evaluation) and `log_candidate_runs()` | validation metrics, baseline check, MLflow run and model IDs |
| `register_best_candidate` | `select_best_candidate()` and `register_candidate()` | selected algorithm, version |
| `assign_challenger_alias` | `designate_challenger()` (`ml/training/register.py`) | previous and new challenger |

Each model is fitted once (`train_candidate_models`) and evaluated once (`evaluate_models`).

**Candidate selection uses validation metrics only.** Candidates must pass the validation
baseline check (better than the naive baseline on validation PR-AUC and Brier). The highest
validation PR-AUC wins; ties go to lower validation Brier, then higher validation ROC-AUC,
then algorithm name. Test metrics are reported but never used. If no candidate qualifies,
nothing is registered and `challenger` is left unchanged.

## Retries

Within one DAG run, every task is safe to retry:
- **File steps:** a step that already published its directory returns its recorded result
  instead of recomputing.
- **MLflow runs:** they're tagged with the run key, and a retry reuses the finished
  session.
- **Registration:** it first looks for a version tagged with this run key and algorithm.
  It creates the version and its tags in a single call, so a retry after a crash finds it.
- **Challenger:** setting the alias is idempotent.

Separate DAG runs are independent: each registers its own version.

## Why challenger is automatic but champion is manual

`challenger` means "the latest candidate worth reviewing". Moving it is harmless, because
nothing serves it. `champion` is what the scoring API serves, so changing it is a product
decision. It needs a person to review the challenger against the current champion, then
run `python -m ml.training.promote --version <N>` and restart the API (see
`docs/MLFLOW.md`). The DAG has no task that touches `champion`, and `challenger` is never
put on the current champion version.

## Tests

- **Pipeline logic** (RevIntel virtualenv, no Airflow): `pytest ml/tests/test_lead_conversion_pipeline.py`
  and `pytest -m integration backend/tests/integration/test_pipeline_extract_integration.py`.
- **DAG structure** (inside the Airflow image):

  ```bash
  docker compose -f airflow/docker-compose.yaml run --rm airflow-cli \
      bash -c "python -m pytest -q /opt/airflow/tests"
  ```
