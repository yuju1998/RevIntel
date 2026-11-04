# MLflow tracking and Model Registry (local)

| | |
|---|---|
| Tracking server | http://127.0.0.1:5001 |
| Backend store | PostgreSQL database `mlflow` (created by `docker/postgres/init`) |
| Artifacts | `artifacts/mlflow/` (gitignored), served through the tracking server |
| Experiment | `lead-conversion` |
| Registered model | `revintel-lead-conversion` |

MLflow only ever writes to its own `mlflow` database. The RevIntel application databases
are never touched.

## 1. Start MLflow

```bash
docker compose up -d postgres
source ~/.virtualenvs/revintel/bin/activate
export MLFLOW_BACKEND_STORE_URI=postgresql+psycopg://revintel:revintel@localhost:5432/mlflow
scripts/start_mlflow.sh
```

The script does the following:
1. Reads `MLFLOW_BACKEND_STORE_URI` from the environment, or from that one key in the
   repo-root `.env` (the file is never sourced).
2. Refuses any URI that isn't PostgreSQL or whose database isn't `mlflow`.
3. Creates `artifacts/mlflow/`.
4. Runs `mlflow db upgrade`.
5. Starts the server on 127.0.0.1:5001, which you can override with
   `MLFLOW_HOST` / `MLFLOW_PORT`.

The URI must use the `postgresql+psycopg` driver because psycopg2 isn't installed. Stop
the server with Ctrl-C.

## 2. Run tracked training

```bash
python scripts/train_lead_models.py --track
```

**Options:**
- `--tracking-uri`: default `$MLFLOW_TRACKING_URI`, then `http://127.0.0.1:5001`
- `--experiment`: default `lead-conversion`
- `--registered-model`: default `$MLFLOW_MODEL_NAME`, then `revintel-lead-conversion`

**What `--track` changes:** training and evaluation are the same as without it, and the
local joblib artifacts in `artifacts/models/lead_conversion/` are still written. On top
of that it:
- logs one `training-session` run, with dataset, feature and split parameters, baseline
  metrics and `evaluation_report.json`;
- logs one nested run per candidate, with:
  - the algorithm, `hp.*` hyperparameters, data and feature hashes, feature spec version
    and fingerprint, and active features
  - split dates, boundaries and purged counts
  - the threshold and its selection rule
  - validation and test ROC-AUC, PR-AUC, precision, recall, F1 and Brier score
  - `metadata.json` and the full Pipeline;
- registers each candidate as a new version of `revintel-lead-conversion`.

**Without `--track`:** MLflow isn't used at all.

**Failure behaviour with `--track`:**
- If the server isn't reachable, the command exits with code 2 **before** training.
- If logging fails afterwards, it exits with code 1. The local artifacts have already been
  written.

### The registered model

- **Loading:** a version is the full sklearn Pipeline, preprocessing included, stored with
  skops. Loading it needs this repo on `PYTHONPATH`.
- **Prediction:** the generic MLflow interface (`mlflow.pyfunc`) returns `predict_proba`:
  an array of shape (rows, 2) holding [P(not converted), P(converted)].
- **Signature:**
  - **Inputs:** the 12 active features in spec order, as strings, longs and doubles;
    missing values are allowed.
  - **Output:** that probability array.
- **Threshold:** the decision threshold isn't applied by the model. It's in the version's
  `threshold` tag and the run's `threshold.value` parameter.
- **Version tags:**
  - `algorithm`, `role=candidate`, `threshold`
  - `validation_baseline_check`: `passed` if the candidate beats the naive baseline on both
    validation PR-AUC and validation Brier score, otherwise `failed`. It's a sanity check
    only, not a review decision.
  - `dataset_sha256`, `features_sha256`, `feature_spec_fingerprint`
  - `validation_pr_auc`, `test_pr_auc`
  - `training_session_run_id`, `git_commit`

`validation_status` is reserved for a future explicit review or promotion decision.

## 3. Open the UI

Go to http://127.0.0.1:5001.
- **Experiments → lead-conversion:** expand a `training-session` run to compare its
  candidate runs.
- **Models → revintel-lead-conversion:** shows the registered versions and their tags.

## 4. Inspect registered versions

```bash
python - <<'EOF'
from mlflow import MlflowClient
client = MlflowClient("http://127.0.0.1:5001")
for v in client.search_model_versions("name='revintel-lead-conversion'"):
    print(v.version, v.tags.get("algorithm"), v.tags.get("validation_baseline_check"),
          v.tags.get("validation_pr_auc"), v.aliases)
EOF
```

## 5. Assign `champion` / `challenger` manually

Training never assigns aliases. After reviewing the versions:

```bash
python - <<'EOF'
from mlflow import MlflowClient
client = MlflowClient("http://127.0.0.1:5001")
name = "revintel-lead-conversion"
client.set_registered_model_alias(name, "champion", "2")      # the reviewed version
client.set_registered_model_alias(name, "challenger", "1")    # optional
client.set_model_version_tag(name, "2", "validation_status", "approved")
print(client.get_model_version_by_alias(name, "champion").version)
EOF
```

`models:/revintel-lead-conversion@champion` then resolves to that version. An alias moves
when you set it on another version, and `client.delete_registered_model_alias(name,
"challenger")` removes one.
