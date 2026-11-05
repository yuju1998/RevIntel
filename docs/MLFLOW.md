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
  skops. Loading it needs the repo root and `backend` on the path: run from the repo root
  with `PYTHONPATH=.:backend`.
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

`validation_status` is set only by the manual promotion command (see section 5). Tracked
training never sets it.

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

## 5. Choose a `challenger` and promote it to `champion`

Training only registers candidates and never assigns aliases. Promotion is a manual,
three-step decision. Run these commands from the repo root with the tracking server
running.

**1. Mark a registered version as the challenger:**

```bash
python -m ml.training.register --version <N>
```

This doesn't create a model version. It moves the `challenger` alias to version N and
sets no tags. It refuses a version that doesn't exist, isn't `READY`, or is the current
`champion`.

**2. Evaluate it manually.** Compare it with the current champion using the tags and
metrics from section 4 or the UI. For checks of your own, load it from the repo root with:

```bash
PYTHONPATH=.:backend python - <<'EOF'
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5001")
model = mlflow.pyfunc.load_model("models:/revintel-lead-conversion@challenger")
print(model.metadata.signature)
EOF
```

**3. Promote it:**

```bash
python -m ml.training.promote --version <N>
```

This refuses unless version N is exactly the current `challenger`. If it is, the command:
- sets `validation_status=approved` on version N
- moves the `champion` alias to version N
- removes the `challenger` alias

Both commands print the previous and new alias targets, for example:

```
revintel-lead-conversion:
  champion: version 1 -> version 2
  challenger: version 2 -> none
version 2 (xgboost): validation_baseline_check=passed
version 2: validation_status=approved
Version 1 is kept in the registry.
```

`models:/revintel-lead-conversion@champion` then resolves to the new version.

**How the tags and aliases behave:**
- **Which version is current:** only the `champion` alias says this.
  `validation_status=approved` stays on a version after it's replaced, so it means "was
  approved at some point".
- **Old versions:** they are never deleted. A replaced champion stays in the registry
  with its tags.
- **`validation_baseline_check`:** never changed by these commands. If it's `failed`, they
  print a warning but don't block, because the decision is yours.
- **A failed promotion:** if `promote` stops partway, rerun the same command to finish it.

**Options:** both commands accept `--registered-model` (default `$MLFLOW_MODEL_NAME`, then
`revintel-lead-conversion`) and `--tracking-uri` (default `$MLFLOW_TRACKING_URI`, then
`http://127.0.0.1:5001`).

**Exit codes:** 2 for a refusal or an unreachable server, 1 for an MLflow failure.
