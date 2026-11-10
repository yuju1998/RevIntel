# Lead conversion model results

This is a snapshot of the current experiment. It is not a run history: detailed run
tracking will move to MLflow. For how the features, the split and the training work, see
`docs/DATABASE.md` ("Lead conversion features") and `docs/ARCHITECTURE.md`.

**Task:** at prediction time T, predict whether a lead converts in `(T, T + 30d]`
(`converted_within_30d`).

## Data

| | |
|---|---|
| Source | Synthetic CRM dataset, `dataset_sha256` `e32406eb…5604c06` |
| Feature table | `data/processed/lead_features.parquet`, `features_sha256` `3c4cc613d3cce71ce7ca3bc496e0fcc695213135c68cac7bd9521b811a687b8c` |
| Feature spec | version 1.0.0, fingerprint `6c924eb0…`; 12 active features (`notes_count_30d` deferred) |
| Snapshots | 342,044 lead snapshots, 4.33% positive |

### Split

The data is split by `prediction_timestamp`: the 70% and 85% points of the timeline
mark where validation and test begin. A snapshot is kept only if its label window
`(T, T + 30d]` ends strictly before the next split begins. Purging removed 24,506 rows
at the train/validation boundary and 25,643 at the validation/test boundary (14.7% in
total).

| Split | Rows | Positives | Positive rate | Prediction timestamps |
|---|---|---|---|---|
| Train | 214,924 | 9,506 | 4.42% | 2025-04-16 → 2026-04-04 |
| Validation | 25,664 | 1,068 | 4.16% | 2026-05-04 → 2026-06-04 |
| Test | 51,307 | 2,171 | 4.23% | 2026-07-04 → 2026-08-31 |

All three models used the same rows in each split.

## Results

| Model | Split | ROC-AUC | PR-AUC | Brier | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Baseline | Validation | 0.500 | 0.042 | 0.0399 | 0.0442 | 0.042 | 1.000 | 0.080 |
| Baseline | Test | 0.500 | 0.042 | 0.0405 | 0.0442 | 0.042 | 1.000 | 0.081 |
| Logistic Regression (v3) | Validation | 0.718 | 0.135 | 0.0382 | 0.0984 | 0.196 | 0.257 | 0.222 |
| Logistic Regression (v3) | Test | 0.701 | 0.114 | 0.0393 | 0.0984 | 0.157 | 0.211 | 0.180 |
| XGBoost (v4) | Validation | 0.733 | 0.145 | 0.0379 | 0.1115 | 0.207 | 0.262 | 0.232 |
| XGBoost (v4) | Test | 0.711 | 0.120 | 0.0391 | 0.1115 | 0.165 | 0.204 | 0.182 |

v3 and v4 are versions of the registered model `revintel-lead-conversion`.

- **Baseline:** a constant score equal to the training positive rate (0.0442). At its
  threshold, every lead is predicted positive.
- **Thresholds:** each model's threshold maximises F1 on validation. If several
  thresholds tie, the higher one is used.
- **XGBoost:**
  - `max_depth` 3 was chosen over 5 on validation PR-AUC (0.145 vs 0.139).
  - Early stopping on validation log loss settled at best iteration 169, out of a
    maximum of 1,000.

**Held-out test:** XGBoost depth, early stopping and every threshold were chosen on
validation only. The thresholds were frozen before the test split was scored, and test
was scored once, at the end.

## Interpretation

- **XGBoost (v4) is the current `champion`.** It is ahead of Logistic Regression on
  every test metric (ROC-AUC, PR-AUC, Brier, precision and F1) except recall, where
  Logistic Regression is ahead (0.211 vs 0.204). Its test PR-AUC is about 2.8× the
  positive rate.
- **v2 is the former champion.** It is an XGBoost model trained on the earlier feature
  table (`7d9a14ed…f026c6`), built under the old snapshot policy.
- **Logistic Regression remains the interpretable baseline.** It performs reasonably
  close to XGBoost and is simpler to explain.
- **Both models carry real signal over the naive baseline.** The gain in Brier score is
  small, as expected for a rare (about 4%) label. Precision at the chosen thresholds is
  still low (16–17%).
- **Validation performance is slightly higher than test performance.** Test scores are a 
  little lower than validation for both models, consistent with selection on validation.

## Product tiers

Champion v4 scores are grouped into tiers:

| Tier | Probability |
|---|---|
| HOT | `>= 0.15` |
| WARM | `>= 0.05 and < 0.15` |
| COLD | `< 0.05` |

These are product-policy thresholds, separate from the model evaluation threshold
(about 0.1115) used for precision, recall and F1. They were chosen on validation and then
checked once on test.

| Tier | Validation share | Validation conversion | Test share | Test conversion |
|---|---|---|---|---|
| HOT | 2.44% | 25.4% | 2.54% | 21.1% |
| WARM | 21.45% | 8.0% | 21.99% | 7.6% |
| COLD | 76.11% | 2.4% | 75.48% | 2.7% |

## Limitations

- **Synthetic data:** the generator's conversion behaviour is simulated, not fitted to
  real CRM data, so the absolute numbers say nothing about production performance.
- **Single split:** one chronological split with no backtesting over several periods,
  so there are no confidence intervals or variance estimates.
- **Small, reused validation set:** about 1,100 positives, used for early stopping,
  depth and threshold.
- **Correlated rows:** each lead contributes up to 21 snapshots. A lead can also appear
  in more than one split, at different T.
- **Threshold rule:** maximising F1 is a default rather than a business-cost choice. No
  probability calibration step was applied beyond what the Brier score shows.
- **Features:**
  - lead attributes never change over time in the synthetic data
  - no Notes feature yet
  - snapshot eligibility relies on the synthetic terminal-time and observation-end
    conventions
  - a snapshot needs its full 30-day label window within the observation horizon, with no
    exception for a later conversion or disqualification
- **Not yet in place:** models are not served online. Tracked training registers the
  candidates in MLflow (`docs/MLFLOW.md`).
