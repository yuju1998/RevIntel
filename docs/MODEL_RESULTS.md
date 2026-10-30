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
| Feature table | `data/processed/lead_features.parquet`, `features_sha256` `7d9a14ed…f026c6` |
| Feature spec | version 1.0.0, fingerprint `6c924eb0…`; 12 active features (`notes_count_30d` deferred) |
| Snapshots | 345,494 lead snapshots, 4.42% positive |

### Split

The data is split by `prediction_timestamp`: the 70% and 85% points of the timeline
mark where validation and test begin. A snapshot is kept only if its label window
`(T, T + 30d]` ends strictly before the next split begins. Purging removed 24,821 rows
at the train/validation boundary and 25,983 at the validation/test boundary (14.7% in
total).

| Split | Rows | Positives | Positive rate | Prediction timestamps |
|---|---|---|---|---|
| Train | 217,025 | 9,738 | 4.49% | 2025-04-16 → 2026-04-05 |
| Validation | 25,841 | 1,105 | 4.28% | 2026-05-05 → 2026-06-04 |
| Test | 51,824 | 2,254 | 4.35% | 2026-07-04 → 2026-08-31 |

All three models used the same rows in each split.

## Results

| Model | Split | ROC-AUC | PR-AUC | Brier | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Baseline | Validation | 0.500 | 0.043 | 0.0409 | 0.0449 | 0.043 | 1.000 | 0.082 |
| Baseline | Test | 0.500 | 0.043 | 0.0416 | 0.0449 | 0.043 | 1.000 | 0.083 |
| Logistic Regression | Validation | 0.708 | 0.132 | 0.0394 | 0.0896 | 0.179 | 0.280 | 0.219 |
| Logistic Regression | Test | 0.690 | 0.111 | 0.0405 | 0.0896 | 0.144 | 0.235 | 0.179 |
| XGBoost | Validation | 0.737 | 0.151 | 0.0387 | 0.1079 | 0.201 | 0.284 | 0.236 |
| XGBoost | Test | 0.715 | 0.124 | 0.0401 | 0.1079 | 0.165 | 0.228 | 0.192 |

- **Baseline:** a constant score equal to the training positive rate (0.0449). At its
  threshold, every lead is predicted positive.
- **Thresholds:** each model's threshold maximises F1 on validation. If several
  thresholds tie, the higher one is used.
- **XGBoost:**
  - `max_depth` 3 was chosen over 5 on validation PR-AUC (0.151 vs 0.145).
  - Early stopping on validation log loss settled at best iteration 254, out of a
    maximum of 1,000.

**Held-out test:** XGBoost depth, early stopping and every threshold were chosen on
validation only. The thresholds were frozen before the test split was scored, and test
was scored once, at the end.

## Interpretation

- **XGBoost is the current leading candidate.** It is ahead of Logistic Regression on
  every test metric (ROC-AUC, PR-AUC, Brier, precision and F1) except recall, where
  Logistic Regression is ahead (0.235 vs 0.228). Its test PR-AUC is about 2.8× the
  positive rate.
- **Logistic Regression remains the interpretable baseline.** It performs reasonably
  close to XGBoost and is simpler to explain.
- **Both models carry real signal over the naive baseline.** The gain in Brier score is
  small, as expected for a rare (about 4%) label. Precision at the chosen thresholds is
  still low (14–17%).
- **Validation performance is slightly higher than test performance.** Test scores are a 
  little lower than validation for both models, consistent with selection on validation.

## Limitations

- **Synthetic data:** the generator's conversion behaviour is simulated, not fitted to
  real CRM data, so the absolute numbers say nothing about production performance.
- **Single split:** one chronological split with no backtesting over several periods,
  so there are no confidence intervals or variance estimates.
- **Small, reused validation set:** about 1,100 positives, used for early stopping,
  depth and threshold.
- **Correlated rows:** each lead contributes up to 25 snapshots. A lead can also appear
  in more than one split, at different T.
- **Threshold rule:** maximising F1 is a default rather than a business-cost choice. No
  probability calibration step was applied beyond what the Brier score shows.
- **Features:**
  - lead attributes never change over time in the synthetic data
  - no Notes feature yet
  - snapshot eligibility relies on the synthetic terminal-time and observation-end
    conventions
- **Not yet in place:** models are not tracked in MLflow, registered or served online.
