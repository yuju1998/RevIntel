"""Invariants (hard failures) and concise statistics for a built lead feature table."""

from typing import Any

import numpy as np
import pandas as pd

from ml.features.feature_spec import (
    ACTIVE_FEATURES,
    CATEGORICAL_FEATURES,
    DEFERRED_FEATURES,
    FEATURES_BY_NAME,
    KEY_COLUMNS,
    LABEL,
    NUMERIC_FEATURES,
    OUTPUT_COLUMNS,
    WINDOW_30D,
)
from ml.features.snapshots import SnapshotPolicy, lead_timeline

TABLE_COLUMNS = (*OUTPUT_COLUMNS, LABEL)
_COUNTS_30D = ("email_count_30d", "meeting_count_30d", "inbound_response_count_30d")


class FeatureValidationError(ValueError):
    """The feature table violates an invariant; nothing should be written."""


def validate_features(table: pd.DataFrame, leads: pd.DataFrame, policy: SnapshotPolicy) -> None:
    """Raise FeatureValidationError listing every violated invariant."""
    problems: list[str] = []

    def check(ok: bool | np.bool_, message: str) -> None:
        if not ok:
            problems.append(message)

    if tuple(table.columns) != TABLE_COLUMNS:
        raise FeatureValidationError(f"columns {list(table.columns)} != {list(TABLE_COLUMNS)}")
    check(len(table) > 0, "no rows")
    check(not table.duplicated(list(KEY_COLUMNS)).any(), "duplicate (lead_id, T) keys")
    check(
        all(spec.name not in table.columns for spec in DEFERRED_FEATURES),
        "deferred feature present",
    )
    for name in ACTIVE_FEATURES:
        if not FEATURES_BY_NAME[name].nullable:
            check(not table[name].isna().any(), f"{name} has NULLs")
    check(not table[LABEL].isna().any(), "label has NULLs")

    count_7d, count_30d = table["activity_count_7d"], table["activity_count_30d"]
    check((count_7d >= 0).all(), "negative counts")
    check((count_7d <= count_30d).all(), "activity_count_7d > activity_count_30d")
    for name in _COUNTS_30D:
        check(((table[name] >= 0) & (table[name] <= count_30d)).all(), f"{name} out of range")
    active_days = table["active_days_30d"]
    check(
        ((active_days >= 0) & (active_days <= np.minimum(count_30d, WINDOW_30D.days + 1))).all(),
        "active_days_30d out of range",
    )
    days_since = table["days_since_last_activity"]
    check(not (days_since < 0).any(), "negative days_since_last_activity")
    check(not (days_since.isna() & (count_30d > 0)).any(), "NULL days_since with 30d activity")
    check(
        not (days_since.notna() & (count_30d == 0) & (days_since < WINDOW_30D.days)).any(),
        "last activity inside the 30d window but no 30d activity",
    )

    # Snapshot policy, re-checked against the lead timeline.
    lead_info = pd.concat(
        [leads[["lead_id", "created_at", "converted_at"]], lead_timeline(leads, policy)], axis=1
    ).set_index("lead_id")
    joined = table[["lead_id", "prediction_timestamp", LABEL]].join(lead_info, on="lead_id")
    t = joined["prediction_timestamp"]
    window_end = t + policy.label_window
    terminal, observation_end = joined["terminal_at"], joined["observation_end"]
    converted = joined["converted_at"]
    check((window_end <= policy.as_of).all(), "label window extends past as_of")
    check(((t - joined["created_at"]) % policy.cadence == pd.Timedelta(0)).all(), "T off-grid")
    check((t > joined["created_at"]).all(), "T at or before created_at")
    check((terminal.isna() | (t < terminal)).all(), "snapshot after the lead's terminal time")
    resolved = terminal.notna() & (terminal <= window_end) & (terminal <= observation_end)
    check(((window_end <= observation_end) | resolved).all(), "unobservable label window")
    expected = converted.notna() & (t < converted) & (converted <= window_end)
    check((expected == joined[LABEL]).all(), "label mismatch")

    if problems:
        raise FeatureValidationError("; ".join(problems))


def availability_exclusions(snapshots: pd.DataFrame, activities: pd.DataFrame) -> dict[str, Any]:
    """How many (snapshot, activity) pairs with occurred_at <= T were not yet logged at T."""
    pairs = snapshots[["lead_id", "prediction_timestamp"]].merge(
        activities[["lead_id", "occurred_at", "created_at"]], on="lead_id"
    )
    t = pairs["prediction_timestamp"]
    happened = pairs["occurred_at"] <= t
    not_logged = happened & (pairs["created_at"] > t)
    return {
        "happened_by_t": int(happened.sum()),
        "excluded_not_logged_by_t": int(not_logged.sum()),
        "excluded_share": _r(not_logged.sum() / happened.sum()) if happened.any() else None,
    }


def roc_auc(scores: pd.Series, labels: pd.Series) -> float | None:
    """Rank-based ROC AUC over rows where the score is not NULL."""
    present = scores.notna()
    scores, labels = scores[present], labels[present]
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = scores.rank(method="average")
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _r(value: float) -> float:
    return round(float(value), 4)


def feature_statistics(table: pd.DataFrame, snapshots: pd.DataFrame) -> dict[str, Any]:
    """Concise build statistics. AUC is reported for numeric/count features only;
    categoricals get per-category share and positive rate (no target encoding)."""
    label = table[LABEL]
    per_lead = table.groupby("lead_id").size()
    by_month = table.groupby(table["prediction_timestamp"].dt.strftime("%Y-%m"))[LABEL]
    numeric = {}
    for name in NUMERIC_FEATURES:
        values = table[name].astype("float64")
        numeric[name] = {
            "mean": _r(values.mean()),
            "p50": _r(values.median()),
            "p95": _r(values.quantile(0.95)),
            "null_share": _r(values.isna().mean()),
            "zero_share": _r((values == 0).mean()),
            "auc": None if (auc := roc_auc(values, label)) is None else _r(auc),
        }
    categorical = {}
    for name in CATEGORICAL_FEATURES:
        values = table[name].astype(object).where(table[name].notna(), "<NULL>")
        grouped = label.groupby(values)
        categorical[name] = {
            "null_share": _r(table[name].isna().mean()),
            "categories": {
                str(key): {"share": _r(size / len(table)), "positive_rate": _r(rate)}
                for key, size, rate in zip(
                    grouped.size().index, grouped.size(), grouped.mean(), strict=True
                )
            },
        }
    past_horizon = snapshots["resolved_past_horizon"]
    return {
        "rows": len(table),
        "leads": int(per_lead.size),
        "snapshots_per_lead": {
            "min": int(per_lead.min()),
            "mean": _r(per_lead.mean()),
            "max": int(per_lead.max()),
        },
        "positive_rate": _r(label.mean()),
        "positive_rate_by_k": {
            int(k): _r(v) for k, v in label.groupby(snapshots["k"]).mean().items()
        },
        "positive_rate_by_month": {str(m): _r(v) for m, v in by_month.mean().items()},
        "resolved_past_horizon": {
            "rows": int(past_horizon.sum()),
            "positive_rate": _r(label[past_horizon].mean()) if past_horizon.any() else None,
        },
        "numeric_features": numeric,
        "categorical_features": categorical,
    }
