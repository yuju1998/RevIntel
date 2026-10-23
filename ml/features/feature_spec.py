"""Lead conversion feature specification and the one shared feature computation.

Offline training data and online scoring both call `compute_features`, so a feature has
exactly one definition and one implementation.

Time semantics (all timestamps are timezone-aware UTC; a day is exactly 24 hours):
- An activity is *known at T* when `created_at <= T AND occurred_at <= T`: it must have
  existed in the CRM at prediction time. `activities.created_at` is an availability input
  only; it never shapes a feature value.
- Known activities are then placed in windows by `occurred_at` (event time). Windows are
  left-exclusive and right-inclusive: 7d is (T - 7d, T], 30d is (T - 30d, T].
- The label window is (T, T + 30d] on `leads.converted_at` (see `ml.features.snapshots`).
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa

from app.models.enums import ActivityDirection, ActivityType, LeadSource

FEATURE_SPEC_VERSION = "1.0.0"

DAY = timedelta(days=1)
WINDOW_7D = timedelta(days=7)
WINDOW_30D = timedelta(days=30)
LABEL_WINDOW = timedelta(days=30)

KEY_COLUMNS = ("lead_id", "prediction_timestamp")
LABEL = "converted_within_30d"
AVAILABILITY_RULE = "activities.created_at <= T AND activities.occurred_at <= T"
WINDOW_SEMANTICS = {
    "7d": "occurred_at in (T - 7d, T]",
    "30d": "occurred_at in (T - 30d, T]",
    "label": "converted_at in (T, T + 30d]",
}

# Columns compute_features reads; anything else in its input frames is ignored.
LEAD_INPUT_COLUMNS = ("lead_source", "industry", "company_size", "estimated_value", "created_at")
ACTIVITY_INPUT_COLUMNS = ("lead_id", "occurred_at", "created_at", "activity_type", "direction")

# Columns whose values may shape a model feature.
MODEL_FEATURE_SOURCES = frozenset(
    {
        "leads.lead_source",
        "leads.industry",
        "leads.company_size",
        "leads.estimated_value",
        "leads.created_at",
        "activities.occurred_at",
        "activities.activity_type",
        "activities.direction",
        "notes.created_at",
    }
)
# Control inputs: only decide whether a row was known at T.
AVAILABILITY_INPUTS = frozenset({"activities.created_at"})
# Control inputs: only decide snapshot eligibility and the label.
SNAPSHOT_CONTROL_INPUTS = frozenset({"leads.status", "leads.converted_at", "leads.updated_at"})
# Never a source of any feature value, and never an output column.
FORBIDDEN_FEATURE_INPUTS = frozenset(
    {
        "leads.status",
        "leads.converted_at",
        "leads.updated_at",
        "leads.owner_id",
        "activities.created_at",
    }
)

_ACTIVITY_VALUES = frozenset({"activities.occurred_at"})


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str
    kind: Literal["categorical", "numeric", "count"]
    feature_sources: frozenset[str]
    availability_inputs: frozenset[str]
    window: timedelta | None
    nullable: bool
    description: str
    enabled: bool = True
    disabled_reason: str | None = None


def _activity_count(name: str, window: timedelta, extra: frozenset[str], what: str) -> FeatureSpec:
    days = window.days
    return FeatureSpec(
        name,
        "int32",
        "count",
        _ACTIVITY_VALUES | extra,
        AVAILABILITY_INPUTS,
        window,
        False,
        f"Number of known {what} with occurred_at in (T - {days}d, T].",
    )


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "lead_source",
        "category",
        "categorical",
        frozenset({"leads.lead_source"}),
        frozenset(),
        None,
        False,
        "Lead source; fixed categories in LeadSource order.",
    ),
    FeatureSpec(
        "industry",
        "string",
        "categorical",
        frozenset({"leads.industry"}),
        frozenset(),
        None,
        True,
        "Lead industry (open set).",
    ),
    FeatureSpec(
        "company_size",
        "Int64",
        "numeric",
        frozenset({"leads.company_size"}),
        frozenset(),
        None,
        True,
        "Lead company size (employees).",
    ),
    FeatureSpec(
        "estimated_value",
        "decimal128(14,2)",
        "numeric",
        frozenset({"leads.estimated_value"}),
        frozenset(),
        None,
        True,
        "Estimated deal value; exact decimal, cast to float inside the model pipeline.",
    ),
    FeatureSpec(
        "lead_age_days",
        "float64",
        "numeric",
        frozenset({"leads.created_at"}),
        frozenset(),
        None,
        False,
        "(T - created_at) in days.",
    ),
    _activity_count("activity_count_7d", WINDOW_7D, frozenset(), "activities"),
    _activity_count("activity_count_30d", WINDOW_30D, frozenset(), "activities"),
    _activity_count(
        "email_count_30d",
        WINDOW_30D,
        frozenset({"activities.activity_type"}),
        "activities with activity_type = EMAIL",
    ),
    _activity_count(
        "meeting_count_30d",
        WINDOW_30D,
        frozenset({"activities.activity_type"}),
        "activities with activity_type = MEETING (DEMO excluded)",
    ),
    _activity_count(
        "inbound_response_count_30d",
        WINDOW_30D,
        frozenset({"activities.direction"}),
        "activities with direction = INBOUND",
    ),
    FeatureSpec(
        "active_days_30d",
        "int32",
        "count",
        _ACTIVITY_VALUES,
        AVAILABILITY_INPUTS,
        WINDOW_30D,
        False,
        "Distinct UTC calendar dates with >= 1 known activity with occurred_at in "
        "(T - 30d, T].",
    ),
    FeatureSpec(
        "days_since_last_activity",
        "float64",
        "numeric",
        _ACTIVITY_VALUES,
        AVAILABILITY_INPUTS,
        None,
        True,
        "(T - latest occurred_at among known activities) in days; NULL if there is none.",
    ),
    FeatureSpec(
        "notes_count_30d",
        "int32",
        "count",
        frozenset({"notes.created_at"}),
        frozenset(),
        WINDOW_30D,
        False,
        "Number of notes with created_at in (T - 30d, T].",
        enabled=False,
        disabled_reason=(
            "synthetic dataset currently contains no Notes; enable after Notes generation "
            "is implemented"
        ),
    ),
)

ACTIVE_FEATURES: tuple[str, ...] = tuple(spec.name for spec in FEATURES if spec.enabled)
DEFERRED_FEATURES: tuple[FeatureSpec, ...] = tuple(spec for spec in FEATURES if not spec.enabled)
FEATURES_BY_NAME = {spec.name: spec for spec in FEATURES}
CATEGORICAL_FEATURES = tuple(
    name for name in ACTIVE_FEATURES if FEATURES_BY_NAME[name].kind == "categorical"
)
NUMERIC_FEATURES = tuple(
    name for name in ACTIVE_FEATURES if FEATURES_BY_NAME[name].kind != "categorical"
)
OUTPUT_COLUMNS = (*KEY_COLUMNS, *ACTIVE_FEATURES)

_PANDAS_DTYPES: dict[str, Any] = {
    "category": pd.CategoricalDtype([source.value for source in LeadSource]),
    "string": pd.StringDtype(),
    "Int64": pd.Int64Dtype(),
    "decimal128(14,2)": pd.ArrowDtype(pa.decimal128(14, 2)),
    "float64": np.dtype("float64"),
    "int32": np.dtype("int32"),
}


def spec_fingerprint(features: tuple[FeatureSpec, ...] = FEATURES) -> str:
    """SHA-256 of the canonical specification (any definition change changes it)."""
    canonical = {
        "version": FEATURE_SPEC_VERSION,
        "availability_rule": AVAILABILITY_RULE,
        "window_semantics": WINDOW_SEMANTICS,
        "features": [
            {
                "name": spec.name,
                "dtype": spec.dtype,
                "kind": spec.kind,
                "feature_sources": sorted(spec.feature_sources),
                "availability_inputs": sorted(spec.availability_inputs),
                "window_seconds": spec.window.total_seconds() if spec.window else None,
                "nullable": spec.nullable,
                "enabled": spec.enabled,
            }
            for spec in features
        ],
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def spec_metadata() -> dict[str, Any]:
    return {
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "feature_spec_fingerprint": spec_fingerprint(),
        "active_features": list(ACTIVE_FEATURES),
        "deferred_features": [
            {"name": spec.name, "reason": spec.disabled_reason} for spec in DEFERRED_FEATURES
        ],
        "label": LABEL,
        "label_window_days": LABEL_WINDOW.days,
        "window_semantics": WINDOW_SEMANTICS,
        "availability_rule": AVAILABILITY_RULE,
    }


# --- shared computation -----------------------------------------------------------------------


def _utc(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name} must be timezone-aware, got {series.dtype}")
    return series.dt.tz_convert("UTC").dt.as_unit("us")


def _days(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / DAY.total_seconds()


def compute_features(snapshots: pd.DataFrame, activities: pd.DataFrame) -> pd.DataFrame:
    """Features for each snapshot row, in input order, as `OUTPUT_COLUMNS`.

    `snapshots` needs `lead_id`, `prediction_timestamp` and `LEAD_INPUT_COLUMNS`;
    `activities` needs `ACTIVITY_INPUT_COLUMNS` (lead activities). Other columns are
    ignored. Activities not known at T (see module docstring) never contribute, whatever
    the caller passes in.
    """
    snaps = snapshots.loc[:, [*KEY_COLUMNS, *LEAD_INPUT_COLUMNS]].reset_index(drop=True)
    snaps["lead_id"] = snaps["lead_id"].astype(str)
    t = _utc(snaps["prediction_timestamp"], "prediction_timestamp")
    created = _utc(snaps["created_at"], "leads.created_at")

    acts = activities.loc[:, list(ACTIVITY_INPUT_COLUMNS)]
    acts = acts.assign(
        lead_id=acts["lead_id"].astype(str),
        occurred_at=_utc(acts["occurred_at"], "activities.occurred_at"),
        created_at=_utc(acts["created_at"], "activities.created_at"),
        activity_type=acts["activity_type"].astype(str),
        direction=acts["direction"].astype(str),
    )

    pairs = pd.DataFrame({"row": snaps.index, "lead_id": snaps["lead_id"], "t": t}).merge(
        acts, on="lead_id", how="inner"
    )
    known = (pairs["created_at"] <= pairs["t"]) & (pairs["occurred_at"] <= pairs["t"])
    # created_at is dropped here: past this point only event-time values are used.
    pairs = pairs.loc[known, ["row", "t", "occurred_at", "activity_type", "direction"]]
    in_7d = pairs["occurred_at"] > pairs["t"] - WINDOW_7D
    in_30d = pairs["occurred_at"] > pairs["t"] - WINDOW_30D
    flags = pd.DataFrame(
        {
            "row": pairs["row"],
            "activity_count_7d": in_7d,
            "activity_count_30d": in_30d,
            "email_count_30d": in_30d & (pairs["activity_type"] == ActivityType.EMAIL),
            "meeting_count_30d": in_30d & (pairs["activity_type"] == ActivityType.MEETING),
            "inbound_response_count_30d": in_30d
            & (pairs["direction"] == ActivityDirection.INBOUND),
            "date_30d": pairs["occurred_at"].dt.floor("D").where(in_30d),
            "occurred_at": pairs["occurred_at"],
        }
    )
    grouped = flags.groupby("row")
    counts = grouped[
        [
            "activity_count_7d",
            "activity_count_30d",
            "email_count_30d",
            "meeting_count_30d",
            "inbound_response_count_30d",
        ]
    ].sum()
    counts["active_days_30d"] = grouped["date_30d"].nunique()
    counts = counts.reindex(snaps.index, fill_value=0)
    last = grouped["occurred_at"].max().reindex(snaps.index)

    computed = pd.DataFrame(
        {
            "lead_id": snaps["lead_id"],
            "prediction_timestamp": t,
            "lead_source": snaps["lead_source"].astype(str),
            "industry": snaps["industry"],
            "company_size": snaps["company_size"],
            "estimated_value": snaps["estimated_value"],
            "lead_age_days": _days(t - created),
            **{name: counts[name] for name in counts.columns},
            "days_since_last_activity": _days(t - last),
        }
    )
    missing = set(ACTIVE_FEATURES) - set(computed.columns)
    if missing:
        raise RuntimeError(f"enabled features without an implementation: {sorted(missing)}")
    return _cast(computed)


def _cast(frame: pd.DataFrame) -> pd.DataFrame:
    unknown = set(frame["lead_source"]) - {source.value for source in LeadSource}
    if unknown:
        raise ValueError(f"unknown lead_source values: {sorted(unknown)}")
    out = frame.loc[:, list(OUTPUT_COLUMNS)].copy()
    for name in ACTIVE_FEATURES:
        dtype = _PANDAS_DTYPES[FEATURES_BY_NAME[name].dtype]
        column = out[name]
        if isinstance(dtype, pd.ArrowDtype):
            column = column.astype(object).where(column.notna(), None)
        out[name] = column.astype(dtype)
    return out


def _python(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Plain Python rows (None for missing; Decimal for money) for comparisons and serving."""
    columns = {name: frame[name].astype(object).tolist() for name in frame.columns}
    return [
        {name: _python(columns[name][i]) for name in frame.columns} for i in range(len(frame))
    ]
