"""Purged chronological train/validation/test split on prediction_timestamp.

Boundaries are quantiles of T. Because each label covers (T, T + 30d], a snapshot belongs
to a split only if its whole label window ends strictly before the next split starts:

    train:       T + 30d < validation_start
    validation:  validation_start <= T  and  T + 30d < test_start
    test:        T >= test_start

Snapshots whose label window crosses a boundary are purged. The same lead may appear in
several splits (at different T); lead_id is never a feature.
"""

import hashlib
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ml.features.feature_spec import LABEL, LABEL_WINDOW

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class ChronologicalSplit:
    train: np.ndarray  # row positions
    validation: np.ndarray
    test: np.ndarray
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    purged_train_validation: int
    purged_validation_test: int
    train_fraction: float
    validation_fraction: float
    label_window: timedelta

    def rows(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def metadata(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Boundaries, ranges, counts, positive rates, purged rows and row-ID hashes."""
        splits = {}
        for name in SPLITS:
            part = frame.iloc[self.rows(name)]
            t = part["prediction_timestamp"]
            keys = (part["lead_id"].astype(str) + "@" + t.map(pd.Timestamp.isoformat)).sort_values()
            splits[name] = {
                "rows": len(part),
                "positives": int(part[LABEL].sum()),
                "positive_rate": round(float(part[LABEL].mean()), 6),
                "first_prediction_timestamp": t.min().isoformat(),
                "last_prediction_timestamp": t.max().isoformat(),
                "leads": int(part["lead_id"].nunique()),
                "row_ids_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
            }
        return {
            "method": "purged chronological split on prediction_timestamp",
            "fractions": {
                "train": self.train_fraction,
                "validation": self.validation_fraction,
                "test": round(1 - self.train_fraction - self.validation_fraction, 6),
            },
            "label_window_days": self.label_window.days,
            "validation_start": self.validation_start.isoformat(),
            "test_start": self.test_start.isoformat(),
            "purged": {
                "train_validation": self.purged_train_validation,
                "validation_test": self.purged_validation_test,
            },
            "splits": splits,
        }


def assign_splits(
    t: pd.Series,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
    label_window: timedelta = LABEL_WINDOW,
) -> dict[str, pd.Series]:
    """Boolean masks for each split and for the rows purged at each boundary."""
    window_end = t + label_window
    train = window_end < validation_start
    validation = (t >= validation_start) & (window_end < test_start)
    return {
        "train": train,
        "validation": validation,
        "test": t >= test_start,
        "purged_train_validation": (t < validation_start) & ~train,
        "purged_validation_test": (t >= validation_start) & (t < test_start) & ~validation,
    }


def purged_chronological_split(
    timestamps: pd.Series,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    label_window: timedelta = LABEL_WINDOW,
) -> ChronologicalSplit:
    if not 0 < train_fraction < train_fraction + validation_fraction < 1:
        raise ValueError("need 0 < train_fraction < train + validation fraction < 1")
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("prediction_timestamp must be timezone-aware")
    t = timestamps.reset_index(drop=True)
    validation_start = t.quantile(train_fraction)
    test_start = t.quantile(train_fraction + validation_fraction)
    masks = assign_splits(t, validation_start, test_start, label_window)

    split = ChronologicalSplit(
        train=np.flatnonzero(masks["train"]),
        validation=np.flatnonzero(masks["validation"]),
        test=np.flatnonzero(masks["test"]),
        validation_start=validation_start,
        test_start=test_start,
        purged_train_validation=int(masks["purged_train_validation"].sum()),
        purged_validation_test=int(masks["purged_validation_test"].sum()),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        label_window=label_window,
    )
    empty = [name for name in SPLITS if not len(split.rows(name))]
    if empty:
        raise ValueError(f"empty split(s) after purging: {empty}")
    return split
