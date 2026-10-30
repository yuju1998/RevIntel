"""Purged chronological split: exact boundaries and no label-window leakage."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.training.splits import SPLITS, assign_splits, purged_chronological_split

DAY = timedelta(days=1)
US = timedelta(microseconds=1)
START = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW = timedelta(days=30)


def _timestamps(values: list[datetime]) -> pd.Series:
    return pd.Series(pd.to_datetime(values, utc=True))


def test_label_window_must_end_strictly_before_the_next_split() -> None:
    validation_start = START + 100 * DAY
    test_start = START + 150 * DAY
    t = _timestamps(
        [
            validation_start - WINDOW - US,  # window ends just before validation: train
            validation_start - WINDOW,  # window ends exactly at validation start: purged
            validation_start - US,  # purged
            validation_start,  # validation
            test_start - WINDOW - US,  # validation
            test_start - WINDOW,  # purged
            test_start,  # test
        ]
    )

    masks = assign_splits(t, pd.Timestamp(validation_start), pd.Timestamp(test_start), WINDOW)

    assert np.flatnonzero(masks["train"]).tolist() == [0]
    assert np.flatnonzero(masks["purged_train_validation"]).tolist() == [1, 2]
    assert np.flatnonzero(masks["validation"]).tolist() == [3, 4]
    assert np.flatnonzero(masks["purged_validation_test"]).tolist() == [5]
    assert np.flatnonzero(masks["test"]).tolist() == [6]


def test_split_partitions_rows_without_boundary_leakage() -> None:
    # Irregular, repeated timestamps (several snapshots can share T).
    offsets = np.sort(np.random.default_rng(0).integers(0, 400 * 86_400, 5_000))
    t = _timestamps([START + timedelta(seconds=int(s)) for s in offsets])

    split = purged_chronological_split(t)

    parts = [set(split.rows(name).tolist()) for name in SPLITS]
    assert all(not a & b for i, a in enumerate(parts) for b in parts[i + 1 :])
    purged = split.purged_train_validation + split.purged_validation_test
    assert sum(map(len, parts)) + purged == len(t)
    train, validation, test = (t.iloc[split.rows(name)] for name in SPLITS)
    assert (train + WINDOW).max() < split.validation_start <= validation.min()
    assert (validation + WINDOW).max() < split.test_start <= test.min()
    assert split.validation_start == t.quantile(0.70)
    assert split.test_start == t.quantile(0.85)


def test_split_rejects_bad_input() -> None:
    naive = pd.Series(pd.to_datetime([START.replace(tzinfo=None)] * 3))
    with pytest.raises(ValueError, match="timezone-aware"):
        purged_chronological_split(naive)
    with pytest.raises(ValueError, match="fraction"):
        purged_chronological_split(
            _timestamps([START]), train_fraction=0.9, validation_fraction=0.2
        )
    with pytest.raises(ValueError, match="empty split"):
        purged_chronological_split(_timestamps([START + i * DAY for i in range(10)]))
