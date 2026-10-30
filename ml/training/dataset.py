"""Load the lead feature table and enforce the feature contract before training."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ml.features.build_features import METADATA_KEY, features_sha256
from ml.features.feature_spec import (
    ACTIVE_FEATURES,
    FEATURE_SPEC_VERSION,
    KEY_COLUMNS,
    LABEL,
    spec_fingerprint,
)

FEATURE_TABLE_COLUMNS = (*KEY_COLUMNS, *ACTIVE_FEATURES, LABEL)


class FeatureSchemaError(ValueError):
    """The feature table or model does not match the current feature specification."""


@dataclass(frozen=True)
class FeatureTable:
    path: Path
    frame: pd.DataFrame
    metadata: dict[str, Any]


def check_spec_metadata(metadata: dict[str, Any], what: str) -> None:
    """Fail unless `metadata` was produced by the current feature specification."""
    expected = {
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "feature_spec_fingerprint": spec_fingerprint(),
        "active_features": list(ACTIVE_FEATURES),
        "label": LABEL,
    }
    mismatched = {
        key: {"found": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatched:
        raise FeatureSchemaError(
            f"{what} does not match the current feature specification: {mismatched}. "
            "Rebuild the features (and retrain) with the current code."
        )


def check_feature_table(frame: pd.DataFrame, metadata: dict[str, Any]) -> None:
    check_spec_metadata(metadata, "Feature table")
    if tuple(frame.columns) != FEATURE_TABLE_COLUMNS:
        raise FeatureSchemaError(
            f"Feature table columns {list(frame.columns)} != expected {list(FEATURE_TABLE_COLUMNS)}"
        )
    if features_sha256(frame) != metadata.get("features_sha256"):
        raise FeatureSchemaError(
            "Feature table content does not match its recorded features_sha256; "
            "the file was modified after it was built."
        )


def load_feature_table(path: Path) -> FeatureTable:
    table = pq.read_table(path)
    raw = (table.schema.metadata or {}).get(METADATA_KEY)
    if raw is None:
        raise FeatureSchemaError(f"{path} has no lead feature metadata")
    metadata = json.loads(raw)
    frame = table.to_pandas()
    check_feature_table(frame, metadata)
    return FeatureTable(path=path, frame=frame, metadata=metadata)


def features_and_label(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Model inputs (active features only; never lead_id or T) and the boolean label."""
    return frame.loc[:, list(ACTIVE_FEATURES)], frame[LABEL].to_numpy(dtype=bool)
