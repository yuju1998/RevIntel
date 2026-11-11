"""A reproducible on-disk snapshot of the feature pipeline's source rows.

It persists exactly what `extract_sources()` read in its read-only transaction, plus the
manifest fields the feature build needs. Features built from a snapshot are identical to
features built straight from the database (same `features_sha256`).
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import make_url

from ml.features.build_features import SourceExtract, SourceManifest

LEADS_FILE = "leads.parquet"
ACTIVITIES_FILE = "lead_activities.parquet"
SNAPSHOT_FILE = "snapshot.json"
_DATETIME_COLUMNS = {
    LEADS_FILE: ("created_at", "converted_at", "updated_at"),
    ACTIVITIES_FILE: ("occurred_at", "created_at"),
}


def _content_sha256(*frames: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.to_csv(index=False, lineterminator="\n").encode())
    return digest.hexdigest()


def write_source_snapshot(
    directory: Path, extract: SourceExtract, manifest: SourceManifest, database_url: str
) -> dict[str, Any]:
    """Write the extracted rows and their provenance; return the snapshot metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    extract.leads.to_parquet(directory / LEADS_FILE, index=False)
    extract.activities.to_parquet(directory / ACTIVITIES_FILE, index=False)
    metadata = {
        "source_database": make_url(database_url).database,  # name only, no credentials
        "row_counts": extract.row_counts,
        "content_sha256": _content_sha256(extract.leads, extract.activities),
        "manifest": {
            "path": str(manifest.path),
            "dataset_sha256": manifest.dataset_sha256,
            "generator_version": manifest.generator_version,
            "as_of": manifest.as_of.isoformat(),
            "max_lead_age_days": manifest.max_lead_age_days,
            "row_counts": manifest.row_counts,
        },
    }
    (directory / SNAPSHOT_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _read(directory: Path, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(directory / name)
    for column in _DATETIME_COLUMNS[name]:
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.as_unit("us")
    return frame


def read_source_snapshot(
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, SourceManifest]:
    """The snapshot's leads, lead activities and the manifest it was extracted under."""
    metadata = json.loads((directory / SNAPSHOT_FILE).read_text())
    leads, activities = _read(directory, LEADS_FILE), _read(directory, ACTIVITIES_FILE)
    if _content_sha256(leads, activities) != metadata["content_sha256"]:
        raise ValueError(f"source snapshot in {directory} does not match its content hash")
    saved = metadata["manifest"]
    manifest = SourceManifest(
        path=Path(saved["path"]),
        dataset_sha256=saved["dataset_sha256"],
        generator_version=saved["generator_version"],
        as_of=datetime.fromisoformat(saved["as_of"]),
        max_lead_age_days=saved["max_lead_age_days"],
        row_counts=saved["row_counts"],
    )
    return leads, activities, manifest
