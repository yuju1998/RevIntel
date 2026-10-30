"""Build the point-in-time lead conversion feature table from the synthetic CRM dataset.

    SYNTHETIC_DATABASE_URL=postgresql+psycopg://revintel:revintel@localhost:5432/revintel_ml \\
        python scripts/build_lead_features.py

Reads (never writes) the database in one read-only transaction, checks it matches the
synthetic dataset manifest, derives snapshots and labels, computes features with the
shared `compute_features`, validates the result and writes
`data/processed/lead_features.parquet` plus a JSON summary.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.features.feature_spec import LABEL, compute_features, spec_metadata
from ml.features.snapshots import SnapshotPolicy, build_snapshots
from ml.features.sources import (
    UnsafeSourceError,
    load_lead_activities,
    load_leads,
    read_only_connection,
    source_row_counts,
)
from ml.features.validation import (
    FeatureValidationError,
    availability_exclusions,
    feature_statistics,
    validate_features,
)
from ml.synthetic.conventions import OBSERVATION_END_CONVENTION, TERMINAL_TIME_CONVENTION

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data" / "raw" / "synthetic_crm_manifest.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "lead_features.parquet"
METADATA_KEY = b"revintel.lead_features"
ROW_GROUP_SIZE = 100_000


class SourceMismatchError(RuntimeError):
    """The database does not hold the dataset the manifest describes."""


@dataclass(frozen=True)
class SourceManifest:
    path: Path
    dataset_sha256: str
    generator_version: str
    as_of: datetime
    max_lead_age_days: int
    row_counts: dict[str, int]

    @classmethod
    def read(cls, path: Path) -> SourceManifest:
        data = json.loads(path.read_text())
        as_of = datetime.fromisoformat(data["config"]["as_of"])
        if as_of.tzinfo is None:
            raise ValueError(f"{path}: as_of must be timezone-aware")
        return cls(
            path=path,
            dataset_sha256=data["dataset_sha256"],
            generator_version=data["generator_version"],
            as_of=as_of,
            max_lead_age_days=int(data["config"]["max_lead_age_days"]),
            row_counts={
                "leads": int(data["row_counts"]["leads"]),
                "activities": int(data["row_counts"]["activities"]),
                "lead_activities": int(data["stats"]["activities"]["lead_activities"]),
            },
        )


@dataclass(frozen=True)
class FeatureBuild:
    table: pd.DataFrame
    metadata: dict[str, Any]
    statistics: dict[str, Any]


def features_sha256(table: pd.DataFrame) -> str:
    """Content hash of the (sorted) feature rows; independent of parquet file layout."""
    return hashlib.sha256(table.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def build(
    database_url: str, manifest: SourceManifest, progress: Callable[[str], None] = print
) -> FeatureBuild:
    policy = SnapshotPolicy(as_of=manifest.as_of, max_lead_age_days=manifest.max_lead_age_days)
    started = time.perf_counter()
    with read_only_connection(database_url) as connection:
        counts = source_row_counts(connection)
        if counts != manifest.row_counts:
            raise SourceMismatchError(
                f"Database row counts {counts} do not match the manifest {manifest.row_counts}; "
                "is the database the synthetic CRM dataset this manifest describes?"
            )
        leads = load_leads(connection)
        activities = load_lead_activities(connection)
    progress(f"  loaded {len(leads):,} leads, {len(activities):,} lead activities (read-only)")

    snapshots = build_snapshots(leads, policy)
    frame = snapshots.frame
    progress(f"  snapshots: {len(frame):,}")
    table = compute_features(frame, activities)
    table[LABEL] = frame[LABEL].to_numpy()
    validate_features(table, leads, policy)

    statistics = feature_statistics(table, frame)
    statistics["terminal_after_horizon_excluded"] = snapshots.terminal_after_horizon_excluded
    statistics["availability"] = availability_exclusions(frame, activities)
    statistics["build_seconds"] = round(time.perf_counter() - started, 2)
    metadata = {
        **spec_metadata(),
        "snapshot_policy": policy.metadata(),
        "conventions": {
            "terminal_time": TERMINAL_TIME_CONVENTION,
            "observation_end": OBSERVATION_END_CONVENTION,
        },
        "source": {
            "manifest": str(manifest.path),
            "dataset_sha256": manifest.dataset_sha256,
            "generator_version": manifest.generator_version,
        },
        "features_sha256": features_sha256(table),
        "row_count": len(table),
        "positive_rate": statistics["positive_rate"],
        "git_commit": git_commit(),
    }
    return FeatureBuild(table=table, metadata=metadata, statistics=statistics)


def write_outputs(result: FeatureBuild, output: Path) -> Path:
    """Write the parquet (atomically) and its JSON summary; return the summary path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(result.table, preserve_index=False)
    schema_metadata = {**(table.schema.metadata or {}), METADATA_KEY: json.dumps(result.metadata)}
    table = table.replace_schema_metadata(schema_metadata)
    tmp = output.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", row_group_size=ROW_GROUP_SIZE)
    tmp.replace(output)
    summary = output.with_name(f"{output.stem}_summary.json")
    summary.write_text(
        json.dumps({"metadata": result.metadata, "statistics": result.statistics}, indent=2)
        + "\n"
    )
    return summary


def read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(pq.read_schema(path).metadata[METADATA_KEY])


def _format(result: FeatureBuild) -> str:
    stats, meta = result.statistics, result.metadata
    lines = [
        f"Rows: {stats['rows']:,} snapshots over {stats['leads']:,} leads "
        f"(per lead {stats['snapshots_per_lead']})",
        f"Positive rate ({LABEL}): {stats['positive_rate']}",
        f"Kept past the observation horizon (terminal outcome resolved): "
        f"{stats['resolved_past_horizon']}",
        f"Excluded, terminal after the observation horizon: "
        f"{stats['terminal_after_horizon_excluded']}",
        f"Availability: {stats['availability']}",
        "Numeric features (mean / p50 / p95 / null / zero / AUC):",
        *(
            f"  {name:<28} {s['mean']} / {s['p50']} / {s['p95']} / {s['null_share']} / "
            f"{s['zero_share']} / {s['auc']}"
            for name, s in stats["numeric_features"].items()
        ),
        "Categorical features (null share; positive rate by category):",
        *(
            f"  {name:<28} {s['null_share']}; "
            + ", ".join(f"{k}={v['positive_rate']}" for k, v in s["categories"].items())
            for name, s in stats["categorical_features"].items()
        ),
        f"Feature spec: {meta['feature_spec_version']} ({meta['feature_spec_fingerprint']})",
        f"Deferred features: {[d['name'] for d in meta['deferred_features']]}",
        f"Source dataset SHA-256: {meta['source']['dataset_sha256']}",
        f"Features SHA-256: {meta['features_sha256']}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("SYNTHETIC_DATABASE_URL"),
        help="source database (default: $SYNTHETIC_DATABASE_URL; DATABASE_URL is never used)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if not args.database_url:
        print(
            "Error: no source database. Pass --database-url or set SYNTHETIC_DATABASE_URL "
            "(DATABASE_URL is deliberately ignored).",
            file=sys.stderr,
        )
        return 2
    started = time.perf_counter()
    try:
        manifest = SourceManifest.read(args.manifest)
        print(f"Building lead features from {args.manifest} (as_of {manifest.as_of.isoformat()})")
        result = build(args.database_url, manifest)
    except (
        FileNotFoundError,
        UnsafeSourceError,
        SourceMismatchError,
        FeatureValidationError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    summary = write_outputs(result, args.output)
    print(_format(result))
    print(f"Wrote {args.output} and {summary}")
    print(f"Total time: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
