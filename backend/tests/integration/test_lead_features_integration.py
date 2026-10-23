"""Feature pipeline against a real database (revintel_test): read-only extraction, parquet
output and offline/online parity through the shared SQL."""

import dataclasses
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError

from ml.features.build_features import (
    SourceManifest,
    SourceMismatchError,
    build,
    read_metadata,
    write_outputs,
)
from ml.features.feature_spec import ACTIVE_FEATURES, LABEL, OUTPUT_COLUMNS, to_records
from ml.features.online_features import fetch_online_features
from ml.features.sources import read_only_connection
from ml.synthetic.config import GenerationConfig
from ml.synthetic.generator import generate
from ml.synthetic.summary import build_manifest, sanity_counters, summary_stats, write_manifest
from ml.synthetic.writer import CRM_TABLES, write_dataset

pytestmark = pytest.mark.integration

CONFIG = GenerationConfig(
    seed=11, accounts=20, contacts=60, leads=300, activities=1500, reps=4, admins=1
)


@pytest.fixture
def source(migrated_database: str, clean_tables: None, tmp_path: Path) -> SourceManifest:
    dataset = generate(CONFIG)
    write_dataset(dataset, migrated_database)
    path = tmp_path / "manifest.json"
    manifest = build_manifest(dataset, summary_stats(dataset), sanity_counters(dataset))
    write_manifest(manifest, path)
    return SourceManifest.read(path)


def _checksum(engine: Engine) -> str:
    query = "SELECT md5(coalesce(string_agg(t::text, '|' ORDER BY t.id), '')) FROM {} t"
    with engine.connect() as connection:
        return ":".join(
            connection.execute(text(query.format(table))).scalar_one() for table in CRM_TABLES
        )


def _quiet(_: str) -> None:
    pass


def test_build_writes_parquet_matching_online_features(
    source: SourceManifest, migrated_database: str, db_engine: Engine, tmp_path: Path
) -> None:
    before = _checksum(db_engine)

    result = build(migrated_database, source, progress=_quiet)
    output = tmp_path / "lead_features.parquet"
    summary = write_outputs(result, output)

    assert _checksum(db_engine) == before  # extraction never writes
    assert summary.exists()
    stored = pq.read_table(output).to_pandas()
    assert tuple(stored.columns) == (*OUTPUT_COLUMNS, LABEL)
    metadata = read_metadata(output)
    assert metadata["active_features"] == list(ACTIVE_FEATURES)
    assert metadata["deferred_features"][0]["name"] == "notes_count_30d"
    assert metadata["source"]["dataset_sha256"] == source.dataset_sha256
    assert metadata["features_sha256"] == result.metadata["features_sha256"]
    pd.testing.assert_frame_equal(stored, result.table, check_dtype=False)

    rows = to_records(result.table)
    with read_only_connection(migrated_database) as connection:
        for row in rows[:: max(1, len(rows) // 40)]:
            at = row["prediction_timestamp"].to_pydatetime()
            online = fetch_online_features(connection, row["lead_id"], at)
            assert online == {name: row[name] for name in ACTIVE_FEATURES}

    again = build(migrated_database, source, progress=_quiet)
    assert again.metadata["features_sha256"] == result.metadata["features_sha256"]


def test_extraction_transaction_is_read_only(
    source: SourceManifest, migrated_database: str
) -> None:
    with read_only_connection(migrated_database) as connection:
        assert connection.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
        assert (
            connection.execute(text("SHOW transaction_isolation")).scalar_one() == "repeatable read"
        )
        with pytest.raises(DBAPIError, match="read-only transaction"):
            connection.execute(text("DELETE FROM activities"))


def test_database_not_matching_the_manifest_is_refused(
    source: SourceManifest, migrated_database: str
) -> None:
    other = dataclasses.replace(source, row_counts={**source.row_counts, "leads": 1})

    with pytest.raises(SourceMismatchError, match="do not match the manifest"):
        build(migrated_database, other, progress=_quiet)
