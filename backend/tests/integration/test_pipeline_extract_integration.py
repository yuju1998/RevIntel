"""Pipeline extraction against PostgreSQL (revintel_test): the materialized source snapshot
yields exactly the features a direct database build produces."""

from pathlib import Path

import pytest

from ml.features.build_features import SourceManifest, build
from ml.pipeline import lead_conversion as pipeline
from ml.synthetic.config import GenerationConfig
from ml.synthetic.generator import generate
from ml.synthetic.summary import build_manifest, sanity_counters, summary_stats, write_manifest
from ml.synthetic.writer import write_dataset

pytestmark = pytest.mark.integration


def test_extract_then_build_matches_a_direct_build(
    migrated_database: str, clean_tables: None, tmp_path: Path
) -> None:
    dataset = generate(
        GenerationConfig(seed=13, accounts=15, contacts=45, leads=250, activities=1200, reps=3)
    )
    write_dataset(dataset, migrated_database)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        build_manifest(dataset, summary_stats(dataset), sanity_counters(dataset)), manifest_path
    )
    config = pipeline.PipelineConfig(
        pipeline_root=tmp_path / "runs",
        source_database_url=migrated_database,
        manifest_path=manifest_path,
        tracking_uri="unused",
        registered_model="unused",
        experiment="unused",
    )
    run_key = pipeline.run_key_for("manual__1")

    snapshot = pipeline.extract_crm_snapshot(run_key, config)
    features = pipeline.build_features(run_key, config)
    again = pipeline.extract_crm_snapshot(run_key, config)

    direct = build(migrated_database, SourceManifest.read(manifest_path), progress=lambda _: None)
    assert features["features_sha256"] == direct.metadata["features_sha256"]
    assert snapshot["row_counts"]["leads"] == 250
    assert snapshot["source_database"] == "revintel_test"
    assert again["reused"] and again["content_sha256"] == snapshot["content_sha256"]
