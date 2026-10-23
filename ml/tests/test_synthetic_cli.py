"""CLI arguments and target-database guards (no database)."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from ml.synthetic.writer import UnsafeTargetError, require_safe_database_url

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_crm_data.py"
SMALL_ARGS = ["--accounts", "10", "--contacts", "30", "--leads", "40", "--activities", "400"]


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.delenv("SYNTHETIC_DATABASE_URL", raising=False)
    spec = importlib.util.spec_from_file_location("generate_crm_data", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://u:p@localhost:5432/revintel_test",
        "postgresql+psycopg://u:p@localhost:5432/revintel_ml",
    ],
)
def test_safe_targets_are_accepted(url: str) -> None:
    assert require_safe_database_url(url).startswith("revintel_")


@pytest.mark.parametrize(
    "database", ["revintel", "postgres", "revintel_testing", "revintel_ml_prod", "mlflow"]
)
def test_unsafe_targets_are_refused(database: str) -> None:
    with pytest.raises(UnsafeTargetError, match="Refusing"):
        require_safe_database_url(f"postgresql+psycopg://u:p@localhost:5432/{database}")


def test_dev_database_url_is_never_used(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/revintel")

    assert cli.main(SMALL_ARGS) == 2
    assert "DATABASE_URL is deliberately ignored" in capsys.readouterr().err


def test_unsafe_database_url_is_refused_before_generating(
    cli: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    url = "postgresql+psycopg://u:p@localhost:5432/revintel"

    assert cli.main([*SMALL_ARGS, "--database-url", url]) == 2
    assert "Refusing" in capsys.readouterr().err


def test_dry_run_writes_nothing_but_the_manifest(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(*_: object, **__: object) -> None:
        raise AssertionError("dry run must not write to a database")

    monkeypatch.setattr("ml.synthetic.writer.write_dataset", fail)
    manifest = tmp_path / "manifest.json"

    assert cli.main([*SMALL_ARGS, "--dry-run", "--manifest", str(manifest)]) == 0

    data = json.loads(manifest.read_text())
    assert data["row_counts"]["activities"] == 400
    assert len(data["dataset_sha256"]) == 64
    assert all(value == 0 for value in data["sanity"].values())
    assert "Synthetic-dataset convention" in data["conventions"]["terminal_time"]


def test_dry_run_hash_is_reproducible(cli: ModuleType, tmp_path: Path) -> None:
    hashes = []
    for name in ("a.json", "b.json"):
        path = tmp_path / name
        assert cli.main([*SMALL_ARGS, "--dry-run", "--manifest", str(path)]) == 0
        hashes.append(json.loads(path.read_text())["dataset_sha256"])

    assert hashes[0] == hashes[1]


def test_activity_budget_overflow_exits_with_error(
    cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    args = ["--accounts", "10", "--contacts", "30", "--leads", "400", "--activities", "5"]

    assert cli.main([*args, "--dry-run", "--manifest", str(tmp_path / "m.json")]) == 2
    assert "not recalibrated" in capsys.readouterr().err


def test_naive_as_of_is_rejected(cli: ModuleType) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--as-of", "2026-01-01T00:00:00", "--dry-run"])
