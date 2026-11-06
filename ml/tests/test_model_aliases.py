"""Manual challenger/champion workflow against a temporary SQLite registry (no server)."""

import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow import MlflowClient  # noqa: E402

from ml.training import promote, register  # noqa: E402
from ml.training.mlflow_config import DEFAULT_REGISTERED_MODEL, DEFAULT_TRACKING_URI  # noqa: E402

MODEL = "lead-model"
ROOT = Path(__file__).resolve().parents[2]
BASELINE_TAG = "validation_baseline_check"


def _uri(database: Path) -> str:
    return f"sqlite:///{database}"


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A registry with three candidate versions; copied per test (migrations are slow)."""
    database = tmp_path_factory.mktemp("registry") / "mlflow.db"
    client = MlflowClient(tracking_uri=_uri(database), registry_uri=_uri(database))
    client.create_registered_model(MODEL)
    for check, algorithm in (
        ("passed", "logistic_regression"),
        ("passed", "xgboost"),
        ("failed", "xgboost"),
    ):
        client.create_model_version(
            MODEL,
            source=str(database.parent),
            tags={"algorithm": algorithm, BASELINE_TAG: check},
        )
    return database


@pytest.fixture
def database(template: Path, tmp_path: Path) -> Path:
    return Path(shutil.copy(template, tmp_path / "mlflow.db"))


@pytest.fixture
def client(database: Path) -> MlflowClient:
    return MlflowClient(tracking_uri=_uri(database), registry_uri=_uri(database))


@pytest.fixture
def cli(database: Path) -> Callable[..., int]:
    def run(main: Callable[[list[str]], int], version: str = "1") -> int:
        return main(
            ["--version", version, "--registered-model", MODEL, "--tracking-uri", _uri(database)]
        )

    return run


def _aliases(client: MlflowClient) -> dict[str, str]:
    return {k: str(v) for k, v in client.get_registered_model(MODEL).aliases.items()}


def _tags(client: MlflowClient, version: str) -> dict[str, str]:
    return client.get_model_version(MODEL, version).tags


def _version_count(client: MlflowClient) -> int:
    return len(client.search_model_versions(f"name='{MODEL}'"))


# --- register ---------------------------------------------------------------------------------


def test_register_sets_challenger_without_creating_a_version_or_status(
    client: MlflowClient, cli: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli(register.main, "2") == 0

    assert _aliases(client) == {"challenger": "2"}
    assert _version_count(client) == 3
    assert _tags(client, "2") == {"algorithm": "xgboost", BASELINE_TAG: "passed"}
    assert "challenger: none -> version 2" in capsys.readouterr().out


def test_register_moves_the_alias_and_rerun_changes_nothing(
    client: MlflowClient, cli: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    cli(register.main, "1")
    capsys.readouterr()

    assert cli(register.main, "2") == 0
    assert "challenger: version 1 -> version 2" in capsys.readouterr().out
    assert cli(register.main, "2") == 0
    assert "nothing changed" in capsys.readouterr().out
    assert _aliases(client) == {"challenger": "2"}
    assert _version_count(client) == 3


def test_register_refuses_the_current_champion(
    client: MlflowClient, cli: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    client.set_registered_model_alias(MODEL, "champion", "1")

    assert cli(register.main, "1") == 2

    assert "it is the current champion" in capsys.readouterr().err
    assert _aliases(client) == {"champion": "1"}


def test_failed_baseline_check_warns_but_does_not_block(
    client: MlflowClient, cli: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli(register.main, "3") == 0
    assert "Warning" in capsys.readouterr().out
    assert cli(promote.main, "3") == 0

    assert "Warning" in capsys.readouterr().out
    assert _aliases(client) == {"champion": "3"}
    assert _tags(client, "3")[BASELINE_TAG] == "failed"


# --- promote ----------------------------------------------------------------------------------


def test_promote_assigns_champion_approves_and_clears_challenger(
    client: MlflowClient, cli: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    cli(register.main, "2")
    capsys.readouterr()

    assert cli(promote.main, "2") == 0

    assert _aliases(client) == {"champion": "2"}
    assert _tags(client, "2") == {
        "algorithm": "xgboost",
        BASELINE_TAG: "passed",
        "validation_status": "approved",
    }
    assert "validation_status" not in _tags(client, "1")
    output = capsys.readouterr().out
    assert "champion: none -> version 2" in output
    assert "challenger: version 2 -> none" in output


def test_replacing_a_champion_keeps_the_old_version_and_its_tags(
    client: MlflowClient, cli: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    for version in ("1", "2"):
        cli(register.main, version)
        cli(promote.main, version)

    assert _aliases(client) == {"champion": "2"}
    assert _version_count(client) == 3
    assert _tags(client, "1")["validation_status"] == "approved"
    output = capsys.readouterr().out
    assert "champion: version 1 -> version 2" in output
    assert "Version 1 is kept in the registry." in output


@pytest.mark.parametrize(
    ("challenger", "message"),
    [(None, "current challenger is none"), ("2", "current challenger is version 2")],
)
def test_promote_refuses_unless_the_version_is_the_challenger(
    client: MlflowClient,
    cli: Callable[..., int],
    capsys: pytest.CaptureFixture[str],
    challenger: str | None,
    message: str,
) -> None:
    if challenger:
        client.set_registered_model_alias(MODEL, "challenger", challenger)
    before = _aliases(client)

    assert cli(promote.main, "1") == 2

    assert message in capsys.readouterr().err
    assert _aliases(client) == before
    assert "validation_status" not in _tags(client, "1")


def test_promote_refuses_a_finished_promotion(
    client: MlflowClient, cli: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    cli(register.main, "1")
    cli(promote.main, "1")

    assert cli(promote.main, "1") == 2
    assert "already the champion" in capsys.readouterr().err


def test_promote_can_finish_after_a_partial_failure(
    client: MlflowClient, cli: Callable[..., int]
) -> None:
    # The alias removal failed last time: the version holds both aliases.
    client.set_model_version_tag(MODEL, "1", "validation_status", "approved")
    client.set_registered_model_alias(MODEL, "challenger", "1")
    client.set_registered_model_alias(MODEL, "champion", "1")

    assert cli(promote.main, "1") == 0
    assert _aliases(client) == {"champion": "1"}


# --- validation -------------------------------------------------------------------------------


@pytest.mark.parametrize("main", [register.main, promote.main])
def test_missing_version_or_model_is_refused(
    database: Path,
    client: MlflowClient,
    capsys: pytest.CaptureFixture[str],
    main: Callable[[list[str]], int],
) -> None:
    uri = ["--tracking-uri", _uri(database)]

    assert main(["--version", "9", "--registered-model", MODEL, *uri]) == 2
    assert "Version 9" in capsys.readouterr().err
    assert main(["--version", "1", "--registered-model", "no-such-model", *uri]) == 2
    assert "'no-such-model' does not exist" in capsys.readouterr().err
    assert _aliases(client) == {}


def test_version_that_is_not_ready_is_refused(
    database: Path, client: MlflowClient, cli: Callable[..., int]
) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE model_versions SET status = 'FAILED_REGISTRATION'")

    assert cli(register.main, "1") == 2
    assert _aliases(client) == {}


@pytest.mark.parametrize("version", ["0", "-1"])
def test_version_must_be_a_positive_integer(version: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        register.main([f"--version={version}"])

    assert exit_info.value.code == 2


def test_unreachable_server_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    code = promote.main(["--version", "1", "--tracking-uri", "http://127.0.0.1:9"])

    assert code == 2
    assert "scripts/start_mlflow.sh" in capsys.readouterr().err


def test_defaults_follow_the_existing_mlflow_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_MODEL_NAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    defaults = register.build_parser("x").parse_args(["--version", "1"])
    assert defaults.registered_model == DEFAULT_REGISTERED_MODEL
    assert defaults.tracking_uri == DEFAULT_TRACKING_URI

    monkeypatch.setenv("MLFLOW_MODEL_NAME", "other-model")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.test")
    configured = register.build_parser("x").parse_args(["--version", "1"])
    assert configured.registered_model == "other-model"
    assert configured.tracking_uri == "http://mlflow.test"


def test_modules_run_with_python_m_without_the_backend_on_the_path(
    database: Path, client: MlflowClient
) -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    args = ["--version", "2", "--registered-model", MODEL, "--tracking-uri", _uri(database)]

    for module in ("ml.training.register", "ml.training.promote"):
        result = subprocess.run(
            [sys.executable, "-m", module, *args], cwd=ROOT, env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

    assert _aliases(client) == {"champion": "2"}
