"""API container configuration (static checks; no Docker needed)."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
REQUIREMENTS = ROOT / "requirements"


def _name(requirement: str) -> str:
    return re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def _requirements(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            names |= _requirements(path.parent / line[3:].strip())
        else:
            names.add(_name(line))
    return names


def _lock() -> dict[str, str]:
    pins = {}
    for line in (REQUIREMENTS / "lock.txt").read_text().splitlines():
        if "==" in line:
            name, version = line.strip().split("==", 1)
            pins[_name(name)] = version
    return pins


def test_every_api_requirement_is_pinned_in_the_lock() -> None:
    missing = _requirements(REQUIREMENTS / "api.txt") - set(_lock())

    assert not missing, f"not pinned in requirements/lock.txt: {sorted(missing)}"
    # Installed by the Dockerfile as CPU-only builds at their locked versions.
    assert {"torch", "xgboost"} <= set(_lock())


def test_api_image_excludes_dev_tools_and_the_mlflow_server() -> None:
    api = _requirements(REQUIREMENTS / "api.txt")

    assert not api & {"pytest", "pytest-asyncio", "pytest-cov", "ruff", "httpx2", "mlflow"}
    assert "mlflow-skinny" in api


def test_dockerfile_runs_offline_as_non_root_with_exec_form_fastapi() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert re.search(r"^FROM python:3\.14|ARG PYTHON_IMAGE=python:3\.14", dockerfile, re.M)
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert re.search(r"^USER app$", dockerfile, re.M)
    assert re.search(r'^CMD \["fastapi", "run", .*"0\.0\.0\.0", "--port", "8000"', dockerfile, re.M)
    assert "--index-url https://download.pytorch.org/whl/cpu" in dockerfile
    assert '"xgboost-cpu==${xgboost_version}"' in dockerfile


def test_dockerignore_keeps_secrets_data_and_artifacts_out() -> None:
    ignored = set((ROOT / ".dockerignore").read_text().split())

    assert {".env", ".env.*", ".git", "artifacts", "data", "mlruns", "backend/tests"} <= ignored


def test_compose_api_service_uses_service_hostnames_and_optional_env_file() -> None:
    api = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["api"]

    assert api["env_file"] == [{"path": ".env", "required": False}]
    assert "@postgres:5432/" in api["environment"]["DATABASE_URL"]
    assert api["environment"]["REDIS_URL"] == "redis://redis:6379/0"
    # Build-time (baked) and runtime embedding model come from the same variable.
    assert api["build"]["args"]["EMBEDDING_MODEL"] == api["environment"]["EMBEDDING_MODEL"]
    assert {name: dep["condition"] for name, dep in api["depends_on"].items()} == {
        "postgres": "service_healthy",
        "redis": "service_healthy",
    }
