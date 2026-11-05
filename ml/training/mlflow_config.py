"""MLflow connection defaults and the tracking server reachability check.

Standard library only: the alias commands (`python -m ml.training.register` / `promote`)
import this without the training code, MLflow or the backend package.
"""

import os
import urllib.error
import urllib.request

DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
DEFAULT_EXPERIMENT = "lead-conversion"
DEFAULT_REGISTERED_MODEL = "revintel-lead-conversion"


class TrackingUnavailableError(RuntimeError):
    """The MLflow tracking server cannot be reached."""


def default_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def default_registered_model() -> str:
    return os.environ.get("MLFLOW_MODEL_NAME", DEFAULT_REGISTERED_MODEL)


def require_tracking_server(tracking_uri: str, timeout: float = 3.0) -> None:
    """Fail fast when an HTTP tracking server is not reachable (non-HTTP stores are local)."""
    if not tracking_uri.startswith(("http://", "https://")):
        return
    try:
        with urllib.request.urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=timeout):
            pass
    except (urllib.error.URLError, OSError) as exc:
        raise TrackingUnavailableError(
            f"MLflow tracking server at {tracking_uri} is not reachable ({exc}). "
            "Start it with scripts/start_mlflow.sh."
        ) from exc
