"""The `champion` lead conversion model, resolved and loaded once per process.

On first use the provider resolves the registered model's `champion` alias to a concrete
version, loads exactly that version as an MLflow pyfunc model (whose inference function
is `predict_proba`) and checks it against the model contract. A successful load is cached
for the life of the process; failures are not cached, so a later request retries. After a
new champion is promoted, restart the API. The registry is only read, never changed.
"""

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.core.exceptions import ScoringUnavailableError
from ml.features.feature_spec import spec_fingerprint
from ml.model_contract import ModelContractError, check_inference_function, check_signature
from ml.training.mlflow_config import TrackingUnavailableError, require_tracking_server

CHAMPION_ALIAS = "champion"
FINGERPRINT_TAG = "feature_spec_fingerprint"


@dataclass(frozen=True)
class ChampionModel:
    name: str
    version: str
    pyfunc_model: Any  # mlflow.pyfunc.PyFuncModel serving predict_proba

    def predict_proba(self, wire: pd.DataFrame) -> Any:
        """Class probabilities from the pyfunc model (never a hard-class predict())."""
        try:
            return self.pyfunc_model.predict(wire)
        except Exception as exc:  # any failure inside MLflow/the model means no score
            raise ScoringUnavailableError(f"model prediction failed: {exc}") from exc


def _mlflow_client(tracking_uri: str) -> Any:
    from mlflow import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def _load_pyfunc(tracking_uri: str, name: str, version: str) -> Any:
    """Download exactly this version through `tracking_uri` and load it as a pyfunc model.

    Uses explicit URIs only: MLflow's process-global tracking/registry URIs are untouched.
    """
    import mlflow

    source = _mlflow_client(tracking_uri).get_model_version_download_uri(name, version)
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=source, tracking_uri=tracking_uri)
    return mlflow.pyfunc.load_model(local_path)


class ChampionModelProvider:
    def __init__(
        self,
        tracking_uri: str,
        model_name: str,
        *,
        client_factory: Callable[[str], Any] = _mlflow_client,
        loader: Callable[[str, str, str], Any] = _load_pyfunc,
        health_check: Callable[[str], None] = require_tracking_server,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._model_name = model_name
        self._client_factory = client_factory
        self._loader = loader
        self._health_check = health_check
        self._lock = threading.Lock()
        self._champion: ChampionModel | None = None

    def get(self) -> ChampionModel:
        """The cached champion, loading it on first use (thread-safe)."""
        champion = self._champion
        if champion is None:
            with self._lock:
                if self._champion is None:
                    self._champion = self._load()
                champion = self._champion
        return champion

    def _load(self) -> ChampionModel:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        try:
            self._health_check(self._tracking_uri)
        except TrackingUnavailableError as exc:
            raise ScoringUnavailableError(str(exc)) from exc
        name = self._model_name
        try:
            version = self._client_factory(self._tracking_uri).get_model_version_by_alias(
                name, CHAMPION_ALIAS
            )
        except Exception as exc:  # MLflow raises for a missing alias, model or server error
            raise ScoringUnavailableError(f"cannot resolve {name}@{CHAMPION_ALIAS}: {exc}") from exc

        number = str(version.version)
        fingerprint = (version.tags or {}).get(FINGERPRINT_TAG)
        if fingerprint != spec_fingerprint():
            raise ScoringUnavailableError(
                f"{name} version {number} was trained with feature spec {fingerprint!r}, "
                f"not the current {spec_fingerprint()!r}"
            )
        try:
            # Load the resolved version, never the alias: the cached model and the stored
            # version can then never disagree.
            model = self._loader(self._tracking_uri, name, number)
        except Exception as exc:  # download, deserialization or dependency failures
            raise ScoringUnavailableError(f"cannot load {name} version {number}: {exc}") from exc
        try:
            check_inference_function(model.metadata.flavors)
            check_signature(model.metadata.signature)
        except ModelContractError as exc:
            raise ScoringUnavailableError(f"{name} version {number}: {exc}") from exc
        return ChampionModel(name=name, version=number, pyfunc_model=model)
