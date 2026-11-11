"""Lead scoring: tiers, the scoring service, the champion provider and the routes."""

import threading
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.api.auth import get_current_user
from app.api.deps import get_lead_scoring_service
from app.core.config import Settings
from app.core.exceptions import NotFoundError, ScoringUnavailableError
from app.main import create_app
from app.ml_inference.champion import ChampionModel, ChampionModelProvider
from app.models import Lead, LeadPrediction
from app.models.enums import LeadSource, LeadStatus, LeadTier
from app.repositories.lead import LeadRepository
from app.repositories.lead_features import LeadFeatureRepository
from app.repositories.lead_prediction import LeadPredictionRepository
from app.services.lead_scoring_service import LeadScoringService, lead_tier
from ml.features.feature_spec import ACTIVE_FEATURES, spec_fingerprint
from ml.model_contract import model_signature
from ml.training.mlflow_config import TrackingUnavailableError
from tests.unit.helpers import make_user, mock_repo

T = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
LEAD_ID = uuid.uuid4()
FEATURES: dict[str, Any] = {
    "lead_source": "REFERRAL",
    "industry": "Software",
    "company_size": 120,
    "estimated_value": 25000.5,
    "lead_age_days": 14.0,
    "activity_count_7d": 1,
    "activity_count_30d": 3,
    "email_count_30d": 1,
    "meeting_count_30d": 1,
    "inbound_response_count_30d": 1,
    "active_days_30d": 2,
    "days_since_last_activity": None,
}


class FakePyfunc:
    """Stands in for an MLflow pyfunc model whose predict() serves predict_proba."""

    def __init__(self, output: Any = ((0.8, 0.2),)) -> None:
        self.output = output
        self.calls: list[Any] = []
        self.metadata = SimpleNamespace(
            flavors={"python_function": {"predict_fn": "predict_proba"}},
            signature=model_signature(),
        )

    def predict(self, wire: Any) -> Any:
        self.calls.append(wire)
        return np.asarray(self.output)


class FakeProvider:
    def __init__(self, model: Any = None, error: Exception | None = None) -> None:
        self.model = model if model is not None else FakePyfunc()
        self.error = error
        self.calls = 0

    def get(self) -> ChampionModel:
        self.calls += 1
        if self.error:
            raise self.error
        return ChampionModel("revintel-lead-conversion", "4", self.model)


def _lead() -> Lead:
    return Lead(
        id=LEAD_ID,
        company_name="Initech",
        contact_name="Peter",
        lead_source=LeadSource.REFERRAL,
        status=LeadStatus.WORKING,
        created_at=T,
        updated_at=T,
    )


def _service(
    provider: Any,
    *,
    lead: Lead | None = None,
    latest: LeadPrediction | None = None,
    timer: Callable[[], float] | None = None,
) -> tuple[LeadScoringService, MagicMock, MagicMock]:
    leads = MagicMock(spec=LeadRepository)
    leads.get.return_value = lead
    predictions = mock_repo(LeadPredictionRepository)
    predictions.latest_for_lead.return_value = latest
    features = MagicMock(spec=LeadFeatureRepository)
    features.features_at.return_value = dict(FEATURES)
    ticks = iter([100.0, 100.0125])
    service = LeadScoringService(
        MagicMock(),
        leads,
        predictions,
        features,
        provider,
        clock=lambda: T,
        timer=timer or (lambda: next(ticks)),
    )
    return service, predictions, features


# --- tiers ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("probability", "tier"),
    [
        (0.0, LeadTier.COLD),
        (0.0499999, LeadTier.COLD),
        (0.05, LeadTier.WARM),
        (0.1115, LeadTier.WARM),  # the model's evaluation threshold has no role here
        (0.1499999, LeadTier.WARM),
        (0.15, LeadTier.HOT),
        (1.0, LeadTier.HOT),
    ],
)
def test_tier_boundaries(probability: float, tier: LeadTier) -> None:
    assert lead_tier(probability) == tier


# --- scoring service --------------------------------------------------------------------------


def test_score_persists_the_exact_result() -> None:
    model = FakePyfunc(output=[[0.827, 0.173]])
    service, predictions, features = _service(FakeProvider(model), lead=_lead())

    prediction = service.score(LEAD_ID)

    predictions.add.assert_called_once_with(prediction)
    features.features_at.assert_called_once_with(LEAD_ID, T)  # one T for features...
    assert prediction.prediction_timestamp == T  # ...and the stored row
    assert prediction.conversion_probability == pytest.approx(0.173)
    assert prediction.lead_tier == LeadTier.HOT
    assert (prediction.model_name, prediction.model_version) == ("revintel-lead-conversion", "4")
    assert prediction.inference_ms == 12.5  # only the timed predict call
    assert list(prediction.features) == list(ACTIVE_FEATURES)
    assert prediction.features["days_since_last_activity"] is None
    (wire,) = model.calls
    assert list(wire.columns) == list(ACTIVE_FEATURES)


def test_missing_lead_never_touches_the_model() -> None:
    provider = FakeProvider()
    service, predictions, _ = _service(provider, lead=None)

    with pytest.raises(NotFoundError, match="not found"):
        service.score(LEAD_ID)

    assert provider.calls == 0
    predictions.add.assert_not_called()


@pytest.mark.parametrize(
    "output",
    [
        [[0.2]],  # one column
        [[0.8, 0.2], [0.7, 0.3]],  # two rows for one lead
        [[np.nan, np.nan]],
        [[1.2, -0.2]],  # outside [0, 1]
        [[0.5, 0.6]],  # rows must sum to 1
        "not-an-array",
    ],
)
def test_malformed_model_output_is_unavailable_and_logged(output: Any) -> None:
    service, predictions, _ = _service(FakeProvider(FakePyfunc(output)), lead=_lead())

    with capture_logs() as logs, pytest.raises(ScoringUnavailableError) as error:
        service.score(LEAD_ID)

    assert error.value.message == "Lead scoring is currently unavailable."
    predictions.add.assert_not_called()
    (log,) = [entry for entry in logs if entry["event"] == "lead_scoring_unavailable"]
    assert log["lead_id"] == str(LEAD_ID) and log["reason"]


def test_unavailable_model_is_logged_with_its_reason() -> None:
    provider = FakeProvider(error=ScoringUnavailableError("no champion alias"))
    service, predictions, _ = _service(provider, lead=_lead())

    with capture_logs() as logs, pytest.raises(ScoringUnavailableError):
        service.score(LEAD_ID)

    assert [entry["reason"] for entry in logs] == ["no champion alias"]
    predictions.add.assert_not_called()


def test_latest_returns_the_stored_row_without_the_model() -> None:
    stored = LeadPrediction(lead_id=LEAD_ID, lead_tier=LeadTier.COLD, conversion_probability=0.9)
    provider = FakeProvider(error=AssertionError("GET must not use the model"))
    service, _, features = _service(provider, lead=_lead(), latest=stored)

    assert service.latest(LEAD_ID) is stored
    assert provider.calls == 0
    features.features_at.assert_not_called()


def test_latest_distinguishes_missing_lead_and_missing_score() -> None:
    with pytest.raises(NotFoundError, match=f"Lead {LEAD_ID} not found"):
        _service(FakeProvider(), lead=None)[0].latest(LEAD_ID)
    with pytest.raises(NotFoundError, match="No score found"):
        _service(FakeProvider(), lead=_lead(), latest=None)[0].latest(LEAD_ID)


# --- champion provider ------------------------------------------------------------------------


class FakeClient:
    def __init__(self, version: Any = None, error: Exception | None = None) -> None:
        self.version = version or SimpleNamespace(
            version=4, tags={"feature_spec_fingerprint": spec_fingerprint()}
        )
        self.error = error
        self.aliases: list[tuple[str, str]] = []

    def get_model_version_by_alias(self, name: str, alias: str) -> Any:
        self.aliases.append((name, alias))
        if self.error:
            raise self.error
        return self.version


def _provider(
    client: FakeClient | None = None,
    model: Any = None,
    loads: list[str] | None = None,
    health: Callable[[str], None] = lambda _: None,
) -> ChampionModelProvider:
    client = client or FakeClient()

    def loader(_uri: str, name: str, version: str) -> Any:
        if loads is not None:
            loads.append(f"{name}/{version}")
        return model if model is not None else FakePyfunc()

    return ChampionModelProvider(
        "http://mlflow.test",
        "revintel-lead-conversion",
        client_factory=lambda _: client,
        loader=loader,
        health_check=health,
    )


def test_provider_resolves_the_alias_and_caches_that_exact_version() -> None:
    client, loads = FakeClient(), []
    provider = _provider(client, loads=loads)

    first, second = provider.get(), provider.get()

    assert first is second
    assert first.version == "4"
    assert client.aliases == [("revintel-lead-conversion", "champion")]  # resolved once
    assert loads == ["revintel-lead-conversion/4"]  # the resolved version, not the alias


def test_provider_loads_once_under_concurrency() -> None:
    loads: list[str] = []
    provider = _provider(loads=loads)
    threads = [threading.Thread(target=provider.get) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(loads) == 1


def _signature_without_last_feature() -> Any:
    signature = model_signature()
    signature.inputs = type(signature.inputs)(signature.inputs.inputs[:-1])
    return signature


def _server_down(uri: str) -> None:
    raise TrackingUnavailableError(f"{uri} is not reachable")


def _model(**metadata: Any) -> FakePyfunc:
    model = FakePyfunc()
    for key, value in metadata.items():
        setattr(model.metadata, key, value)
    return model


def _failing_loader(*_: object) -> Any:
    raise OSError("download failed")


UNUSABLE_CHAMPIONS: dict[str, Callable[[], ChampionModelProvider]] = {
    "server_down": lambda: _provider(health=_server_down),
    "no_champion": lambda: _provider(FakeClient(error=RuntimeError("alias not found"))),
    "fingerprint": lambda: _provider(
        FakeClient(SimpleNamespace(version=4, tags={"feature_spec_fingerprint": "other"}))
    ),
    "load_error": lambda: ChampionModelProvider(
        "http://mlflow.test",
        "revintel-lead-conversion",
        client_factory=lambda _: FakeClient(),
        loader=_failing_loader,
        health_check=lambda _: None,
    ),
    "no_signature": lambda: _provider(model=_model(signature=None)),
    "input_signature": lambda: _provider(model=_model(signature=_signature_without_last_feature())),
    "hard_class_predict": lambda: _provider(
        model=_model(flavors={"python_function": {"predict_fn": "predict"}})
    ),
}


@pytest.mark.parametrize("case", sorted(UNUSABLE_CHAMPIONS))
def test_provider_rejects_unusable_champions(case: str) -> None:
    provider = UNUSABLE_CHAMPIONS[case]()

    with pytest.raises(ScoringUnavailableError) as error:
        provider.get()

    assert error.value.message == "Lead scoring is currently unavailable."
    assert error.value.reason


def test_provider_failure_is_not_cached() -> None:
    client = FakeClient(error=RuntimeError("registry briefly unavailable"))
    provider = _provider(client)

    with pytest.raises(ScoringUnavailableError):
        provider.get()
    client.error = None

    assert provider.get().version == "4"


# --- routes -----------------------------------------------------------------------------------


@pytest.fixture
def scoring_client(settings: Settings) -> Iterator[tuple[TestClient, MagicMock]]:
    service = MagicMock(spec=LeadScoringService)
    app = create_app(settings)
    app.dependency_overrides[get_lead_scoring_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: make_user()
    with TestClient(app) as client:
        yield client, service


def _stored(**fields: Any) -> LeadPrediction:
    values = {
        "lead_id": LEAD_ID,
        "conversion_probability": 0.173,
        "lead_tier": LeadTier.HOT,
        "model_name": "revintel-lead-conversion",
        "model_version": "4",
        "prediction_timestamp": T,
        "inference_ms": 14.8,
    }
    return LeadPrediction(**{**values, **fields})


def test_post_returns_201_with_the_documented_shape(
    scoring_client: tuple[TestClient, MagicMock],
) -> None:
    client, service = scoring_client
    service.score.return_value = _stored()

    response = client.post(f"/api/v1/leads/{LEAD_ID}/scores")

    assert response.status_code == 201
    assert response.json() == {
        "lead_id": str(LEAD_ID),
        "conversion_probability": 0.173,
        "lead_tier": "HOT",
        "model_name": "revintel-lead-conversion",
        "model_version": "4",
        "prediction_timestamp": "2026-11-01T12:00:00Z",
        "inference_ms": 14.8,
    }


def test_unavailable_scoring_returns_a_generic_503(
    scoring_client: tuple[TestClient, MagicMock],
) -> None:
    client, service = scoring_client
    service.score.side_effect = ScoringUnavailableError(
        "cannot load revintel-lead-conversion version 4: http://mlflow.internal/... boom"
    )

    response = client.post(f"/api/v1/leads/{LEAD_ID}/scores")

    assert response.status_code == 503
    assert response.json() == {"detail": "Lead scoring is currently unavailable."}


def test_get_latest_and_not_found(scoring_client: tuple[TestClient, MagicMock]) -> None:
    client, service = scoring_client
    service.latest.return_value = _stored(lead_tier=LeadTier.COLD)
    assert client.get(f"/api/v1/leads/{LEAD_ID}/scores/latest").json()["lead_tier"] == "COLD"

    service.latest.side_effect = NotFoundError(f"No score found for lead {LEAD_ID}")
    response = client.get(f"/api/v1/leads/{LEAD_ID}/scores/latest")
    assert response.status_code == 404
    assert response.json() == {"detail": f"No score found for lead {LEAD_ID}"}
