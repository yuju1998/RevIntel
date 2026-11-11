"""Lead scoring endpoints against revintel_test: persistence, history, read-back, errors.

MLflow is replaced by a fake provider except in the last test, which uses a real pipeline
logged to a temporary SQLite registry (never the real MLflow registry).
"""

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from app.api.deps import get_champion_models
from app.core.config import Settings
from app.core.exceptions import ScoringUnavailableError
from app.main import create_app
from app.ml_inference.champion import ChampionModel, ChampionModelProvider
from app.models import Activity, LeadPrediction
from app.models.enums import ActivityDirection, ActivityType
from ml.features.feature_spec import ACTIVE_FEATURES
from ml.features.online_features import fetch_online_features
from ml.model_contract import wire_record
from tests.integration.conftest import bearer_headers
from tests.integration.factories import create_lead

pytestmark = pytest.mark.integration

MODEL = "revintel-lead-conversion"


class FakeModel:
    def __init__(self) -> None:
        self.outputs = [[[0.8, 0.2]]]
        self.inputs: list[pd.DataFrame] = []

    def predict(self, wire: pd.DataFrame) -> np.ndarray:
        self.inputs.append(wire)
        return np.asarray(self.outputs[min(len(self.inputs), len(self.outputs)) - 1])


class FakeProvider:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.error: Exception | None = None
        self.calls = 0

    def get(self) -> ChampionModel:
        self.calls += 1
        if self.error:
            raise self.error
        return ChampionModel(MODEL, "7", self.model)


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


def _client(settings: Settings, provider: Any, headers: dict[str, str]) -> TestClient:
    app = create_app(settings)
    app.dependency_overrides[get_champion_models] = lambda: provider
    return TestClient(app, headers=headers)


@pytest.fixture
def scoring_client(
    integration_settings: Settings, migrated_database: str, auth_user: Any, provider: FakeProvider
) -> Iterator[TestClient]:
    headers = bearer_headers(auth_user, integration_settings)
    with _client(integration_settings, provider, headers) as client:
        yield client


def _url(lead_id: Any, latest: bool = False) -> str:
    return f"/api/v1/leads/{lead_id}/scores" + ("/latest" if latest else "")


def _rows(db_session: Session, lead_id: Any) -> list[LeadPrediction]:
    db_session.expire_all()
    stmt = (
        select(LeadPrediction)
        .where(LeadPrediction.lead_id == lead_id)
        .order_by(LeadPrediction.prediction_timestamp)
    )
    return list(db_session.scalars(stmt))


def test_post_scores_with_point_in_time_features_and_persists(
    scoring_client: TestClient, db_session: Session, provider: FakeProvider
) -> None:
    lead = create_lead(db_session, industry="Software", company_size=40)
    db_session.add(
        Activity(
            lead_id=lead.id,
            activity_type=ActivityType.MEETING,
            direction=ActivityDirection.INBOUND,
            occurred_at=datetime.now(UTC) - timedelta(days=2),
        )
    )
    db_session.commit()

    response = scoring_client.post(_url(lead.id))

    assert response.status_code == 201, response.json()
    body = response.json()
    assert body["conversion_probability"] == pytest.approx(0.2)
    assert body["lead_tier"] == "HOT"
    assert (body["model_name"], body["model_version"]) == (MODEL, "7")
    assert body["inference_ms"] >= 0
    (row,) = _rows(db_session, lead.id)
    assert row.model_version == "7"
    assert row.prediction_timestamp == datetime.fromisoformat(body["prediction_timestamp"])
    assert set(row.features) == set(ACTIVE_FEATURES)  # JSONB does not keep key order
    assert row.features["meeting_count_30d"] == 1
    assert row.features["inbound_response_count_30d"] == 1
    assert row.features["industry"] == "Software"
    (wire,) = provider.model.inputs
    assert row.features == pytest.approx(wire_record(wire))  # exactly what the model saw


def test_repeated_scoring_keeps_history_and_latest_is_newest(
    scoring_client: TestClient, db_session: Session, provider: FakeProvider
) -> None:
    lead = create_lead(db_session)
    provider.model.outputs = [[[0.8, 0.2]], [[0.97, 0.03]]]

    first = scoring_client.post(_url(lead.id)).json()
    second = scoring_client.post(_url(lead.id)).json()

    assert [r.lead_tier for r in _rows(db_session, lead.id)] == ["HOT", "COLD"]
    latest = scoring_client.get(_url(lead.id, latest=True))
    assert latest.status_code == 200
    assert latest.json() == second
    assert first != second


def test_latest_returns_the_stored_values_without_the_model(
    scoring_client: TestClient, db_session: Session, provider: FakeProvider
) -> None:
    lead = create_lead(db_session)
    scoring_client.post(_url(lead.id))
    # Change the stored row: the read must return it as stored, not recompute anything.
    db_session.execute(
        text(
            "UPDATE lead_predictions SET lead_tier = 'COLD', conversion_probability = 0.9 "
            "WHERE lead_id = :lead_id"
        ),
        {"lead_id": lead.id},
    )
    db_session.commit()
    provider.error = AssertionError("GET latest must not use the model")
    calls = provider.calls

    body = scoring_client.get(_url(lead.id, latest=True)).json()

    assert (body["lead_tier"], body["conversion_probability"]) == ("COLD", 0.9)
    assert provider.calls == calls


def test_missing_lead_and_missing_score(
    scoring_client: TestClient, db_session: Session, provider: FakeProvider
) -> None:
    missing = uuid.uuid4()
    assert scoring_client.post(_url(missing)).status_code == 404
    assert scoring_client.get(_url(missing, latest=True)).json() == {
        "detail": f"Lead {missing} not found"
    }
    assert provider.calls == 0

    lead = create_lead(db_session)
    response = scoring_client.get(_url(lead.id, latest=True))
    assert response.status_code == 404
    assert response.json() == {"detail": f"No score found for lead {lead.id}"}


def test_unavailable_scoring_is_a_generic_503_and_stores_nothing(
    scoring_client: TestClient, db_session: Session, provider: FakeProvider
) -> None:
    lead = create_lead(db_session)
    provider.error = ScoringUnavailableError("MLflow tracking server at http://... is down")

    response = scoring_client.post(_url(lead.id))

    assert response.status_code == 503
    assert response.json() == {"detail": "Lead scoring is currently unavailable."}
    assert _rows(db_session, lead.id) == []


def test_post_requires_a_token_and_get_is_public(
    integration_settings: Settings,
    migrated_database: str,
    db_session: Session,
    provider: FakeProvider,
) -> None:
    lead = create_lead(db_session)
    with _client(integration_settings, provider, headers={}) as anonymous:
        assert anonymous.post(_url(lead.id)).status_code == 401
        assert provider.calls == 0
        assert anonymous.get(_url(lead.id, latest=True)).status_code == 404  # no score yet


def test_deleting_a_lead_removes_its_scores(
    scoring_client: TestClient, db_session: Session
) -> None:
    lead = create_lead(db_session)
    scoring_client.post(_url(lead.id))

    db_session.execute(text("DELETE FROM leads WHERE id = :id"), {"id": lead.id})
    db_session.commit()

    assert _rows(db_session, lead.id) == []


# --- real model through a temporary MLflow registry -------------------------------------------


@pytest.fixture(scope="module")
def temporary_registry(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, Any]]:
    """A real pipeline registered as `champion` in a throwaway SQLite registry."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from mlflow import MlflowClient

    from ml.features.feature_spec import LABEL, compute_features, spec_fingerprint
    from ml.features.snapshots import SnapshotPolicy, build_snapshots
    from ml.features.sources import activities_frame, leads_frame
    from ml.model_contract import PYFUNC_PREDICT_FN, model_signature, to_wire_frame
    from ml.synthetic.config import GenerationConfig
    from ml.synthetic.generator import generate
    from ml.training.pipelines import build_logistic_regression
    from ml.training.tracking import SKOPS_TRUSTED_TYPES

    dataset = generate(
        GenerationConfig(seed=5, accounts=10, contacts=30, leads=400, activities=2000, reps=3)
    )
    leads = leads_frame(dataset.leads)
    snapshots = build_snapshots(
        leads, SnapshotPolicy(dataset.config.as_of, dataset.config.max_lead_age_days)
    )
    features = compute_features(snapshots, activities_frame(dataset.activities))
    pipeline = build_logistic_regression().fit(
        features.loc[:, list(ACTIVE_FEATURES)], snapshots[LABEL].to_numpy()
    )

    directory = tmp_path_factory.mktemp("registry")
    uri = f"sqlite:///{directory / 'mlflow.db'}"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.create_experiment("scoring", artifact_location=directory.as_uri())
    with mlflow.start_run(experiment_id=experiment):
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=model_signature(),
            input_example=to_wire_frame(features.head(2)),
            pyfunc_predict_fn=PYFUNC_PREDICT_FN,
            skops_trusted_types=SKOPS_TRUSTED_TYPES["logistic_regression"],
            registered_model_name=MODEL,
        )
    mlflow.set_tracking_uri(previous)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    version = str(info.registered_model_version)
    client.set_model_version_tag(MODEL, version, "feature_spec_fingerprint", spec_fingerprint())
    client.set_registered_model_alias(MODEL, "champion", version)  # temporary registry only
    yield uri, pipeline


def test_real_champion_scores_like_the_offline_pipeline(
    integration_settings: Settings,
    migrated_database: str,
    auth_user: Any,
    db_session: Session,
    db_engine: Engine,
    temporary_registry: tuple[str, Any],
) -> None:
    from ml.model_contract import to_wire_frame

    uri, pipeline = temporary_registry
    lead = create_lead(db_session, industry="Software", company_size=120)
    db_session.add(
        Activity(
            lead_id=lead.id,
            activity_type=ActivityType.DEMO,
            direction=ActivityDirection.INBOUND,
            occurred_at=datetime.now(UTC) - timedelta(days=3),
        )
    )
    db_session.commit()
    provider = ChampionModelProvider(uri, MODEL)
    headers = bearer_headers(auth_user, integration_settings)

    with _client(integration_settings, provider, headers) as client:
        body = client.post(_url(lead.id)).json()
        again = client.post(_url(lead.id)).json()

    assert body["model_version"] == again["model_version"] == "1"
    at = datetime.fromisoformat(body["prediction_timestamp"])
    with db_engine.connect() as connection:
        online = fetch_online_features(connection, lead.id, at)
    expected = pipeline.predict_proba(to_wire_frame(pd.DataFrame([online])))[0, 1]
    assert body["conversion_probability"] == pytest.approx(expected)
    first, second = _rows(db_session, lead.id)
    assert first.features == pytest.approx(wire_record(to_wire_frame(pd.DataFrame([online]))))
    assert second.prediction_timestamp > first.prediction_timestamp
