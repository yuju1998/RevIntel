"""Online lead scoring with the champion model, and stored score lookups."""

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd
import structlog
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError, ScoringUnavailableError
from app.ml_inference.champion import ChampionModelProvider
from app.models.enums import LeadTier
from app.models.lead_prediction import LeadPrediction
from app.repositories.lead import LeadRepository
from app.repositories.lead_features import LeadFeatureRepository
from app.repositories.lead_prediction import LeadPredictionRepository
from app.services.transaction import write_transaction
from ml.model_contract import (
    ModelContractError,
    positive_class_probabilities,
    to_wire_frame,
    wire_record,
)

logger = structlog.get_logger("app.scoring")

# Product policy, independent of the model's evaluation threshold. Changing these is a
# code change, so every stored tier stays explainable.
HOT_THRESHOLD = 0.15
WARM_THRESHOLD = 0.05


def lead_tier(probability: float) -> LeadTier:
    if probability >= HOT_THRESHOLD:
        return LeadTier.HOT
    if probability >= WARM_THRESHOLD:
        return LeadTier.WARM
    return LeadTier.COLD


def _lead_not_found(lead_id: uuid.UUID) -> NotFoundError:
    return NotFoundError(f"Lead {lead_id} not found")


class LeadScoringService:
    def __init__(
        self,
        session: Session,
        leads: LeadRepository,
        predictions: LeadPredictionRepository,
        features: LeadFeatureRepository,
        models: ChampionModelProvider,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._session = session
        self._leads = leads
        self._predictions = predictions
        self._features = features
        self._models = models
        self._clock = clock
        self._timer = timer

    def score(self, lead_id: uuid.UUID) -> LeadPrediction:
        """Score the lead now with the champion model and store the result."""
        if self._leads.get(lead_id) is None:
            raise _lead_not_found(lead_id)
        at = self._clock()  # the single prediction timestamp T
        try:
            champion = self._models.get()
            features = self._features.features_at(lead_id, at)
            if features is None:
                raise _lead_not_found(lead_id)
            wire = to_wire_frame(pd.DataFrame([features]))
            started = self._timer()
            output = champion.predict_proba(wire)
            inference_ms = round((self._timer() - started) * 1000, 3)
            probability = float(positive_class_probabilities(output, rows=1)[0])
        except ModelContractError as exc:
            self._log_unavailable(lead_id, str(exc))
            raise ScoringUnavailableError(str(exc)) from exc
        except ScoringUnavailableError as exc:
            self._log_unavailable(lead_id, exc.reason)
            raise

        prediction = LeadPrediction(
            lead_id=lead_id,
            prediction_timestamp=at,
            conversion_probability=probability,
            lead_tier=lead_tier(probability),
            model_name=champion.name,
            model_version=champion.version,
            features=wire_record(wire),
            inference_ms=inference_ms,
        )
        constraint_errors = {"fk_lead_predictions_lead_id_leads": lambda: _lead_not_found(lead_id)}
        with write_transaction(self._session, constraint_errors):
            self._predictions.add(prediction)
        return prediction

    def latest(self, lead_id: uuid.UUID) -> LeadPrediction:
        """The most recent stored score, as stored (no MLflow, no recomputation)."""
        if self._leads.get(lead_id) is None:
            raise _lead_not_found(lead_id)
        prediction = self._predictions.latest_for_lead(lead_id)
        if prediction is None:
            raise NotFoundError(f"No score found for lead {lead_id}")
        return prediction

    @staticmethod
    def _log_unavailable(lead_id: uuid.UUID, reason: str) -> None:
        logger.warning("lead_scoring_unavailable", lead_id=str(lead_id), reason=reason)
