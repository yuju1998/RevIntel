"""Point-in-time lead features for online scoring.

Delegates to the shared online feature path (`ml.features.online_features`): the same
extraction queries and `compute_features()` that build the offline training table, run on
the request session's connection.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from ml.features.online_features import fetch_online_features


class LeadFeatureRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def features_at(self, lead_id: uuid.UUID, at: datetime) -> dict[str, Any] | None:
        """Active features of the lead as known at `at`, or None if the lead is gone."""
        try:
            return fetch_online_features(self._session.connection(), lead_id, at)
        except LookupError:
            return None
