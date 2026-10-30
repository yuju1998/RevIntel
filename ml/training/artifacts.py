"""Save and load a trained lead conversion model with its metadata.

`model.joblib` holds the full sklearn Pipeline (preprocessing included) and is a pickle:
load only artifacts this project produced. `metadata.json` records the dataset hash,
feature spec, threshold and split; loading fails if the spec no longer matches.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.features.feature_spec import ACTIVE_FEATURES
from ml.training.dataset import FeatureSchemaError, check_spec_metadata

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class LeadConversionModel:
    pipeline: Pipeline
    metadata: dict[str, Any]

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"]["value"])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Conversion probability for each row of active features."""
        missing = [name for name in ACTIVE_FEATURES if name not in features.columns]
        if missing:
            raise FeatureSchemaError(f"missing active features: {missing}")
        return self.pipeline.predict_proba(features.loc[:, list(ACTIVE_FEATURES)])[:, 1]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(features) >= self.threshold


def save_model(model: LeadConversionModel, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, directory / MODEL_FILE)
    (directory / METADATA_FILE).write_text(json.dumps(model.metadata, indent=2) + "\n")


def load_model(directory: Path) -> LeadConversionModel:
    metadata = json.loads((directory / METADATA_FILE).read_text())
    check_spec_metadata(metadata, f"Model {directory}")
    return LeadConversionModel(pipeline=joblib.load(directory / MODEL_FILE), metadata=metadata)
