from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from ml.features.build_features import features_sha256
from ml.features.feature_spec import LABEL, compute_features, spec_metadata
from ml.features.snapshots import SnapshotPolicy, build_snapshots
from ml.features.sources import activities_frame, leads_frame
from ml.synthetic.config import GenerationConfig
from ml.synthetic.generator import SyntheticDataset, generate
from ml.training.dataset import FeatureTable
from ml.training.train import TrainingResult, train_candidates

SMALL = GenerationConfig(
    seed=42, accounts=60, contacts=240, leads=600, activities=3000, reps=5, admins=1
)


@pytest.fixture(scope="session")
def small_dataset() -> SyntheticDataset:
    return generate(SMALL)


@pytest.fixture(scope="session")
def realism_dataset() -> SyntheticDataset:
    # Enough leads for stable rates; accounts are irrelevant to lead conversion.
    return generate(
        GenerationConfig(
            seed=7, accounts=50, contacts=150, leads=5000, activities=20000, reps=10, admins=1
        )
    )


@dataclass(frozen=True)
class FeatureInputs:
    leads: pd.DataFrame
    activities: pd.DataFrame
    policy: SnapshotPolicy
    snapshots: pd.DataFrame
    features: pd.DataFrame


@pytest.fixture(scope="session")
def feature_inputs(small_dataset: SyntheticDataset) -> FeatureInputs:
    leads = leads_frame(small_dataset.leads)
    activities = activities_frame(small_dataset.activities)
    policy = SnapshotPolicy(small_dataset.config.as_of, small_dataset.config.max_lead_age_days)
    snapshots = build_snapshots(leads, policy).frame
    return FeatureInputs(
        leads, activities, policy, snapshots, compute_features(snapshots, activities)
    )


def feature_table_metadata(table: pd.DataFrame) -> dict:
    return {
        **spec_metadata(),
        "features_sha256": features_sha256(table),
        "source": {"dataset_sha256": "0" * 64, "generator_version": "test"},
    }


@pytest.fixture(scope="session")
def training_table(feature_inputs: FeatureInputs) -> FeatureTable:
    frame = feature_inputs.features.assign(**{LABEL: feature_inputs.snapshots[LABEL].to_numpy()})
    return FeatureTable(Path("in-memory"), frame, feature_table_metadata(frame))


# Small XGBoost search so the suite stays fast; production settings are in pipelines.py.
FAST_XGBOOST = {
    "xgboost_depths": (2, 3),
    "xgboost_max_estimators": 60,
    "xgboost_early_stopping_rounds": 10,
}


@pytest.fixture(scope="session")
def trained(training_table: FeatureTable) -> TrainingResult:
    return train_candidates(training_table, **FAST_XGBOOST)
