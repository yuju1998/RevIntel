from dataclasses import dataclass

import pandas as pd
import pytest

from ml.features.feature_spec import compute_features
from ml.features.snapshots import SnapshotPolicy, build_snapshots
from ml.features.sources import activities_frame, leads_frame
from ml.synthetic.config import GenerationConfig
from ml.synthetic.generator import SyntheticDataset, generate

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
