import pytest

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
