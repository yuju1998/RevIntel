"""Train and evaluate the lead conversion models (Logistic Regression, XGBoost).

Example:

    python scripts/train_lead_models.py [--features PATH] [--output-dir PATH]

Reads data/processed/lead_features.parquet (no database access) and writes each model's
Pipeline and metadata plus an evaluation report to artifacts/models/lead_conversion/.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    # Make `app` (backend) and `ml` importable when run as a script.
    for path in (str(ROOT / "backend"), str(ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from ml.training.train import main as train_main

    return train_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
