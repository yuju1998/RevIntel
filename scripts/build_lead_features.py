"""Build the point-in-time lead conversion feature table (no training, no database writes).

Example:

    SYNTHETIC_DATABASE_URL=postgresql+psycopg://revintel:revintel@localhost:5432/revintel_ml \\
    python scripts/build_lead_features.py

Reads the synthetic CRM dataset read-only (source from --database-url or
SYNTHETIC_DATABASE_URL, never DATABASE_URL) and writes data/processed/lead_features.parquet.
See docs/DATABASE.md ("Lead conversion features").
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    # Make `app` (backend) and `ml` importable when run as a script.
    for path in (str(ROOT / "backend"), str(ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from ml.features.build_features import main as build_main

    return build_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
