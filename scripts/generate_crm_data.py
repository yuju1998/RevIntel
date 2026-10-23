"""Generate deterministic synthetic CRM data (raw rows only; no ML features or labels).

Example:

    SYNTHETIC_DATABASE_URL=postgresql+psycopg://revintel:revintel@localhost:5432/revintel_ml \\
    python scripts/generate_crm_data.py --seed 42 --accounts 5000 --contacts 20000 \\
        --leads 30000 --activities 100000

The target comes only from --database-url or SYNTHETIC_DATABASE_URL (never DATABASE_URL)
and its name must end in `_test` or `_ml`. Use --dry-run to generate and summarize
without touching any database. See docs/DATABASE.md ("Synthetic CRM data").
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "raw" / "synthetic_crm_manifest.json"


def _parse_as_of(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("--as-of must include a timezone, e.g. ...T00:00:00Z")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accounts", type=int, default=5000)
    parser.add_argument("--contacts", type=int, default=20000)
    parser.add_argument("--leads", type=int, default=30000)
    parser.add_argument("--activities", type=int, default=100000)
    parser.add_argument("--reps", type=int, default=25, help="synthetic SALES_REP users")
    parser.add_argument("--admins", type=int, default=2, help="synthetic ADMIN users")
    parser.add_argument("--as-of", type=_parse_as_of, default="2026-10-01T00:00:00Z")
    parser.add_argument("--lead-history-days", type=int, default=540)
    parser.add_argument("--max-lead-age-days", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("SYNTHETIC_DATABASE_URL"),
        help="target database (default: $SYNTHETIC_DATABASE_URL; DATABASE_URL is never used)",
    )
    parser.add_argument("--reset", action="store_true", help="replace existing CRM rows")
    parser.add_argument("--dry-run", action="store_true", help="generate without writing")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Make `app` (backend) and `ml` importable when run as a script.
    for path in (str(ROOT / "backend"), str(ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from ml.synthetic.config import GenerationConfig
    from ml.synthetic.generator import ActivityBudgetExceeded, generate
    from ml.synthetic.summary import (
        build_manifest,
        format_summary,
        sanity_counters,
        summary_stats,
        write_manifest,
    )
    from ml.synthetic.writer import UnsafeTargetError, require_safe_database_url, write_dataset

    if args.batch_size < 1:
        print("Error: --batch-size must be >= 1", file=sys.stderr)
        return 2
    if not args.dry_run:
        if not args.database_url:
            print(
                "Error: no target database. Pass --database-url or set SYNTHETIC_DATABASE_URL "
                "(DATABASE_URL is deliberately ignored), or use --dry-run.",
                file=sys.stderr,
            )
            return 2
        try:
            require_safe_database_url(args.database_url)
        except UnsafeTargetError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    config = GenerationConfig(
        seed=args.seed,
        accounts=args.accounts,
        contacts=args.contacts,
        leads=args.leads,
        activities=args.activities,
        reps=args.reps,
        admins=args.admins,
        as_of=args.as_of,
        lead_history_days=args.lead_history_days,
        max_lead_age_days=args.max_lead_age_days,
    )
    started = time.perf_counter()
    print(f"Generating synthetic CRM data (seed {config.seed}, as_of {config.as_of.isoformat()})")
    try:
        dataset = generate(config, progress=print)
    except (ValueError, ActivityBudgetExceeded) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    stats, sanity = summary_stats(dataset), sanity_counters(dataset)
    if any(sanity.values()):
        print(format_summary(stats, sanity))
        print("Error: sanity counters are not all zero; nothing was written.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("Dry run: no database writes.")
    else:
        write_started = time.perf_counter()
        try:
            write_dataset(
                dataset,
                args.database_url,
                reset=args.reset,
                batch_size=args.batch_size,
                progress=print,
            )
        except UnsafeTargetError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        print(f"  database write took {time.perf_counter() - write_started:.1f}s")

    manifest = build_manifest(dataset, stats, sanity)
    write_manifest(manifest, args.manifest)
    print(format_summary(stats, sanity))
    print(f"Dataset SHA-256: {manifest['dataset_sha256']}")
    print(f"Manifest: {args.manifest}")
    print(f"Total time: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
