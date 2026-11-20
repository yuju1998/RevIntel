"""Databases the ML tooling may target (never the development `revintel` database)."""

# Names must end with one of these: test databases and the synthetic ML dataset.
SAFE_DATABASE_SUFFIXES = ("_test", "_ml")
