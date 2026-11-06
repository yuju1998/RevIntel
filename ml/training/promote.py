"""Manually promote the current `challenger` model version to `champion`.

    python -m ml.training.promote --version <N>

Refuses unless version N is the current challenger. Nothing is retrained or re-evaluated,
and old versions stay in the registry. `validation_status=approved` stays on a version
after it is replaced: the `champion` alias alone says which version is current. See
docs/MLFLOW.md.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ml.training.register import (
    CHALLENGER,
    CHAMPION,
    AliasWorkflowError,
    build_parser,
    describe,
    registry_state,
    run_alias_command,
    target,
)

if TYPE_CHECKING:
    from mlflow import MlflowClient

VALIDATION_STATUS_TAG = "validation_status"
APPROVED = "approved"


def promote_challenger(client: MlflowClient, name: str, version: str) -> list[str]:
    aliases, model_version = registry_state(client, name, version)
    challenger, champion = aliases.get(CHALLENGER), aliases.get(CHAMPION)
    if challenger != version:
        if champion == version:
            raise AliasWorkflowError(f"Version {version} is already the {CHAMPION}.")
        raise AliasWorkflowError(
            f"Refusing to promote version {version}: the current {CHALLENGER} is "
            f"{target(challenger)}. Run `python -m ml.training.register --version {version}` "
            "and evaluate it first."
        )
    # MLflow has no transaction across these calls. The challenger alias goes last, so a
    # failure partway leaves a state this same command can finish when rerun.
    client.set_model_version_tag(name, version, VALIDATION_STATUS_TAG, APPROVED)
    client.set_registered_model_alias(name, CHAMPION, version)
    client.delete_registered_model_alias(name, CHALLENGER)

    lines = [
        f"{name}:",
        f"  {CHAMPION}: {target(champion)} -> version {version}",
        f"  {CHALLENGER}: version {version} -> none",
        *describe(model_version),
        f"version {version}: {VALIDATION_STATUS_TAG}={APPROVED}",
    ]
    if champion not in (None, version):
        lines.append(f"Version {champion} is kept in the registry.")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser("Promote the current challenger model version to champion.")
    return run_alias_command(parser.parse_args(argv), promote_challenger)


if __name__ == "__main__":
    raise SystemExit(main())
