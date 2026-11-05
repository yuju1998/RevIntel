"""Designate an already-registered model version as the current `challenger`.

    python -m ml.training.register --version <N>

This never creates a model version: tracked training (`ml.training.tracking`) registers
candidates. It only moves the `challenger` alias, leaving every other version and all tags
as they are. See docs/MLFLOW.md.
"""

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ml.training.mlflow_config import (
    TrackingUnavailableError,
    default_registered_model,
    default_tracking_uri,
    require_tracking_server,
)

if TYPE_CHECKING:
    from mlflow import MlflowClient
    from mlflow.entities.model_registry import ModelVersion

CHALLENGER = "challenger"
CHAMPION = "champion"
BASELINE_CHECK_TAG = "validation_baseline_check"


class AliasWorkflowError(ValueError):
    """The requested alias change is invalid or refused; nothing was changed."""


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--version", type=_positive_int, required=True, help="model version")
    parser.add_argument("--registered-model", default=default_registered_model())
    parser.add_argument("--tracking-uri", default=default_tracking_uri())
    return parser


def registry_state(
    client: MlflowClient, name: str, version: str
) -> tuple[dict[str, str], ModelVersion]:
    """Current alias -> version map and the requested version, which must be READY."""
    from mlflow.exceptions import MlflowException

    try:
        aliases = client.get_registered_model(name).aliases
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            raise AliasWorkflowError(f"Registered model {name!r} does not exist.") from exc
        raise
    try:
        model_version = client.get_model_version(name, version)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            raise AliasWorkflowError(f"Version {version} of {name!r} does not exist.") from exc
        raise
    if model_version.status != "READY":
        raise AliasWorkflowError(
            f"Version {version} of {name!r} is not ready (status {model_version.status})."
        )
    return {alias: str(target) for alias, target in aliases.items()}, model_version


def target(version: str | None) -> str:
    return "none" if version is None else f"version {version}"


def describe(model_version: ModelVersion) -> list[str]:
    """The version being changed; a failed baseline check warns but never blocks."""
    tags = model_version.tags
    check = tags.get(BASELINE_CHECK_TAG, "unset")
    lines = [
        f"version {model_version.version} ({tags.get('algorithm', 'unknown algorithm')}): "
        f"{BASELINE_CHECK_TAG}={check}"
    ]
    if check == "failed":
        lines.append("Warning: this version did not beat the naive baseline on validation.")
    return lines


def designate_challenger(client: MlflowClient, name: str, version: str) -> list[str]:
    aliases, model_version = registry_state(client, name, version)
    if aliases.get(CHAMPION) == version:
        raise AliasWorkflowError(
            f"Refusing to mark version {version} as {CHALLENGER}: it is the current {CHAMPION}."
        )
    previous = aliases.get(CHALLENGER)
    if previous == version:
        return [
            f"{CHALLENGER}: already version {version}; nothing changed.",
            *describe(model_version),
        ]
    client.set_registered_model_alias(name, CHALLENGER, version)
    return [
        f"{name}:",
        f"  {CHALLENGER}: {target(previous)} -> version {version}",
        f"  {CHAMPION}: {target(aliases.get(CHAMPION))} (unchanged)",
        *describe(model_version),
    ]


def run_alias_command(
    args: argparse.Namespace, action: Callable[[MlflowClient, str, str], list[str]]
) -> int:
    """Shared CLI flow: 2 for a refusal or an unreachable server, 1 for an MLflow failure."""
    try:
        require_tracking_server(args.tracking_uri)
    except TrackingUnavailableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient(tracking_uri=args.tracking_uri, registry_uri=args.tracking_uri)
    try:
        lines = action(client, args.registered_model, str(args.version))
    except AliasWorkflowError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (MlflowException, OSError) as exc:
        print(f"Error: MLflow request failed ({exc}).", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser("Mark an already-registered model version as the current challenger.")
    return run_alias_command(parser.parse_args(argv), designate_challenger)


if __name__ == "__main__":
    raise SystemExit(main())
