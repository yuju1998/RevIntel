"""Deterministic randomness: independent numpy streams, UUIDs and Faker instances.

Every stage draws from its own stream spawned from one SeedSequence, so the same seed and
arguments always reproduce the same rows. Nothing reads the wall clock or uuid4().
"""

import uuid

import numpy as np
from faker import Faker

STREAMS = (
    "users",
    "accounts",
    "contacts",
    "leads",
    "simulation",
    "account_activities",
    "ids",
    "text",
)


class Streams:
    def __init__(self, seed: int) -> None:
        root = np.random.SeedSequence(seed)
        rng_seeds = root.spawn(len(STREAMS))
        faker_seeds = root.spawn(len(STREAMS))
        self._rngs = {
            name: np.random.default_rng(child)
            for name, child in zip(STREAMS, rng_seeds, strict=True)
        }
        self._faker_seeds = {
            name: int(child.generate_state(1)[0])
            for name, child in zip(STREAMS, faker_seeds, strict=True)
        }

    def rng(self, name: str) -> np.random.Generator:
        return self._rngs[name]

    def faker(self, name: str) -> Faker:
        fake = Faker("en_US")
        fake.seed_instance(self._faker_seeds[name])
        return fake


def deterministic_uuids(rng: np.random.Generator, count: int) -> list[uuid.UUID]:
    """Version-4-shaped UUIDs drawn from `rng` (reproducible, unlike uuid.uuid4())."""
    raw = rng.integers(0, 256, size=(count, 16), dtype=np.uint8)
    raw[:, 6] = (raw[:, 6] & 0x0F) | 0x40  # version 4
    raw[:, 8] = (raw[:, 8] & 0x3F) | 0x80  # RFC 4122 variant
    return [uuid.UUID(bytes=row.tobytes()) for row in raw]
