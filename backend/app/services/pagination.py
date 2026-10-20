"""Paginated results returned by services."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageResult[T]:
    items: Sequence[T]
    # Number of rows matching the filters, not the whole table.
    total: int
    limit: int
    offset: int
