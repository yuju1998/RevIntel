"""Column type helpers shared by ORM models."""

from enum import StrEnum

import sqlalchemy as sa

ENUM_LENGTH = 32


def str_enum(enum_cls: type[StrEnum], name: str) -> sa.Enum:
    """Non-native enum: VARCHAR(32) plus a CHECK constraint named `ck_<table>_<name>`."""
    return sa.Enum(
        enum_cls,
        name=name,
        native_enum=False,
        # SQLAlchemy 2.x does not create the CHECK constraint unless asked to.
        create_constraint=True,
        # Fixed length so new values can be added without widening the column.
        length=ENUM_LENGTH,
        values_callable=lambda members: [member.value for member in members],
        validate_strings=True,
    )
