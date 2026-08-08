"""Immutable records returned by the HyperData CLI boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from typing_extensions import Self

from hyperspectrum.contracts.json import (
    FrozenJsonValue,
    freeze_json_value,
    thaw_json_value,
)


class HydCommandResult(BaseModel):
    """A detached, immutable result of one external HyperData CLI command."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    payload: object | None

    @field_validator("payload", mode="before")
    @classmethod
    def freeze_payload(cls, value: object | None) -> object | None:
        """Detach JSON output so callers cannot mutate the recorded response."""
        if value is None:
            return None
        return freeze_json_value(value)

    @field_serializer("payload")
    def serialize_payload(self, value: object | None) -> object | None:
        """Expose standard JSON values through Pydantic serialization."""
        if value is None:
            return None
        return thaw_json_value(cast(FrozenJsonValue, value))

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Revalidate updates so Pydantic copies retain payload immutability."""
        _ = deep
        data = self.model_dump(round_trip=True)
        if update is not None:
            data.update(update)
        return type(self).model_validate(data)
