"""Contract for reproducible model and tool predictions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from typing_extensions import Self

from .artifact import ArtifactRef
from .json import FrozenJsonMapping, freeze_json_mapping, thaw_json_mapping

_REQUIRED_DIGESTS = (
    "model_digest",
    "tool_digest",
    "data_digest",
    "environment_digest",
)


class PredictionBundle(BaseModel):
    """Immutable output manifest with mandatory execution provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    schema_version: Literal["hyperspectrum-prediction/v1"]
    run_id: str
    task_id: str
    predictions: tuple[ArtifactRef, ...]
    failures: tuple[FrozenJsonMapping, ...]
    provenance: FrozenJsonMapping

    @field_validator("failures", mode="before")
    @classmethod
    def freeze_failures(cls, value: object) -> tuple[FrozenJsonMapping, ...]:
        """Detach every failure record from caller-owned JSON containers."""
        try:
            if not isinstance(value, (list, tuple)):
                raise TypeError("failures must be a JSON array")
            return tuple(freeze_json_mapping(item) for item in value)
        except TypeError as error:
            raise ValueError(str(error)) from error

    @field_validator("provenance", mode="before")
    @classmethod
    def freeze_provenance(cls, value: object) -> FrozenJsonMapping:
        """Detach and freeze the reproducibility metadata before validation."""
        try:
            return freeze_json_mapping(value)
        except TypeError as error:
            raise ValueError(str(error)) from error

    @field_validator("provenance")
    @classmethod
    def require_provenance_digests(cls, value: FrozenJsonMapping) -> FrozenJsonMapping:
        """Ensure every prediction can be traced to immutable inputs and runtime."""
        for key in _REQUIRED_DIGESTS:
            digest = value.get(key)
            if not isinstance(digest, str) or not digest.strip():
                raise ValueError(f"provenance {key} must be a non-empty string")
        return value

    @field_serializer("failures")
    def serialize_failures(self, value: tuple[FrozenJsonMapping, ...]) -> object:
        """Keep failure records in standard JSON object form when dumped."""
        return [thaw_json_mapping(failure) for failure in value]

    @field_serializer("provenance")
    def serialize_provenance(self, value: FrozenJsonMapping) -> object:
        """Keep provenance in standard JSON object form when dumped."""
        return thaw_json_mapping(value)

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Revalidate updates so copied bundles retain immutable metadata."""
        _ = deep
        data = self.model_dump(round_trip=True)
        if update is not None:
            data.update(update)
        return type(self).model_validate(data)
