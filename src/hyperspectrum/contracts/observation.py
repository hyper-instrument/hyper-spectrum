"""Contract for a single scientific observation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from typing_extensions import Self

from .artifact import ArtifactRef
from .json import FrozenJsonMapping, freeze_json_mapping, thaw_json_mapping


class ObservationBundle(BaseModel):
    """Immutable manifest and references for one observed sample."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    schema_version: Literal["hyperspectrum-observation/v1"]
    sample_id: str
    modality: str
    artifacts: tuple[ArtifactRef, ...]
    context: FrozenJsonMapping
    labels: FrozenJsonMapping
    provenance: FrozenJsonMapping

    @field_validator("context", "labels", "provenance", mode="before")
    @classmethod
    def freeze_json_metadata(cls, value: object) -> FrozenJsonMapping:
        """Detach and freeze JSON metadata before storing it in the bundle."""
        try:
            return freeze_json_mapping(value)
        except TypeError as error:
            raise ValueError(str(error)) from error

    @field_serializer("context", "labels", "provenance")
    def serialize_json_metadata(self, value: FrozenJsonMapping) -> object:
        """Keep Pydantic dumps suitable for canonical JSON digesting."""
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
