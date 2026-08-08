"""Contracts for explicitly described data artifacts."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AxisSpec(BaseModel):
    """A physical axis declared by the producer of an artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    unit: str
    direction: Literal["increasing", "decreasing", "unordered"]
    values_uri: str | None = None

    @field_validator("name", "unit")
    @classmethod
    def require_nonblank_value(cls, value: str) -> str:
        """Reject absent physical metadata instead of inferring it from shape."""
        if not value.strip():
            raise ValueError("axis name and unit must be non-blank")
        return value


class ArtifactRef(BaseModel):
    """A reference to an external artifact, never the array payload itself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    kind: Literal[
        "dense_array",
        "peak_table",
        "image",
        "mask",
        "region",
        "structure",
        "graph",
        "scalar",
        "class",
        "multilabel",
        "distribution",
        "metadata",
    ]
    uri: str
    sha256: str | None = None
    axes: tuple[AxisSpec, ...] = ()

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        """Allow only complete lowercase SHA-256 digests when supplied."""
        if value is not None and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value

    @model_validator(mode="after")
    def validate_axes(self) -> ArtifactRef:
        """Require explicit, unambiguous axes for dense data."""
        if self.kind == "dense_array" and not self.axes:
            raise ValueError("dense_array artifacts require at least one axis")

        axis_names = [axis.name for axis in self.axes]
        if len(axis_names) != len(set(axis_names)):
            raise ValueError("artifact axis names must be unique")
        return self
