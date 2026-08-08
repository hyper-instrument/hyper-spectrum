"""Contract for a single scientific observation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .artifact import ArtifactRef


class ObservationBundle(BaseModel):
    """Immutable manifest and references for one observed sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["hyperspectrum-observation/v1"]
    sample_id: str
    modality: str
    artifacts: tuple[ArtifactRef, ...]
    context: dict[str, object]
    labels: dict[str, object]
    provenance: dict[str, object]
