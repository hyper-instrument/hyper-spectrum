"""Contract for reproducible model and tool predictions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from .artifact import ArtifactRef

_REQUIRED_DIGESTS = (
    "model_digest",
    "tool_digest",
    "data_digest",
    "environment_digest",
)


class PredictionBundle(BaseModel):
    """Immutable output manifest with mandatory execution provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["hyperspectrum-prediction/v1"]
    run_id: str
    task_id: str
    predictions: tuple[ArtifactRef, ...]
    failures: tuple[dict[str, object], ...]
    provenance: dict[str, object]

    @field_validator("provenance")
    @classmethod
    def require_provenance_digests(cls, value: dict[str, object]) -> dict[str, object]:
        """Ensure every prediction can be traced to immutable inputs and runtime."""
        for key in _REQUIRED_DIGESTS:
            digest = value.get(key)
            if not isinstance(digest, str) or not digest.strip():
                raise ValueError(f"provenance {key} must be a non-empty string")
        return value
