"""Immutable records returned by the HyperData CLI boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from typing_extensions import Self

from hyperspectrum.contracts.json import (
    FrozenJsonMapping,
    FrozenJsonValue,
    freeze_json_mapping,
    freeze_json_value,
    thaw_json_mapping,
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


class DatasetCandidate(BaseModel):
    """Immutable catalog evidence for a possible spectroscopy dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    dataset_code: str
    dataset_version: str | None
    content_digest: str | None
    # How the digest was established, and how well the bytes were checked — the
    # hub publishes both alongside the digest. They are a *different* axis from
    # the digest's value: one derived from the ingest-time files fingerprint
    # pins which bytes just as a quality-verified one does, even while the
    # quality status reads "unverified". Wires that do not publish them leave
    # these None, and None means absent, not negative.
    content_digest_source: str | None = None
    quality_status: str | None = None
    title: str
    description: str
    file_count: int
    parsed_file_count: int
    formats: tuple[str, ...]
    license: str | None
    evidence: FrozenJsonMapping

    @field_validator("evidence", mode="before")
    @classmethod
    def freeze_evidence(cls, value: object) -> FrozenJsonMapping:
        """Detach nested discovery evidence from caller-owned JSON objects."""
        return freeze_json_mapping(value)

    @field_serializer("evidence")
    def serialize_evidence(self, value: FrozenJsonMapping) -> object:
        """Expose standard JSON values through Pydantic serialization."""
        return thaw_json_mapping(value)

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Revalidate updates so Pydantic copies retain evidence immutability."""
        _ = deep
        data = self.model_dump(round_trip=True)
        if update is not None:
            data.update(update)
        return type(self).model_validate(data)
