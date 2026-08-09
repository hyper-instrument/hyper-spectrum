"""Contract for reproducible model and tool predictions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from typing_extensions import Self

from .artifact import ArtifactRef
from .json import FrozenJsonMapping, freeze_json_mapping, thaw_json_mapping

_COMMON_REQUIRED_DIGESTS = (
    "model_digest",
    "tool_digest",
    "environment_digest",
)
_V2_SHA256_DIGESTS = (
    "model_digest",
    "tool_digest",
    "implementation_digest",
    "data_digest",
    "environment_digest",
    "plan_digest",
)
_V3_SHA256_DIGESTS = (
    "model_digest",
    "tool_digest",
    "implementation_digest",
    "source_dataset_digest",
    "source_content_manifest_digest",
    "benchmark_asset_digest",
    "environment_digest",
    "plan_digest",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _PredictionBundleBase(BaseModel):
    """Shared immutable prediction fields without a wire-version assertion."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

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
        for key in _COMMON_REQUIRED_DIGESTS:
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


class PredictionBundle(_PredictionBundleBase):
    """Backward-compatible v1 output manifest."""

    schema_version: Literal["hyperspectrum-prediction/v1"]

    @field_validator("provenance")
    @classmethod
    def require_v1_data_identity(cls, value: FrozenJsonMapping) -> FrozenJsonMapping:
        data_digest = value.get("data_digest")
        if not isinstance(data_digest, str) or not data_digest.strip():
            raise ValueError("provenance data_digest must be a non-empty string")
        return value


class PredictionBundleV2(_PredictionBundleBase):
    """Selection-bound prediction contract for M0 success handoffs."""

    schema_version: Literal["hyperspectrum-prediction/v2"]

    @field_validator("provenance")
    @classmethod
    def require_v2_provenance(cls, value: FrozenJsonMapping) -> FrozenJsonMapping:
        for key in _V2_SHA256_DIGESTS:
            digest = value.get(key)
            if (
                not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
                or digest == "0" * 64
            ):
                raise ValueError(f"provenance {key} must be a lowercase SHA-256")
        weight_digest = value.get("weight_digest")
        if weight_digest != "none" and (
            not isinstance(weight_digest, str)
            or _SHA256.fullmatch(weight_digest) is None
            or weight_digest == "0" * 64
        ):
            raise ValueError(
                "provenance weight_digest must be 'none' or a lowercase SHA-256"
            )
        if value.get("plan_schema_version") != "hyperspectrum-run-plan/v2":
            raise ValueError(
                "provenance plan_schema_version must identify a selection-bound v2 plan"
            )
        for key in ("dataset_code", "dataset_version"):
            identity = value.get(key)
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f"provenance {key} must be non-blank")
        return value


class PredictionBundleV3(_PredictionBundleBase):
    """Prediction provenance with unambiguous source and derived-asset digests."""

    schema_version: Literal["hyperspectrum-prediction/v3"]

    @field_validator("provenance")
    @classmethod
    def require_v3_provenance(cls, value: FrozenJsonMapping) -> FrozenJsonMapping:
        if "data_digest" in value:
            raise ValueError("provenance data_digest is forbidden in v3")
        for key in _V3_SHA256_DIGESTS:
            digest = value.get(key)
            if (
                not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
                or digest == "0" * 64
            ):
                raise ValueError(f"provenance {key} must be a lowercase SHA-256")
        explicit_identities = {
            value["source_dataset_digest"],
            value["source_content_manifest_digest"],
            value["benchmark_asset_digest"],
        }
        if len(explicit_identities) != 3:
            raise ValueError(
                "provenance benchmark_asset_digest must be distinct from source identities"
            )
        weight_digest = value.get("weight_digest")
        if weight_digest != "none" and (
            not isinstance(weight_digest, str)
            or _SHA256.fullmatch(weight_digest) is None
            or weight_digest == "0" * 64
        ):
            raise ValueError(
                "provenance weight_digest must be 'none' or a lowercase SHA-256"
            )
        if value.get("plan_schema_version") != "hyperspectrum-run-plan/v3":
            raise ValueError(
                "provenance plan_schema_version must identify an explicit-digest v3 plan"
            )
        for key in ("dataset_code", "dataset_version"):
            identity = value.get(key)
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f"provenance {key} must be non-blank")
        return value
