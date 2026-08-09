"""Typed semantic model for the versioned tool-manifest contract.

Artifact roles are deliberately a closed vocabulary: ``raw_signal``,
``denoised_signal``, ``normalized_signal``, ``ground_truth``, ``prediction``,
``metadata``, ``mask``, ``region``, ``peak_table``, ``structure``, ``graph``,
``scalar``, ``class``, ``multilabel``, and ``distribution``.  A manifest must
not invent an undeclared role because matching is based on these roles.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import PurePath
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_IMAGE_DIGEST = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_VERSION = re.compile(
    r"^v?\d+(?:\.\d+)+(?:[-.]?(?:a|alpha|b|beta|rc|pre|preview|post|dev)\d*(?:\.\d+)*)?(?:\+[0-9A-Za-z.-]+)?$"
)
_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")

ArtifactKind = Literal[
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
Distribution = Literal["open-distribution", "private-validation-only"]
WeightsState = Literal["not-required", "present", "required-missing"]
GpuNeed = Literal["none", "optional", "required"]

ARTIFACT_ROLES = frozenset(
    {
        "raw_signal",
        "denoised_signal",
        "normalized_signal",
        "ground_truth",
        "prediction",
        "metadata",
        "mask",
        "region",
        "peak_table",
        "structure",
        "graph",
        "scalar",
        "class",
        "multilabel",
        "distribution",
    }
)


class RuntimeSpec(BaseModel):
    """A runtime, with container images permitted only by immutable digest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["python", "container"]
    image: str | None = None

    @field_validator("image")
    @classmethod
    def require_immutable_image_digest(cls, value: str | None) -> str | None:
        if value is not None and _IMAGE_DIGEST.fullmatch(value) is None:
            raise ValueError("runtime image must be pinned by an immutable digest")
        return value

    @model_validator(mode="after")
    def require_image_for_container(self) -> RuntimeSpec:
        if self.kind == "container" and self.image is None:
            raise ValueError(
                "container runtime requires an image pinned by an immutable digest"
            )
        return self


class ArtifactSpec(BaseModel):
    """One typed input or output with a role from the documented vocabulary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    kind: ArtifactKind

    @field_validator("role")
    @classmethod
    def require_known_artifact_role(cls, value: str) -> str:
        if value not in ARTIFACT_ROLES:
            raise ValueError(f"unknown artifact role: {value}")
        return value


class ResourceRequirements(BaseModel):
    """Minimum resources a selected tool requires."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu: int = Field(ge=1)
    memory_gb: float = Field(gt=0)
    gpu: GpuNeed


class SourceSpec(BaseModel):
    """Reproducible local-package or external Git source metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["git", "local-package"]
    repository: str | None = None
    commit: str | None = None
    release: str | None = None
    package: str | None = None
    version: str | None = None
    revision: str | None = None

    @model_validator(mode="after")
    def require_reproducible_source_metadata(self) -> SourceSpec:
        if self.kind == "git":
            if not self.repository:
                raise ValueError("git source requires a repository")
            if (self.commit is None) == (self.release is None):
                raise ValueError("git source requires exactly one commit or release")
            if self.commit is not None and _GIT_COMMIT.fullmatch(self.commit) is None:
                raise ValueError(
                    "source commit must be a 40-character lowercase Git commit"
                )
            if (
                self.release is not None
                and _RELEASE_VERSION.fullmatch(self.release) is None
            ):
                raise ValueError("source release must be an immutable release version")
            return self

        if not self.package or not self.version or not self.revision:
            raise ValueError(
                "local-package source requires package, version, and revision"
            )
        if _RELEASE_VERSION.fullmatch(self.version) is None:
            raise ValueError(
                "local-package source version must be an immutable release version"
            )
        if (
            _GIT_COMMIT.fullmatch(self.revision) is None
            and _RELEASE_VERSION.fullmatch(self.revision) is None
        ):
            raise ValueError(
                "local-package source revision must be an immutable revision"
            )
        return self


class WeightsSpec(BaseModel):
    """A declared weight state; tools can never download weights automatically."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    required: bool
    state: WeightsState
    allow_download: Literal[False]
    digest: str | None = None
    asset_id: str | None = None
    source_url: str | None = None
    filename: str | None = None
    size_bytes: int | None = Field(default=None, gt=0)
    license: str | None = None

    @field_validator("digest")
    @classmethod
    def require_sha256_weight_digest(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_DIGEST.fullmatch(value) is None:
            raise ValueError("weight digest must be a 64-character lowercase SHA-256")
        return value

    @field_validator("asset_id", "source_url", "filename", "license")
    @classmethod
    def require_nonblank_asset_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("weight asset metadata must be non-blank")
        return value

    @model_validator(mode="after")
    def require_consistent_weight_state(self) -> WeightsSpec:
        if self.required and self.state == "not-required":
            raise ValueError("required weights cannot have state not-required")
        if not self.required and self.state != "not-required":
            raise ValueError("optional weights must have state not-required")
        if self.state == "present" and self.digest is None:
            raise ValueError("present weights require a weight digest")
        if self.state != "present" and self.digest is not None:
            raise ValueError("only present weights may declare a weight digest")
        asset_values = (
            self.asset_id,
            self.source_url,
            self.filename,
            self.size_bytes,
            self.license,
        )
        if self.state == "present" and any(value is None for value in asset_values):
            raise ValueError(
                "present weights require complete declarative asset metadata"
            )
        if self.state != "present" and any(value is not None for value in asset_values):
            raise ValueError("only present weights may declare asset metadata")
        if self.source_url is not None:
            parsed = urlparse(self.source_url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("weight source_url must be an absolute HTTPS URL")
        if self.filename is not None and PurePath(self.filename).name != self.filename:
            raise ValueError("weight filename must be a basename")
        return self


class TrainingSpec(BaseModel):
    """Training is deliberately unavailable during M0 tool selection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: Literal[False]


class ToolManifest(BaseModel):
    """The immutable semantic representation of a ``hyperspectrum-tool/v1`` file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["hyperspectrum-tool/v1"]
    id: str
    version: str
    modalities: tuple[str, ...] = Field(min_length=1)
    tasks: tuple[str, ...] = Field(min_length=1)
    runtime: RuntimeSpec
    entrypoint: str
    inputs: tuple[ArtifactSpec, ...]
    outputs: tuple[ArtifactSpec, ...] = Field(min_length=1)
    resources: ResourceRequirements
    verify: tuple[str, ...] = Field(min_length=1)
    source: SourceSpec
    license: str
    distribution: Distribution
    weights: WeightsSpec
    training: TrainingSpec

    @field_validator("id", "version", "entrypoint", "license")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tool manifest text fields must be non-blank")
        return value

    @field_validator("modalities", "tasks")
    @classmethod
    def require_unique_nonblank_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values) or len(values) != len(
            set(values)
        ):
            raise ValueError("modalities and tasks must be unique non-blank values")
        return values

    @field_validator("verify")
    @classmethod
    def require_nonempty_argv(cls, argv: tuple[str, ...]) -> tuple[str, ...]:
        if any(not arg.strip() for arg in argv):
            raise ValueError(
                "verify must be a non-empty argv list of non-blank strings"
            )
        return argv

    @model_validator(mode="after")
    def require_unique_roles(self) -> ToolManifest:
        for direction, artifacts in (("input", self.inputs), ("output", self.outputs)):
            roles = tuple(artifact.role for artifact in artifacts)
            if len(roles) != len(set(roles)):
                raise ValueError(f"{direction} artifact roles must be unique")
        return self

    @property
    def tool_digest(self) -> str:
        """SHA-256 over canonical semantic JSON, independent of YAML formatting."""
        canonical = json.dumps(
            self.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return sha256(canonical).hexdigest()


class ResourceBudget(BaseModel):
    """Resources currently available to satisfy a manifest's requirements."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu: int = Field(ge=1)
    memory_gb: float = Field(gt=0)
    gpu_available: bool


class LicensePolicy(BaseModel):
    """Explicit policy supplied by the caller, never inferred from a tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_licenses: tuple[str, ...] = Field(min_length=1)
    allowed_distributions: tuple[Distribution, ...] = Field(min_length=1)

    @classmethod
    def private_validation(cls) -> LicensePolicy:
        """M0 policy allowing unknown-license tools only for private validation."""
        return cls(
            allowed_licenses=("unknown",),
            allowed_distributions=("private-validation-only",),
        )

    @classmethod
    def open_distribution(cls) -> LicensePolicy:
        """Policy for a later approved open distribution path."""
        return cls(
            allowed_licenses=("approved",),
            allowed_distributions=("open-distribution",),
        )


class ToolMatchRequest(BaseModel):
    """All gates that must pass before a registry tool may be selected."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    modality: str
    task: str
    input_roles: tuple[str, ...]
    output_roles: tuple[str, ...]
    license_policy: LicensePolicy
    weights_state: WeightsState
    resources: ResourceBudget

    @field_validator("input_roles", "output_roles")
    @classmethod
    def require_known_unique_roles(cls, roles: tuple[str, ...]) -> tuple[str, ...]:
        if any(role not in ARTIFACT_ROLES for role in roles):
            raise ValueError("match request contains an unknown artifact role")
        if len(roles) != len(set(roles)):
            raise ValueError("match request artifact roles must be unique")
        return roles


AvailabilityReason = Literal[
    "container-unverified",
    "entrypoint-unresolvable",
    "verify-command-unresolvable",
    "verify-module-unresolvable",
    "weights-required-missing",
    "weights-unverified",
    "input_contract_unverified",
]


class ToolRejection(BaseModel):
    """Machine-readable reasons one tool failed an explicit match request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_id: str
    reasons: tuple[str, ...] = Field(min_length=1)


class ToolAvailability(BaseModel):
    """Safe, machine-readable executable availability determined by a resolver."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    available: bool
    reasons: tuple[AvailabilityReason, ...] = ()

    @model_validator(mode="after")
    def require_consistent_availability(self) -> ToolAvailability:
        if self.available and self.reasons:
            raise ValueError("available tools cannot have unavailability reasons")
        if not self.available and not self.reasons:
            raise ValueError("unavailable tools require machine-readable reasons")
        return self
