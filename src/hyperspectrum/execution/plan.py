"""Immutable, reproducible execution planning with fail-closed readiness gates."""

from __future__ import annotations

import json
import platform
import re
from collections.abc import Mapping
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from typing_extensions import Self

from hyperspectrum.contracts import TaskSpec
from hyperspectrum.contracts.json import (
    FrozenJsonMapping,
    freeze_json_mapping,
    thaw_json_mapping,
)
from hyperspectrum.hyperdata.models import DatasetCandidate
from hyperspectrum.registry.models import (
    ResourceBudget,
    ToolAvailability,
    ToolManifest,
)
from hyperspectrum.tasks.recommend import ReadinessVerdict

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one JSON value deterministically for manifests and digests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    """Return the SHA-256 of stable, sorted JSON semantics."""

    return sha256(canonical_json_bytes(value)).hexdigest()


class RunPlan(BaseModel):
    """A detached declaration of every input that determines a run."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    schema_version: Literal["hyperspectrum-run-plan/v1"]
    task: TaskSpec
    dataset_code: str
    dataset_version: str
    data_digest: str
    tool_id: str
    tool_digest: str
    weight_digest: str
    model_digest: str
    environment_digest: str
    backend: Literal["local"]
    resources: ResourceBudget
    max_samples: int = Field(ge=1)
    output_directory: Path
    dry_run: bool
    parameters: FrozenJsonMapping
    data_origin: Literal["real", "synthetic-test"]

    @field_validator(
        "dataset_code",
        "dataset_version",
        "tool_id",
    )
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("plan identity fields must be non-blank")
        return value

    @field_validator("data_digest", "tool_digest", "model_digest", "environment_digest")
    @classmethod
    def require_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("plan digests must be 64 lowercase hexadecimal characters")
        return value

    @field_validator("weight_digest")
    @classmethod
    def require_weight_identity(cls, value: str) -> str:
        if value != "none" and _SHA256.fullmatch(value) is None:
            raise ValueError("weight digest must be 'none' or a 64-character SHA-256")
        return value

    @field_validator("parameters", mode="before")
    @classmethod
    def freeze_parameters(cls, value: object) -> FrozenJsonMapping:
        try:
            return freeze_json_mapping(value)
        except TypeError as error:
            raise ValueError(str(error)) from error

    @field_serializer("parameters")
    def serialize_parameters(self, value: FrozenJsonMapping) -> object:
        return thaw_json_mapping(value)

    @property
    def plan_digest(self) -> str:
        """Identify the complete plan, including destination and dry-run state."""

        return canonical_digest(self.model_dump(mode="json"))

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Revalidate copies so nested algorithm parameters stay immutable."""

        _ = deep
        data = self.model_dump(round_trip=True)
        if update is not None:
            data.update(update)
        return type(self).model_validate(data)


def build_run_plan(
    *,
    task: TaskSpec,
    dataset: DatasetCandidate,
    verdict: ReadinessVerdict,
    tool: ToolManifest,
    availability: ToolAvailability,
    backend: Literal["local"],
    resources: ResourceBudget,
    max_samples: int,
    output_directory: Path,
    dry_run: bool,
    parameters: Mapping[str, object],
    data_origin: Literal["real", "synthetic-test"],
) -> RunPlan:
    """Build a run plan only after data, task, tool, weight, and resource gates pass."""

    dataset_version = dataset.dataset_version
    if dataset_version is None or not dataset_version.strip():
        raise ValueError("dataset version must be pinned before planning")
    data_digest = dataset.content_digest
    if data_digest is None or _SHA256.fullmatch(data_digest) is None:
        raise ValueError("dataset content digest must be a 64-character SHA-256")
    _require_scoreable_task(task, verdict)
    _require_compatible_tool(task, tool)
    _require_available_tool(tool, availability)
    _require_resources(tool, resources)

    frozen_parameters = freeze_json_mapping(parameters)
    weight_digest = _weight_digest(tool)
    model_digest = canonical_digest(
        {
            "tool_digest": tool.tool_digest,
            "parameters": thaw_json_mapping(frozen_parameters),
        }
    )
    return RunPlan(
        schema_version="hyperspectrum-run-plan/v1",
        task=task,
        dataset_code=dataset.dataset_code,
        dataset_version=dataset_version,
        data_digest=data_digest,
        tool_id=tool.id,
        tool_digest=tool.tool_digest,
        weight_digest=weight_digest,
        model_digest=model_digest,
        environment_digest=_environment_digest(),
        backend=backend,
        resources=resources,
        max_samples=max_samples,
        output_directory=output_directory,
        dry_run=dry_run,
        parameters=frozen_parameters,
        data_origin=data_origin,
    )


def _require_scoreable_task(task: TaskSpec, verdict: ReadinessVerdict) -> None:
    if verdict.status != "scoreable":
        raise ValueError("dataset verdict must be scoreable before planning")
    if verdict.candidate_tasks != (task.task_type,):
        raise ValueError("scoreable verdict does not match the requested task")
    if verdict.ground_truth_roles != task.ground_truth_roles:
        raise ValueError("scoreable verdict ground truth does not match the task")
    if verdict.split_group_keys != task.split_group_keys:
        raise ValueError("scoreable verdict split groups do not match the task")


def _require_compatible_tool(task: TaskSpec, tool: ToolManifest) -> None:
    if task.modality not in tool.modalities or task.task_type not in tool.tasks:
        raise ValueError("tool does not support the requested task")
    if {artifact.role for artifact in tool.inputs} != set(task.input_roles):
        raise ValueError("tool inputs do not match the requested task")
    if any(artifact.kind != task.output_kind for artifact in tool.outputs):
        raise ValueError("tool output kind does not match the requested task")


def _require_available_tool(tool: ToolManifest, availability: ToolAvailability) -> None:
    if tool.weights.state == "required-missing":
        raise ValueError("required weights are missing")
    if not availability.available:
        reasons = ", ".join(availability.reasons)
        raise ValueError(f"tool is not verified as executable: {reasons}")


def _require_resources(tool: ToolManifest, resources: ResourceBudget) -> None:
    if resources.cpu < tool.resources.cpu:
        raise ValueError("backend CPU resources do not meet the tool requirement")
    if resources.memory_gb < tool.resources.memory_gb:
        raise ValueError("backend memory resources do not meet the tool requirement")
    if tool.resources.gpu == "required" and not resources.gpu_available:
        raise ValueError("backend GPU resources do not meet the tool requirement")


def _weight_digest(tool: ToolManifest) -> str:
    if not tool.weights.required:
        return "none"
    if tool.weights.state != "present" or tool.weights.digest is None:
        raise ValueError("required weights are missing")
    return tool.weights.digest


def _environment_digest() -> str:
    packages = {package: version(package) for package in ("numpy", "scipy")}
    return canonical_digest(
        {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "packages": packages,
        }
    )
