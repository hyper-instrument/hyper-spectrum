"""Immutable, reproducible execution planning with fail-closed readiness gates."""

from __future__ import annotations

import ast
import json
import platform
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib.machinery import ModuleSpec, PathFinder
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
    implementation_digest: str
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

    @field_validator(
        "data_digest",
        "tool_digest",
        "implementation_digest",
        "model_digest",
        "environment_digest",
    )
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
    _require_scoreable_task(task, dataset, verdict)
    _require_compatible_tool(task, tool)
    _require_available_tool(tool, availability)
    _require_resources(tool, resources)

    frozen_parameters = freeze_json_mapping(parameters)
    resolved = resolve_local_entrypoint(tool)
    weight_digest = _weight_digest(tool)
    model_digest = canonical_digest(
        {
            "tool_digest": tool.tool_digest,
            "implementation_digest": resolved.implementation_digest,
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
        implementation_digest=resolved.implementation_digest,
        weight_digest=weight_digest,
        model_digest=model_digest,
        environment_digest=current_environment_digest(),
        backend=backend,
        resources=resources,
        max_samples=max_samples,
        output_directory=output_directory,
        dry_run=dry_run,
        parameters=frozen_parameters,
        data_origin=data_origin,
    )


def _require_scoreable_task(
    task: TaskSpec, dataset: DatasetCandidate, verdict: ReadinessVerdict
) -> None:
    if (
        verdict.dataset_code != dataset.dataset_code
        or verdict.dataset_version != dataset.dataset_version
        or verdict.content_digest != dataset.content_digest
    ):
        raise ValueError(
            "scoreable verdict dataset identity does not match the dataset"
        )
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
    if len(tool.outputs) != 1 or (
        tool.outputs[0].role,
        tool.outputs[0].kind,
    ) != ("denoised_signal", "dense_array"):
        raise ValueError("denoising tools require one dense denoised_signal output")


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


def current_environment_digest() -> str:
    """Digest the Python and numerical-library environment used by execution."""

    packages = {package: version(package) for package in ("numpy", "scipy")}
    return canonical_digest(
        {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "packages": packages,
        }
    )


@dataclass(frozen=True, slots=True)
class ResolvedEntrypoint:
    """Static local entrypoint identity; resolving it does not import tool code."""

    module_name: str
    object_name: str
    source_path: Path
    source_bytes: bytes
    implementation_digest: str


def resolve_local_entrypoint(tool: ToolManifest) -> ResolvedEntrypoint:
    """Resolve local Python source bytes without importing or executing the tool."""

    if tool.runtime.kind != "python" or ":" not in tool.entrypoint:
        raise ValueError("local execution requires a Python module:object entrypoint")
    module_name, object_name = tool.entrypoint.split(":", 1)
    if not object_name.isidentifier():
        raise ValueError("tool entrypoint object must be a Python identifier")
    spec = _module_spec(module_name)
    if spec is None or spec.origin is None:
        raise ValueError("tool entrypoint module is not resolvable")
    source_path = Path(spec.origin).resolve()
    repository_root = Path(__file__).resolve().parents[3]
    if source_path.suffix != ".py" or not source_path.is_relative_to(repository_root):
        raise ValueError("tool entrypoint must resolve to local Python source")
    source_bytes = source_path.read_bytes()
    try:
        module = ast.parse(source_bytes, filename=str(source_path))
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ValueError("tool entrypoint source is not valid Python") from error
    if not any(
        isinstance(statement, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef))
        and statement.name == object_name
        for statement in module.body
    ):
        raise ValueError("tool entrypoint object is not defined in its source module")
    return ResolvedEntrypoint(
        module_name=module_name,
        object_name=object_name,
        source_path=source_path,
        source_bytes=source_bytes,
        implementation_digest=sha256(source_bytes).hexdigest(),
    )


def _module_spec(module_name: str) -> ModuleSpec | None:
    parts = module_name.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    spec = PathFinder.find_spec(parts[0])
    for part in parts[1:]:
        if spec is None or spec.submodule_search_locations is None:
            return None
        spec = PathFinder.find_spec(part, spec.submodule_search_locations)
    return spec
