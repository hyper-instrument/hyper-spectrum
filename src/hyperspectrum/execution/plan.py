"""Immutable, reproducible execution planning with fail-closed readiness gates."""

from __future__ import annotations

import ast
import json
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.machinery import ModuleSpec, PathFinder
from importlib.metadata import version
from importlib.util import resolve_name
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from hyperspectrum.contracts import TaskSpec
from hyperspectrum.contracts.json import (
    FrozenJsonMapping,
    freeze_json_mapping,
    thaw_json_mapping,
)
from hyperspectrum.datasets import BenchmarkAssetIdentity
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


class _RunPlanBase(BaseModel):
    """Common detached inputs shared by immutable wire plan versions."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    task: TaskSpec
    dataset_code: str
    dataset_version: str
    tool_id: str
    tool_digest: str
    implementation_digest: str
    weight_digest: str
    model_digest: str
    environment_digest: str
    backend: Literal["local"]
    resources: ResourceBudget
    max_samples: int = Field(ge=1)
    selection_policy: Literal["explicit_order"]
    selection_policy_version: Literal["1"]
    selected_sample_ids: tuple[str, ...]
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

    @field_validator("selected_sample_ids")
    @classmethod
    def require_explicit_unique_selection(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if not value:
            raise ValueError("selected sample IDs must not be empty")
        if any(not sample_id.strip() for sample_id in value):
            raise ValueError("selected sample IDs must be non-blank")
        if len(set(value)) != len(value):
            raise ValueError("selected sample IDs must be unique")
        return value

    @model_validator(mode="after")
    def require_selection_within_limit(self) -> Self:
        if len(self.selected_sample_ids) > self.max_samples:
            raise ValueError("selected sample count exceeds max_samples")
        return self

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


class RunPlanV2(_RunPlanBase):
    """Legacy selection-bound plan whose data_digest names input NPZ bytes."""

    schema_version: Literal["hyperspectrum-run-plan/v2"]
    data_digest: str

    @field_validator("data_digest")
    @classmethod
    def require_data_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("plan digests must be 64 lowercase hexadecimal characters")
        return value


class RunPlanV3(_RunPlanBase):
    """Plan with separate source, verified-content, and derived-asset identity."""

    schema_version: Literal["hyperspectrum-run-plan/v3"]
    source_dataset_digest: str
    source_content_manifest_digest: str
    benchmark_asset_digest: str

    @field_validator(
        "source_dataset_digest",
        "source_content_manifest_digest",
        "benchmark_asset_digest",
    )
    @classmethod
    def require_explicit_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("plan digests must be 64 lowercase hexadecimal characters")
        return value


RunPlanType: TypeAlias = RunPlanV2 | RunPlanV3
RunPlan = RunPlanV2


def parse_run_plan(value: object) -> RunPlanType:
    """Dispatch a serialized selection-bound plan without guessing field meaning."""

    if not isinstance(value, Mapping):
        raise TypeError("run plan must be a JSON object")
    schema_version = value.get("schema_version")
    if schema_version == "hyperspectrum-run-plan/v2":
        return RunPlanV2.model_validate(value)
    if schema_version == "hyperspectrum-run-plan/v3":
        return RunPlanV3.model_validate(value)
    raise ValueError(f"unsupported run plan schema: {schema_version}")


def build_run_plan(
    *,
    task: TaskSpec,
    dataset: DatasetCandidate,
    verdict: ReadinessVerdict,
    benchmark_asset: BenchmarkAssetIdentity,
    tool: ToolManifest,
    availability: ToolAvailability,
    backend: Literal["local"],
    resources: ResourceBudget,
    max_samples: int,
    selected_sample_ids: Sequence[str],
    output_directory: Path,
    dry_run: bool,
    parameters: Mapping[str, object],
    data_origin: Literal["real", "synthetic-test"],
) -> RunPlanV3:
    """Build a run plan only after data, task, tool, weight, and resource gates pass."""

    dataset_version = dataset.dataset_version
    if dataset_version is None or not dataset_version.strip():
        raise ValueError("dataset version must be pinned before planning")
    catalog_digest = dataset.content_digest
    if catalog_digest is None or _SHA256.fullmatch(catalog_digest) is None:
        raise ValueError("dataset content digest must be a 64-character SHA-256")
    if (
        benchmark_asset.dataset_code != dataset.dataset_code
        or benchmark_asset.dataset_version != dataset_version
    ):
        raise ValueError("benchmark asset dataset identity does not match the dataset")
    if benchmark_asset.declared_manifest_digest != catalog_digest:
        raise ValueError(
            "benchmark asset declared manifest digest does not match the catalog"
        )
    _require_scoreable_task(task, dataset, verdict)
    _require_compatible_tool(task, tool)
    _require_available_tool(tool, availability)
    _require_resources(tool, resources)

    frozen_parameters = freeze_json_mapping(parameters)
    _require_tool_parameters(tool, frozen_parameters)
    resolved = resolve_local_entrypoint(tool)
    weight_digest = _weight_digest(tool)
    model_digest = canonical_digest(
        {
            "tool_digest": tool.tool_digest,
            "implementation_digest": resolved.implementation_digest,
            "parameters": thaw_json_mapping(frozen_parameters),
        }
    )
    return RunPlanV3(
        schema_version="hyperspectrum-run-plan/v3",
        task=task,
        dataset_code=dataset.dataset_code,
        dataset_version=dataset_version,
        source_dataset_digest=benchmark_asset.source_dataset_digest,
        source_content_manifest_digest=(benchmark_asset.source_content_manifest_digest),
        benchmark_asset_digest=benchmark_asset.benchmark_asset_digest,
        tool_id=tool.id,
        tool_digest=tool.tool_digest,
        implementation_digest=resolved.implementation_digest,
        weight_digest=weight_digest,
        model_digest=model_digest,
        environment_digest=current_environment_digest(),
        backend=backend,
        resources=resources,
        max_samples=max_samples,
        selection_policy="explicit_order",
        selection_policy_version="1",
        selected_sample_ids=tuple(selected_sample_ids),
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
    if tool.outputs[0].kind != task.output_kind:
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


def _require_tool_parameters(tool: ToolManifest, parameters: FrozenJsonMapping) -> None:
    """Freeze non-negotiable scientific semantics for fixed-weight adapters."""

    if tool.id != "xasdenoise":
        return
    expected_keys = {
        "input_contract_status",
        "required_input_artifact",
        "model_normalization_method",
        "preprocessing",
        "device",
    }
    if set(parameters) != expected_keys:
        raise ValueError(
            "XASDenoise parameters must declare only its input contract, preprocessing, and device"
        )
    if parameters["input_contract_status"] != "unverified":
        raise ValueError("XASDenoise raw input contract remains unverified")
    if (
        parameters["required_input_artifact"]
        != "structured_sample_bound_upstream_normalization_artifact"
    ):
        raise ValueError(
            "XASDenoise requires a structured sample-bound normalization artifact"
        )
    if parameters["model_normalization_method"] is not None:
        raise ValueError(
            "the official XASDenoise checkpoint declares null normalization"
        )
    preprocessing = parameters["preprocessing"]
    expected_preprocessing = freeze_json_mapping(
        {
            "schema_version": "hyperspectrum-xasdenoise-step-baseline/v1",
            "method": "symmetric_tanh_step",
            "inverse": "add_same_fitted_baseline",
        }
    )
    if preprocessing != expected_preprocessing:
        raise ValueError(
            "XASDenoise requires the versioned symmetric-tanh step-baseline transform"
        )
    device = parameters["device"]
    if (
        not isinstance(device, str)
        or re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", device) is None
    ):
        raise ValueError("XASDenoise device must be auto, cpu, cuda, or cuda:N")


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
class ResolvedModule:
    """One exact project-local source module in an implementation closure."""

    module_name: str
    source_path: Path
    repository_path: str
    source_bytes: bytes


@dataclass(frozen=True, slots=True)
class ResolvedEntrypoint:
    """Static local entrypoint closure; resolving it never imports tool code."""

    module_name: str
    object_name: str
    modules: tuple[ResolvedModule, ...]
    implementation_digest: str


def resolve_local_entrypoint(tool: ToolManifest) -> ResolvedEntrypoint:
    """Resolve exact project-local dependency bytes without importing tool code."""

    if tool.runtime.kind != "python" or ":" not in tool.entrypoint:
        raise ValueError("local execution requires a Python module:object entrypoint")
    module_name, object_name = tool.entrypoint.split(":", 1)
    if not object_name.isidentifier():
        raise ValueError("tool entrypoint object must be a Python identifier")
    package_source_root = _package_source_root()
    if package_source_root is None:
        raise ValueError("local package source root cannot be resolved")
    resolved_modules, parsed_modules = _resolve_local_dependency_closure(
        module_name, package_source_root
    )
    module = parsed_modules[module_name]
    if not any(
        isinstance(statement, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef))
        and statement.name == object_name
        for statement in module.body
    ):
        raise ValueError("tool entrypoint object is not defined in its source module")
    identities = [
        {
            "module": resolved.module_name,
            "path": resolved.repository_path,
            "sha256": sha256(resolved.source_bytes).hexdigest(),
        }
        for resolved in resolved_modules
    ]
    return ResolvedEntrypoint(
        module_name=module_name,
        object_name=object_name,
        modules=resolved_modules,
        implementation_digest=canonical_digest({"modules": identities}),
    )


def _resolve_local_dependency_closure(
    entrypoint_module: str, package_source_root: Path
) -> tuple[tuple[ResolvedModule, ...], dict[str, ast.Module]]:
    pending = [entrypoint_module]
    resolved: dict[str, ResolvedModule] = {}
    parsed: dict[str, ast.Module] = {}
    while pending:
        module_name = pending.pop()
        if module_name in resolved:
            continue
        source = _resolve_local_module(module_name, package_source_root)
        if source is None:
            if module_name == entrypoint_module:
                raise ValueError("tool entrypoint must resolve to local Python source")
            continue
        resolved_module, syntax = source
        resolved[module_name] = resolved_module
        parsed[module_name] = syntax
        pending.extend(
            dependency
            for dependency in _local_import_candidates(module_name, syntax)
            if dependency not in resolved
            and _resolve_local_module_path(dependency, package_source_root) is not None
        )
    ordered_names = sorted(resolved)
    return tuple(resolved[name] for name in ordered_names), parsed


def _resolve_local_module(
    module_name: str, package_source_root: Path
) -> tuple[ResolvedModule, ast.Module] | None:
    source_path = _resolve_local_module_path(module_name, package_source_root)
    if source_path is None:
        return None
    source_bytes = source_path.read_bytes()
    try:
        syntax = ast.parse(source_bytes, filename=str(source_path))
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ValueError(
            f"local dependency {module_name!r} is not valid Python"
        ) from error
    return (
        ResolvedModule(
            module_name=module_name,
            source_path=source_path,
            repository_path=(
                Path("src/hyperspectrum") / source_path.relative_to(package_source_root)
            ).as_posix(),
            source_bytes=source_bytes,
        ),
        syntax,
    )


def _resolve_local_module_path(
    module_name: str, package_source_root: Path
) -> Path | None:
    spec = _module_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    source_path = Path(spec.origin).resolve()
    if source_path.suffix != ".py" or not source_path.is_relative_to(
        package_source_root
    ):
        return None
    return source_path


def _package_source_root() -> Path | None:
    spec = _module_spec("hyperspectrum")
    if spec is None or spec.submodule_search_locations is None:
        return None
    locations = tuple(spec.submodule_search_locations)
    if len(locations) != 1:
        return None
    return Path(locations[0]).resolve()


def _local_import_candidates(module_name: str, syntax: ast.Module) -> tuple[str, ...]:
    package_name = module_name.rpartition(".")[0]
    candidates: set[str] = set()
    for node in ast.walk(syntax):
        if isinstance(node, ast.Import):
            candidates.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            relative_name = "." * node.level + (node.module or "")
            try:
                imported_name = (
                    resolve_name(relative_name, package_name)
                    if node.level
                    else relative_name
                )
            except (ImportError, ValueError):
                continue
            candidates.add(imported_name)
            candidates.update(
                f"{imported_name}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )
    return tuple(sorted(candidates))


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
