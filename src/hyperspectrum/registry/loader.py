"""Load and match schema-validated, declarative tool manifests."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from functools import lru_cache
from importlib.machinery import ModuleSpec, PathFinder
from importlib.resources import files
from pathlib import Path
from shutil import which

import yaml
from jsonschema import Draft202012Validator

from .models import (
    AvailabilityReason,
    ToolAvailability,
    ToolManifest,
    ToolMatchRequest,
    ToolRejection,
)


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    """Load the versioned public schema once and validate its own integrity."""
    schema = yaml.safe_load(
        files("hyperspectrum.resources")
        .joinpath("schemas/hyperspectrum-tool-v1.schema.json")
        .read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def load_tool_manifest(source: Path | Mapping[str, object]) -> ToolManifest:
    """Parse a YAML file or mapping into a typed semantic manifest.

    Pydantic is the public error boundary. The JSON Schema is then checked against
    the normalized manifest so the distributed schema cannot drift silently from
    the runtime contract.
    """
    if isinstance(source, Path):
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    else:
        raw = dict(source)
    manifest = ToolManifest.model_validate(raw)
    schema_error = next(
        _schema_validator().iter_errors(
            manifest.model_dump(mode="json", exclude_none=True)
        ),
        None,
    )
    if schema_error is not None:
        raise ValueError(
            f"manifest failed its published JSON Schema: {schema_error.message}"
        )
    return manifest


class ToolRegistry:
    """An immutable collection whose selection method fails closed at every gate."""

    def __init__(
        self,
        tools: tuple[ToolManifest, ...],
        availability_resolver: Callable[[ToolManifest], ToolAvailability] | None = None,
    ) -> None:
        ids = tuple(tool.id for tool in tools)
        if len(ids) != len(set(ids)):
            raise ValueError("tool registry IDs must be unique")
        self.tools = tools
        self._availability_resolver = (
            availability_resolver or _safe_default_availability
        )

    def availability(self, tool: ToolManifest) -> ToolAvailability:
        """Return safe, machine-readable availability without importing tool code."""
        return self._availability_resolver(tool)

    def match(self, request: ToolMatchRequest) -> tuple[ToolManifest, ...]:
        """Return tools only when every compatibility and policy constraint passes."""
        return tuple(
            tool
            for tool in self.tools
            if not _rejection_reasons(tool, request, self.availability(tool))
        )

    def rejections(self, request: ToolMatchRequest) -> tuple[ToolRejection, ...]:
        """Explain every rejected tool using the same fail-closed match policy."""

        rejected: list[ToolRejection] = []
        for tool in self.tools:
            reasons = _rejection_reasons(tool, request, self.availability(tool))
            if reasons:
                rejected.append(ToolRejection(tool_id=tool.id, reasons=reasons))
        return tuple(rejected)


def _safe_default_availability(tool: ToolManifest) -> ToolAvailability:
    """Resolve paths and command/module specs without importing or executing tool code."""
    if tool.runtime.kind == "container":
        return ToolAvailability(available=False, reasons=("container-unverified",))

    reasons: list[AvailabilityReason] = []
    if not _entrypoint_resolvable(tool.entrypoint):
        reasons.append("entrypoint-unresolvable")
    reasons.extend(_verify_unavailability_reasons(tool.verify))
    if tool.weights.required:
        if tool.weights.state == "required-missing":
            reasons.append("weights-required-missing")
        else:
            reasons.append("weights-unverified")
    if tool.id == "xasdenoise":
        # The checkpoint training database is already pre/post-edge normalized.
        # Raw benchmark spectra lack the upstream E0, fit windows, and fitted
        # curves needed to reproduce and exactly invert that normalization.
        reasons.append("input_contract_unverified")
    if reasons:
        return ToolAvailability(available=False, reasons=tuple(reasons))
    return ToolAvailability(available=True)


def _entrypoint_resolvable(entrypoint: str) -> bool:
    if ":" in entrypoint:
        module_name, _, object_name = entrypoint.partition(":")
        return _has_static_top_level_object(_module_spec(module_name), object_name)
    return False


def _verify_unavailability_reasons(
    argv: tuple[str, ...],
) -> tuple[AvailabilityReason, ...]:
    reasons: list[AvailabilityReason] = []
    if which(argv[0]) is None:
        reasons.append("verify-command-unresolvable")
    if "-m" in argv:
        module_index = argv.index("-m") + 1
        if module_index == len(argv) or not _module_resolvable(argv[module_index]):
            reasons.append("verify-module-unresolvable")
    return tuple(reasons)


def _module_resolvable(module_name: str) -> bool:
    """Find a dotted module spec recursively without executing package imports."""
    return _module_spec(module_name) is not None


def _module_spec(module_name: str) -> ModuleSpec | None:
    """Find a dotted module spec recursively without executing package imports."""
    parts = module_name.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    spec = PathFinder.find_spec(parts[0])
    for part in parts[1:]:
        if spec is None or spec.submodule_search_locations is None:
            return None
        spec = PathFinder.find_spec(part, spec.submodule_search_locations)
    return spec


def _has_static_top_level_object(spec: ModuleSpec | None, object_name: str) -> bool:
    """Prove a local module defines an object directly, without importing its code."""
    if spec is None or not object_name.isidentifier() or spec.origin is None:
        return False
    source_path = Path(spec.origin)
    package_root = _package_source_root()
    if (
        package_root is None
        or source_path.suffix != ".py"
        or not source_path.resolve().is_relative_to(package_root)
    ):
        return False
    try:
        module = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    return any(
        _defines_top_level_object(statement, object_name) for statement in module.body
    )


def _defines_top_level_object(statement: ast.stmt, object_name: str) -> bool:
    if isinstance(statement, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)):
        return statement.name == object_name
    return False


def _package_source_root() -> Path | None:
    spec = _module_spec("hyperspectrum")
    if spec is None or spec.submodule_search_locations is None:
        return None
    locations = tuple(spec.submodule_search_locations)
    if len(locations) != 1:
        return None
    return Path(locations[0]).resolve()


def _rejection_reasons(
    tool: ToolManifest,
    request: ToolMatchRequest,
    availability: ToolAvailability,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if request.modality not in tool.modalities:
        reasons.append("modality-mismatch")
    if request.task not in tool.tasks:
        reasons.append("task-mismatch")
    if {artifact.role for artifact in tool.inputs} != set(request.input_roles):
        reasons.append("input-roles-mismatch")
    if {artifact.role for artifact in tool.outputs} != set(request.output_roles):
        reasons.append("output-roles-mismatch")
    if tool.license not in request.license_policy.allowed_licenses:
        reasons.append("license-not-allowed")
    if tool.distribution not in request.license_policy.allowed_distributions:
        reasons.append("distribution-not-allowed")
    if tool.weights.state != request.weights_state:
        reasons.append("weights-state-mismatch")
    if tool.resources.cpu > request.resources.cpu:
        reasons.append("cpu-insufficient")
    if tool.resources.memory_gb > request.resources.memory_gb:
        reasons.append("memory-insufficient")
    if tool.resources.gpu == "required" and not request.resources.gpu_available:
        reasons.append("gpu-unavailable")
    reasons.extend(availability.reasons)
    return tuple(reasons)
