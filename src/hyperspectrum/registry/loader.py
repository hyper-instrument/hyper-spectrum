"""Load and match schema-validated, declarative tool manifests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import lru_cache
from importlib.machinery import PathFinder
from pathlib import Path
from shutil import which

import yaml
from jsonschema import Draft202012Validator

from .models import AvailabilityReason, ToolAvailability, ToolManifest, ToolMatchRequest


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    """Load the versioned public schema once and validate its own integrity."""
    schema_path = Path(__file__).resolve().parents[3] / "schemas/hyperspectrum-tool-v1.schema.json"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
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
        _schema_validator().iter_errors(manifest.model_dump(mode="json", exclude_none=True)), None
    )
    if schema_error is not None:
        raise ValueError(f"manifest failed its published JSON Schema: {schema_error.message}")
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
        self._availability_resolver = availability_resolver or _safe_default_availability

    def availability(self, tool: ToolManifest) -> ToolAvailability:
        """Return safe, machine-readable availability without importing tool code."""
        return self._availability_resolver(tool)

    def match(self, request: ToolMatchRequest) -> tuple[ToolManifest, ...]:
        """Return tools only when every compatibility and policy constraint passes."""
        return tuple(
            tool
            for tool in self.tools
            if _matches(tool, request) and self.availability(tool).available
        )


def _safe_default_availability(tool: ToolManifest) -> ToolAvailability:
    """Resolve paths and command/module specs without importing or executing tool code."""
    reasons: list[AvailabilityReason] = []
    if not _entrypoint_resolvable(tool.entrypoint):
        reasons.append("entrypoint-unresolvable")
    reasons.extend(_verify_unavailability_reasons(tool.verify))
    if tool.weights.required:
        if tool.weights.state == "required-missing":
            reasons.append("weights-required-missing")
        else:
            reasons.append("weights-unverified")
    if reasons:
        return ToolAvailability(available=False, reasons=tuple(reasons))
    return ToolAvailability(available=True)


def _entrypoint_resolvable(entrypoint: str) -> bool:
    if ":" in entrypoint:
        module_name, _, object_name = entrypoint.partition(":")
        return bool(object_name) and _module_resolvable(module_name)
    path = Path(entrypoint)
    return not path.is_absolute() and ".." not in path.parts and (_repository_root() / path).is_file()


def _verify_unavailability_reasons(argv: tuple[str, ...]) -> tuple[AvailabilityReason, ...]:
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
    parts = module_name.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        return False
    spec = PathFinder.find_spec(parts[0])
    for part in parts[1:]:
        if spec is None or spec.submodule_search_locations is None:
            return False
        spec = PathFinder.find_spec(part, spec.submodule_search_locations)
    return spec is not None


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _matches(tool: ToolManifest, request: ToolMatchRequest) -> bool:
    if request.modality not in tool.modalities or request.task not in tool.tasks:
        return False
    if {artifact.role for artifact in tool.inputs} != set(request.input_roles):
        return False
    if {artifact.role for artifact in tool.outputs} != set(request.output_roles):
        return False
    if tool.license not in request.license_policy.allowed_licenses:
        return False
    if tool.distribution not in request.license_policy.allowed_distributions:
        return False
    if tool.weights.state != request.weights_state:
        return False
    if tool.resources.cpu > request.resources.cpu:
        return False
    if tool.resources.memory_gb > request.resources.memory_gb:
        return False
    return tool.resources.gpu != "required" or request.resources.gpu_available
