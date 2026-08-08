"""Load and match schema-validated, declarative tool manifests."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from .models import ToolManifest, ToolMatchRequest


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

    def __init__(self, tools: tuple[ToolManifest, ...]) -> None:
        ids = tuple(tool.id for tool in tools)
        if len(ids) != len(set(ids)):
            raise ValueError("tool registry IDs must be unique")
        self.tools = tools

    def match(self, request: ToolMatchRequest) -> tuple[ToolManifest, ...]:
        """Return tools only when every compatibility and policy constraint passes."""
        return tuple(tool for tool in self.tools if _matches(tool, request))


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
