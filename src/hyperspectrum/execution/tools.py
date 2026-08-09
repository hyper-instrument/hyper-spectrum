"""The registered XAS tools an execution facade may run, by id.

Shared by the two ACE-facing facades rather than duplicated in each: both must
resolve ``savgol`` to the same manifest bytes, and two lookup tables is how the
self-scored and inference-only paths would come to run different numerics under
the same tool name.
"""

from __future__ import annotations

from importlib.resources import files

import yaml

from hyperspectrum.registry import load_tool_manifest
from hyperspectrum.registry.models import ToolManifest

_TOOL_RESOURCES = {
    "savgol": "tools/xas/savgol/tool.yaml",
    "xasdenoise": "tools/xas/xasdenoise/tool.yaml",
}


def load_tool_by_id(tool_id: str) -> ToolManifest:
    """Load one packaged tool manifest, refusing any id that is not registered."""

    resource_name = _TOOL_RESOURCES.get(tool_id)
    if resource_name is None:
        raise ValueError(f"unknown tool: {tool_id}")
    resource = files("hyperspectrum.resources").joinpath(resource_name)
    raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("tool manifest must be a mapping")
    return load_tool_manifest(raw)


__all__ = ["load_tool_by_id"]
