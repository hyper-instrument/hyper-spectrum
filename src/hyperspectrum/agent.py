"""Reusable, JSON-safe services for scientific agents and the public CLI."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from hyperspectrum.contracts import TaskSpec
from hyperspectrum.execution import RunPlan, build_run_plan, execute_local_run
from hyperspectrum.hyperdata import (
    HydAuthenticationError,
    HydClientNotFoundError,
    HydCommandError,
    HydGateway,
    HydTransportError,
    HydUnsupportedClientError,
    HydUnsupportedJsonError,
)
from hyperspectrum.hyperdata.discovery import XAS_QUERIES, discover_xas
from hyperspectrum.hyperdata.models import DatasetCandidate
from hyperspectrum.registry import ResourceBudget, ToolRegistry, load_tool_manifest
from hyperspectrum.registry.models import ToolManifest
from hyperspectrum.tasks import (
    ReadinessVerdict,
    profile_xas_candidate,
    recommend_xas_tasks,
)

JsonObject = dict[str, Any]
_ROOT = Path(__file__).resolve().parents[2]
_TOOL_FILES = {
    "savgol": _ROOT / "tools/xas/savgol/tool.yaml",
    "xasdenoise": _ROOT / "tools/xas/xasdenoise/tool.yaml",
}


@dataclass(frozen=True, slots=True)
class ServiceResponse:
    """One service result plus non-fatal warnings for stderr."""

    result: object
    warnings: tuple[str, ...] = ()


class AgentServiceError(RuntimeError):
    """Stable error boundary shared by non-CLI and CLI agent callers."""

    error_code = "execution_failure"
    exit_code = 5

    def __init__(self, message: str, *, result: object = None) -> None:
        super().__init__(message)
        self.result = result


class AgentRequestError(AgentServiceError):
    """The request is invalid or fails a readiness gate."""

    error_code = "invalid_or_not_ready"
    exit_code = 2


class AgentAuthError(AgentServiceError):
    """Authentication or a downstream connection is unavailable."""

    error_code = "auth_or_connection"
    exit_code = 3


class AgentMissingAssetError(AgentServiceError):
    """A required local asset or declared tool is absent."""

    error_code = "missing_asset_or_tool"
    exit_code = 4


class AgentExecutionError(AgentServiceError):
    """An admitted execution failed."""


def doctor() -> ServiceResponse:
    """Inspect executable and repository readiness without reading credentials."""

    checks: dict[str, dict[str, object]] = {
        "python": {
            "ready": sys.version_info >= (3, 10),
            "version": sys.version.split()[0],
        },
        "hyperdata_cli": {"ready": shutil.which("hyd") is not None, "binary": "hyd"},
        "task_contract": {
            "ready": (_ROOT / "src/hyperspectrum/contracts/task.py").is_file()
        },
        "tool_manifests": {
            "ready": all(path.is_file() for path in _TOOL_FILES.values()),
            "ids": sorted(_TOOL_FILES),
        },
    }
    ready = all(bool(check["ready"]) for check in checks.values())
    result = {"ready": ready, "checks": checks}
    if not ready:
        raise AgentRequestError("one or more readiness checks failed", result=result)
    return ServiceResponse(result=result)


def discover_data(modality: str, profile: str) -> ServiceResponse:
    """Discover catalog-only XAS evidence through the public HyperData gateway."""

    if modality.strip().casefold() != "xas":
        raise AgentRequestError("M0 discovery supports only modality 'xas'")
    if not profile.strip():
        raise AgentRequestError("profile must be a non-blank name")
    try:
        candidates = discover_xas(HydGateway(profile=profile))
    except HydAuthenticationError as error:
        raise AgentAuthError(str(error)) from error
    except (HydTransportError, HydCommandError) as error:
        raise AgentAuthError(str(error)) from error
    except (
        HydClientNotFoundError,
        HydUnsupportedClientError,
        HydUnsupportedJsonError,
    ) as error:
        raise AgentMissingAssetError(str(error)) from error
    return ServiceResponse(
        result={
            "modality": "xas",
            "profile": profile,
            "queries": list(XAS_QUERIES),
            "candidates": [
                candidate.model_dump(mode="json") for candidate in candidates
            ],
        }
    )


def recommend_task(candidate_file: Path) -> ServiceResponse:
    """Recommend scoreability from one explicit candidate evidence artifact."""

    candidate = _read_model(candidate_file, DatasetCandidate, "candidate file")
    verdicts = recommend_xas_tasks(profile_xas_candidate(candidate))
    return ServiceResponse(
        result={
            "candidate": candidate.model_dump(mode="json"),
            "verdicts": [verdict.model_dump(mode="json") for verdict in verdicts],
        }
    )


def match_tools(task: str) -> ServiceResponse:
    """Match built-in manifests and retain machine-readable rejection evidence."""

    if task != "xas-denoising":
        raise AgentRequestError("M0 tool matching supports only task 'xas-denoising'")
    tools = tuple(_load_tool(tool_id) for tool_id in sorted(_TOOL_FILES))
    registry = ToolRegistry(tools)
    matches: list[JsonObject] = []
    blocked: list[JsonObject] = []
    for tool in tools:
        if not _supports_xas_denoising(tool):
            continue
        availability = registry.availability(tool)
        record = {
            "id": tool.id,
            "manifest": tool.model_dump(mode="json"),
            "tool_digest": tool.tool_digest,
            "availability": availability.model_dump(mode="json"),
        }
        if availability.available:
            matches.append(record)
        else:
            blocked.append({**record, "reasons": list(availability.reasons)})
    return ServiceResponse(
        result={"task": task, "matches": matches, "blocked": blocked}
    )


def plan_run(
    *,
    task_file: Path,
    candidate_file: Path,
    verdict_file: Path,
    tool_id: str,
    output_directory: Path,
    max_samples: int,
    dry_run: bool,
) -> ServiceResponse:
    """Build one immutable local plan from explicit public contract files."""

    task = _read_model(task_file, TaskSpec, "task file")
    candidate = _read_model(candidate_file, DatasetCandidate, "candidate file")
    verdict = _read_model(verdict_file, ReadinessVerdict, "verdict file")
    tool = _load_tool(tool_id)
    registry = ToolRegistry((tool,))
    try:
        plan = build_run_plan(
            task=task,
            dataset=candidate,
            verdict=verdict,
            tool=tool,
            availability=registry.availability(tool),
            backend="local",
            resources=ResourceBudget(cpu=1, memory_gb=1.0, gpu_available=False),
            max_samples=max_samples,
            output_directory=output_directory,
            dry_run=dry_run,
            parameters={"window_length": 5, "polyorder": 2},
            data_origin="real",
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise AgentRequestError(str(error)) from error
    return ServiceResponse(
        result={"plan": plan.model_dump(mode="json"), "plan_digest": plan.plan_digest}
    )


def run_local(
    *, plan_file: Path, source_npz: Path, sample_ids: tuple[str, ...]
) -> ServiceResponse:
    """Execute an admitted plan through the public atomic local executor."""

    plan = _read_model(plan_file, RunPlan, "plan file")
    tool = _load_tool(plan.tool_id)
    if not source_npz.is_file():
        raise AgentMissingAssetError(f"source NPZ does not exist: {source_npz}")
    try:
        bundle = execute_local_run(
            plan,
            tool=tool,
            selected_sample_ids=sample_ids,
            source_npz=source_npz,
        )
    except FileNotFoundError as error:
        raise AgentMissingAssetError(str(error)) from error
    except (TypeError, ValueError, FileExistsError, OSError) as error:
        raise AgentExecutionError(str(error)) from error
    return ServiceResponse(result={"prediction_bundle": bundle.model_dump(mode="json")})


def _read_model(path: Path, model: type[BaseModel], label: str) -> Any:
    if not path.is_file():
        raise AgentMissingAssetError(f"{label} does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return model.model_validate(raw)
    except json.JSONDecodeError as error:
        raise AgentRequestError(f"{label} is not valid JSON: {error.msg}") from error
    except ValidationError as error:
        raise AgentRequestError(f"{label} violates its contract: {error}") from error
    except OSError as error:
        raise AgentMissingAssetError(f"cannot read {label}: {error}") from error


def _load_tool(tool_id: str) -> ToolManifest:
    path = _TOOL_FILES.get(tool_id)
    if path is None:
        raise AgentMissingAssetError(f"unknown tool: {tool_id}")
    if not path.is_file():
        raise AgentMissingAssetError(f"tool manifest does not exist: {path}")
    try:
        return load_tool_manifest(path)
    except (OSError, ValueError, ValidationError) as error:
        raise AgentMissingAssetError(
            f"cannot load tool '{tool_id}': {error}"
        ) from error


def _supports_xas_denoising(tool: ToolManifest) -> bool:
    return (
        "xas" in tool.modalities
        and "denoising" in tool.tasks
        and {artifact.role for artifact in tool.inputs} == {"raw_signal"}
        and {artifact.role for artifact in tool.outputs} == {"denoised_signal"}
    )
