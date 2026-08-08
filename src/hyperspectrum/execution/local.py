"""Atomic local execution for the M0 Savitzky-Golay smoke path."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np

from hyperspectrum.contracts import ArtifactRef, AxisSpec, PredictionBundle
from hyperspectrum.plugins.xas.arrays import XASSpectrum
from hyperspectrum.plugins.xas.baselines import (
    BaselineFailure,
    BaselinePrediction,
    savgol_filter,
)
from hyperspectrum.registry.models import ToolManifest

from .plan import RunPlan, canonical_digest, canonical_json_bytes


@dataclass(frozen=True, slots=True)
class MaterializedInputSet:
    """A complete canonical input selection bound to verified dataset contents."""

    data_digest: str
    spectra: tuple[XASSpectrum, ...]

    def __post_init__(self) -> None:
        if len(self.data_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.data_digest
        ):
            raise ValueError("materialized data digest must be a 64-character SHA-256")


Materializer = Callable[[tuple[str, ...]], MaterializedInputSet]


def execute_local_run(
    plan: RunPlan,
    *,
    tool: ToolManifest,
    selected_sample_ids: Sequence[str],
    materialize: Materializer,
) -> PredictionBundle:
    """Execute one complete local SavGol input set and publish it atomically."""

    if plan.dry_run:
        raise ValueError("dry-run plans cannot execute")
    if plan.backend != "local":
        raise ValueError("local executor requires the local backend")
    _require_tool_identity(plan, tool)
    selected = _validate_selection(selected_sample_ids, plan.max_samples)
    destination = plan.output_directory
    if destination.exists():
        raise FileExistsError(f"run directory already exists: {destination}")

    materialized = materialize(selected)
    _validate_materialization(plan.data_digest, selected, materialized)
    spectra = materialized.spectra
    window_length, polyorder = _savgol_parameters(plan)
    results = savgol_filter(
        spectra,
        window_length=window_length,
        polyorder=polyorder,
    )
    _validate_results(selected, results)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        bundle = _write_run(temporary, plan, selected, results)
        _fsync_directory(temporary)
        if destination.exists():
            raise FileExistsError(f"run directory already exists: {destination}")
        temporary.rename(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return bundle


def _validate_selection(
    selected_sample_ids: Sequence[str], max_samples: int
) -> tuple[str, ...]:
    selected = tuple(selected_sample_ids)
    if not selected:
        raise ValueError("selected input set must not be empty")
    if len(selected) > max_samples:
        raise ValueError("selected input set exceeds max_samples")
    if any(
        not isinstance(sample_id, str) or not sample_id.strip()
        for sample_id in selected
    ):
        raise ValueError("selected sample IDs must be non-empty strings")
    if len(selected) != len(set(selected)):
        raise ValueError("selected sample IDs must be unique")
    return selected


def _require_tool_identity(plan: RunPlan, tool: ToolManifest) -> None:
    if plan.tool_id != "savgol" or tool.id != "savgol":
        raise ValueError("M0 local execution supports only the savgol tool")
    if plan.tool_digest != tool.tool_digest:
        raise ValueError("selected tool digest does not match the immutable plan")
    expected_weight_digest = (
        "none"
        if not tool.weights.required
        else tool.weights.digest
        if tool.weights.state == "present"
        else None
    )
    if expected_weight_digest is None or plan.weight_digest != expected_weight_digest:
        raise ValueError("selected weight digest does not match the immutable plan")
    expected_model_digest = canonical_digest(
        {
            "tool_digest": tool.tool_digest,
            "parameters": plan.model_dump(mode="json")["parameters"],
        }
    )
    if plan.model_digest != expected_model_digest:
        raise ValueError(
            "model digest does not match the selected tool and configuration"
        )


def _validate_materialization(
    data_digest: str,
    selected: tuple[str, ...],
    materialized: MaterializedInputSet,
) -> None:
    if not isinstance(materialized, MaterializedInputSet):
        raise TypeError("materializer must return a MaterializedInputSet")
    if materialized.data_digest != data_digest:
        raise ValueError("materialized data digest does not match the immutable plan")
    spectra = materialized.spectra
    if len(spectra) != len(selected) or any(
        not isinstance(spectrum, XASSpectrum) for spectrum in spectra
    ):
        raise ValueError("materializer must return the complete selected input set")
    if tuple(spectrum.sample_id for spectrum in spectra) != selected:
        raise ValueError(
            "materializer must return the complete selected input set in order"
        )


def _savgol_parameters(plan: RunPlan) -> tuple[int, int]:
    window_length = plan.parameters.get("window_length", 5)
    polyorder = plan.parameters.get("polyorder", 2)
    allowed = {"window_length", "polyorder"}
    if set(plan.parameters) - allowed:
        raise ValueError("unsupported SavGol parameter")
    if (
        not isinstance(window_length, int)
        or isinstance(window_length, bool)
        or not isinstance(polyorder, int)
        or isinstance(polyorder, bool)
    ):
        raise TypeError("SavGol window_length and polyorder must be integers")
    return window_length, polyorder


def _validate_results(selected: tuple[str, ...], results: Sequence[object]) -> None:
    if len(results) != len(selected):
        raise RuntimeError("tool did not return one result per selected sample")
    result_ids = tuple(
        result.spectrum.sample_id
        if isinstance(result, BaselinePrediction)
        else result.sample_id
        if isinstance(result, BaselineFailure)
        else None
        for result in results
    )
    if result_ids != selected:
        raise RuntimeError(
            "tool results do not preserve selected sample order and identity"
        )


def _write_run(
    directory: Path,
    plan: RunPlan,
    selected: tuple[str, ...],
    results: Sequence[BaselinePrediction | BaselineFailure],
) -> PredictionBundle:
    arrays_directory = directory / "arrays"
    arrays_directory.mkdir()
    predictions: list[ArtifactRef] = []
    failures: list[dict[str, object]] = []
    for index, result in enumerate(results):
        if isinstance(result, BaselinePrediction):
            predictions.append(_write_prediction_array(arrays_directory, index, result))
        else:
            failures.append(
                {
                    "sample_id": result.sample_id,
                    "group_id": result.group_id,
                    "method": result.method,
                    "error_type": result.error_type,
                    "message": result.message,
                }
            )
    _fsync_directory(arrays_directory)

    run_id = f"run-{plan.plan_digest}"
    bundle = PredictionBundle.model_validate(
        {
            "schema_version": "hyperspectrum-prediction/v1",
            "run_id": run_id,
            "task_id": plan.task.id,
            "predictions": predictions,
            "failures": failures,
            "provenance": {
                "model_digest": plan.model_digest,
                "tool_digest": plan.tool_digest,
                "data_digest": plan.data_digest,
                "environment_digest": plan.environment_digest,
                "weight_digest": plan.weight_digest,
                "plan_digest": plan.plan_digest,
                "dataset_code": plan.dataset_code,
                "dataset_version": plan.dataset_version,
                "backend": plan.backend,
                "data_origin": plan.data_origin,
                "parameters": plan.model_dump(mode="json")["parameters"],
            },
        },
    )
    _write_json(directory / "predictions.json", bundle.model_dump(mode="json"))
    _write_json(
        directory / "run.json",
        {
            "schema_version": "hyperspectrum-run/v1",
            "run_id": run_id,
            "status": "completed_with_failures" if failures else "completed",
            "planned_sample_count": len(selected),
            "prediction_count": len(predictions),
            "failure_count": len(failures),
            "data_origin": plan.data_origin,
            "plan_digest": plan.plan_digest,
            "model_digest": plan.model_digest,
            "tool_digest": plan.tool_digest,
            "weight_digest": plan.weight_digest,
            "data_digest": plan.data_digest,
            "environment_digest": plan.environment_digest,
        },
    )
    return bundle


def _write_prediction_array(
    arrays_directory: Path,
    index: int,
    result: BaselinePrediction,
) -> ArtifactRef:
    sample_digest = sha256(result.spectrum.sample_id.encode("utf-8")).hexdigest()[:16]
    filename = f"{index:06d}-{sample_digest}.npz"
    path = arrays_directory / filename
    with path.open("wb") as stream:
        np.savez(
            stream,
            sample_id=np.array(result.spectrum.sample_id),
            group_id=np.array(result.spectrum.group_id),
            energy=result.spectrum.energy,
            intensity=result.spectrum.intensity,
            energy_unit=np.array(result.spectrum.energy_unit),
        )
        stream.flush()
        os.fsync(stream.fileno())
    digest = sha256(path.read_bytes()).hexdigest()
    uri = (Path("arrays") / filename).as_posix()
    direction: Literal["increasing", "decreasing"] = (
        "increasing"
        if result.spectrum.energy[0] < result.spectrum.energy[-1]
        else "decreasing"
    )
    return ArtifactRef(
        role="denoised_signal",
        kind="dense_array",
        uri=uri,
        sha256=digest,
        axes=(
            AxisSpec(
                name="energy",
                unit=result.spectrum.energy_unit,
                direction=direction,
                values_uri=uri,
            ),
        ),
    )


def _write_json(path: Path, value: object) -> None:
    with path.open("wb") as stream:
        stream.write(canonical_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
