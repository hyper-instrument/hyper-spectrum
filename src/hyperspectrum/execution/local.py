"""Atomic local execution for the M0 Savitzky-Golay smoke path."""

from __future__ import annotations

import ctypes
import errno
import io
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

import numpy as np

from hyperspectrum.contracts import ArtifactRef, AxisSpec, PredictionBundle
from hyperspectrum.plugins.xas.arrays import XASSpectrum
from hyperspectrum.registry.models import ToolManifest

from .plan import (
    ResolvedEntrypoint,
    RunPlan,
    canonical_digest,
    canonical_json_bytes,
    current_environment_digest,
    resolve_local_entrypoint,
)

SavGolCallable = Callable[..., tuple[object, ...]]


@dataclass(frozen=True, slots=True)
class _ResolvedSavGol:
    runner: SavGolCallable
    prediction_type: type[object]
    failure_type: type[object]


def execute_local_run(
    plan: RunPlan,
    *,
    tool: ToolManifest,
    selected_sample_ids: Sequence[str],
    source_npz: Path,
) -> PredictionBundle:
    """Execute one complete local SavGol input set and publish it atomically."""

    if plan.dry_run:
        raise ValueError("dry-run plans cannot execute")
    if plan.backend != "local":
        raise ValueError("local executor requires the local backend")
    selected = _validate_selection(selected_sample_ids, plan.max_samples)
    destination = plan.output_directory
    if destination.exists():
        raise FileExistsError(f"run directory already exists: {destination}")
    if current_environment_digest() != plan.environment_digest:
        raise ValueError(
            "execution environment digest does not match the immutable plan"
        )
    resolved_entrypoint = _verify_savgol_identity(plan, tool)

    source_bytes = source_npz.read_bytes()
    source_digest = sha256(source_bytes).hexdigest()
    if source_digest != plan.data_digest:
        raise ValueError("source NPZ data digest does not match the immutable plan")
    spectra = _load_selected_spectra(source_bytes, selected)
    resolved_tool = _load_savgol_callable(resolved_entrypoint)
    window_length, polyorder = _savgol_parameters(plan)
    results = resolved_tool.runner(
        spectra,
        window_length=window_length,
        polyorder=polyorder,
    )
    _validate_results(selected, results, resolved_tool)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        bundle = _write_run(temporary, plan, selected, results, resolved_tool)
        _fsync_directory(temporary)
        _atomic_rename_noreplace(temporary, destination)
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


def _verify_savgol_identity(plan: RunPlan, tool: ToolManifest) -> ResolvedEntrypoint:
    if plan.tool_id != "savgol" or tool.id != "savgol":
        raise ValueError("M0 local execution supports only the savgol tool")
    if tool.entrypoint != "hyperspectrum.plugins.xas.baselines:savgol_filter":
        raise ValueError("M0 local execution requires the canonical SavGol entrypoint")
    if plan.tool_digest != tool.tool_digest:
        raise ValueError("selected tool digest does not match the immutable plan")
    if tool.weights.required or plan.weight_digest != "none":
        raise ValueError("M0 SavGol execution requires no weights")
    resolved = resolve_local_entrypoint(tool)
    if resolved.implementation_digest != plan.implementation_digest:
        raise ValueError("implementation digest does not match the immutable plan")
    expected_model_digest = canonical_digest(
        {
            "tool_digest": tool.tool_digest,
            "implementation_digest": resolved.implementation_digest,
            "parameters": plan.model_dump(mode="json")["parameters"],
        }
    )
    if plan.model_digest != expected_model_digest:
        raise ValueError("model digest does not match the selected implementation")
    return resolved


def _load_savgol_callable(resolved: ResolvedEntrypoint) -> _ResolvedSavGol:
    execution_module_name = (
        f"{resolved.module_name}__hyperspectrum_run_{resolved.implementation_digest}"
    )
    module = ModuleType(execution_module_name)
    module.__file__ = str(resolved.source_path)
    module.__package__ = resolved.module_name.rpartition(".")[0]
    code = compile(resolved.source_bytes, str(resolved.source_path), "exec")
    sys.modules[execution_module_name] = module
    try:
        exec(code, module.__dict__)  # noqa: S102 - authorized, digest-verified tool boundary
    except BaseException:
        sys.modules.pop(execution_module_name, None)
        raise
    loaded = getattr(module, resolved.object_name, None)
    if not callable(loaded):
        raise TypeError("resolved SavGol entrypoint is not callable")
    prediction_type = getattr(module, "BaselinePrediction", None)
    failure_type = getattr(module, "BaselineFailure", None)
    if not isinstance(prediction_type, type) or not isinstance(failure_type, type):
        raise TypeError("SavGol source does not define its result types")
    return _ResolvedSavGol(
        runner=cast(SavGolCallable, loaded),
        prediction_type=prediction_type,
        failure_type=failure_type,
    )


def _load_selected_spectra(
    source_bytes: bytes, selected: tuple[str, ...]
) -> tuple[XASSpectrum, ...]:
    required = {"energy", "noisy", "sample_ids", "group_ids", "energy_unit"}
    with np.load(io.BytesIO(source_bytes), allow_pickle=False) as source:
        if not required.issubset(source.files):
            raise ValueError("source NPZ is missing required noisy-input arrays")
        energy = np.asarray(source["energy"])
        noisy = np.asarray(source["noisy"])
        sample_ids = tuple(str(value) for value in source["sample_ids"])
        group_ids = tuple(str(value) for value in source["group_ids"])
        energy_unit = str(source["energy_unit"])
        if (
            energy.ndim != 2
            or noisy.ndim != 2
            or energy.shape != noisy.shape
            or len(sample_ids) != energy.shape[0]
            or len(group_ids) != energy.shape[0]
            or len(sample_ids) != len(set(sample_ids))
        ):
            raise ValueError("source NPZ contains malformed noisy-input arrays")
        index_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        if any(sample_id not in index_by_id for sample_id in selected):
            raise ValueError(
                "source NPZ does not contain the complete selected input set"
            )
        return tuple(
            XASSpectrum(
                sample_id=sample_id,
                group_id=group_ids[index_by_id[sample_id]],
                energy=energy[index_by_id[sample_id]],
                intensity=noisy[index_by_id[sample_id]],
                energy_unit=energy_unit,  # type: ignore[arg-type]
            )
            for sample_id in selected
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


def _validate_results(
    selected: tuple[str, ...],
    results: Sequence[object],
    resolved_tool: _ResolvedSavGol,
) -> None:
    if len(results) != len(selected):
        raise RuntimeError("tool did not return one result per selected sample")
    result_ids = tuple(
        cast(Any, result).spectrum.sample_id
        if isinstance(result, resolved_tool.prediction_type)
        else cast(Any, result).sample_id
        if isinstance(result, resolved_tool.failure_type)
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
    results: Sequence[object],
    resolved_tool: _ResolvedSavGol,
) -> PredictionBundle:
    arrays_directory = directory / "arrays"
    arrays_directory.mkdir()
    predictions: list[ArtifactRef] = []
    failures: list[dict[str, object]] = []
    for index, result in enumerate(results):
        typed_result = cast(Any, result)
        if isinstance(result, resolved_tool.prediction_type):
            predictions.append(
                _write_prediction_array(arrays_directory, index, typed_result)
            )
        else:
            failures.append(
                {
                    "sample_id": typed_result.sample_id,
                    "group_id": typed_result.group_id,
                    "method": typed_result.method,
                    "error_type": typed_result.error_type,
                    "message": typed_result.message,
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
                "implementation_digest": plan.implementation_digest,
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
            "implementation_digest": plan.implementation_digest,
            "weight_digest": plan.weight_digest,
            "data_digest": plan.data_digest,
            "environment_digest": plan.environment_digest,
        },
    )
    return bundle


def _write_prediction_array(
    arrays_directory: Path,
    index: int,
    result: Any,
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


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing name."""

    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        try:
            renamex_np = library.renamex_np
        except AttributeError as error:
            raise RuntimeError(
                "renamex_np is unavailable on this macOS host"
            ) from error
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(source), os.fsencode(destination), 0x00000004)
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = library.renameat2
        except AttributeError as error:
            raise RuntimeError("renameat2 is unavailable on this Linux host") from error
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            0x00000001,
        )
    elif os.name == "nt":
        os.rename(source, destination)
        return
    else:
        raise RuntimeError("atomic no-replace directory rename is unsupported")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))
