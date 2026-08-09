"""Atomic local execution for the M0 Savitzky-Golay smoke path."""

from __future__ import annotations

import builtins
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
from importlib.util import resolve_name
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np

from hyperspectrum.contracts import (
    ArtifactRef,
    AxisSpec,
    PredictionBundleV2,
    PredictionBundleV3,
)
from hyperspectrum.registry.models import ToolManifest

from .plan import (
    ResolvedEntrypoint,
    RunPlanType,
    RunPlanV2,
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
    spectrum_type: type[Any]
    isolated_module_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _InferenceSource:
    energy: Any
    noisy: Any
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    energy_unit: str


def execute_local_run(
    plan: RunPlanType,
    *,
    tool: ToolManifest,
    selected_sample_ids: Sequence[str],
    source_npz: Path,
) -> PredictionBundleV2 | PredictionBundleV3:
    """Execute one complete local SavGol input set and publish it atomically."""

    if plan.dry_run:
        raise ValueError("dry-run plans cannot execute")
    if plan.backend != "local":
        raise ValueError("local executor requires the local backend")
    selected = _validate_selection(selected_sample_ids, plan.max_samples)
    if selected != plan.selected_sample_ids:
        raise ValueError("supplied sample selection does not match the run plan")
    destination = plan.output_directory
    if destination.exists():
        raise FileExistsError(f"run directory already exists: {destination}")
    if current_environment_digest() != plan.environment_digest:
        raise ValueError(
            "execution environment digest does not match the immutable plan"
        )
    source_bytes = source_npz.read_bytes()
    source_digest = sha256(source_bytes).hexdigest()
    expected_asset_digest = (
        plan.data_digest
        if isinstance(plan, RunPlanV2)
        else plan.benchmark_asset_digest
    )
    if source_digest != expected_asset_digest:
        raise ValueError(
            "source NPZ benchmark asset digest does not match the immutable plan"
        )
    inference_source = _load_inference_source(
        source_bytes,
        selected,
        require_canonical_benchmark=not isinstance(plan, RunPlanV2),
    )
    resolved_entrypoint = _verify_savgol_identity(plan, tool)
    resolved_tool = _load_savgol_callable(resolved_entrypoint)
    try:
        spectra = _load_selected_spectra(
            inference_source, selected, resolved_tool.spectrum_type
        )
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
    finally:
        for module_name in resolved_tool.isolated_module_names:
            sys.modules.pop(module_name, None)


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


def _verify_savgol_identity(
    plan: RunPlanType, tool: ToolManifest
) -> ResolvedEntrypoint:
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
    sources = {module.module_name: module for module in resolved.modules}
    execution_id = uuid4().hex
    module_cache: dict[str, ModuleType] = {}
    isolated_names: list[str] = []
    original_import = builtins.__import__

    def load_module(module_name: str) -> ModuleType:
        if module_name in module_cache:
            return module_cache[module_name]
        source = sources[module_name]
        isolated_name = f"{module_name}__hyperspectrum_run_{execution_id}"
        module = ModuleType(isolated_name)
        module.__file__ = str(source.source_path)
        module.__package__ = module_name.rpartition(".")[0]
        module.__dict__["__builtins__"] = {
            **vars(builtins),
            "__import__": verified_import,
        }
        module_cache[module_name] = module
        isolated_names.append(isolated_name)
        sys.modules[isolated_name] = module
        code = compile(source.source_bytes, str(source.source_path), "exec")
        exec(code, module.__dict__)  # noqa: S102 - digest-verified source boundary
        return module

    def verified_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        package_name = str((globals or {}).get("__package__", ""))
        imported_name = (
            resolve_name("." * level + name, package_name) if level else name
        )
        if imported_name not in sources:
            return original_import(name, globals, locals, fromlist, level)
        imported = load_module(imported_name)
        for item in fromlist:
            child_name = f"{imported_name}.{item}"
            if child_name in sources:
                setattr(imported, item, load_module(child_name))
        if fromlist:
            return imported
        raise ImportError(
            "verified local modules must be imported with an explicit from-list"
        )

    try:
        module = load_module(resolved.module_name)
        runner = getattr(module, resolved.object_name, None)
        if not callable(runner):
            raise TypeError("resolved SavGol entrypoint is not callable")
        prediction_type = getattr(module, "BaselinePrediction", None)
        failure_type = getattr(module, "BaselineFailure", None)
        spectrum_type = getattr(module, "XASSpectrum", None)
        if (
            not isinstance(prediction_type, type)
            or not isinstance(failure_type, type)
            or not isinstance(spectrum_type, type)
        ):
            raise TypeError("SavGol source does not define its result types")
        return _ResolvedSavGol(
            runner=cast(SavGolCallable, runner),
            prediction_type=prediction_type,
            failure_type=failure_type,
            spectrum_type=spectrum_type,
            isolated_module_names=tuple(isolated_names),
        )
    except BaseException:
        for isolated_name in isolated_names:
            sys.modules.pop(isolated_name, None)
        raise


def _load_selected_spectra(
    source: _InferenceSource,
    selected: tuple[str, ...],
    spectrum_type: type[Any],
) -> tuple[object, ...]:
    index_by_id = {
        sample_id: index for index, sample_id in enumerate(source.sample_ids)
    }
    return tuple(
        spectrum_type(
            sample_id=sample_id,
            group_id=source.group_ids[index_by_id[sample_id]],
            energy=source.energy[index_by_id[sample_id]],
            intensity=source.noisy[index_by_id[sample_id]],
            energy_unit=source.energy_unit,
        )
        for sample_id in selected
    )


def _load_inference_source(
    source_bytes: bytes,
    selected: tuple[str, ...],
    *,
    require_canonical_benchmark: bool,
) -> _InferenceSource:
    """Fully validate and detach noisy inference data before tool code loads."""

    inference_only_keys = {
        "energy",
        "noisy",
        "sample_ids",
        "group_ids",
        "energy_unit",
    }
    benchmark_keys = inference_only_keys | {
        "experiments",
        "i0_counts",
        "ketek_counts",
        "noisy_i0_counts",
        "noisy_ketek_counts",
        "pseudo_clean",
        "sample_seeds",
        "source_energy",
        "source_paths",
        "source_sha256",
        "splits",
    }
    with np.load(io.BytesIO(source_bytes), allow_pickle=False) as source:
        observed = set(source.files)
        is_canonical_benchmark = observed == benchmark_keys and len(
            source.files
        ) == len(benchmark_keys)
        is_inference_only = observed == inference_only_keys and len(
            source.files
        ) == len(inference_only_keys)
        if require_canonical_benchmark and not is_canonical_benchmark:
            raise ValueError("v3 execution requires the canonical benchmark NPZ")
        if not is_canonical_benchmark and not is_inference_only:
            missing = sorted(inference_only_keys - observed)
            extra = sorted(observed - inference_only_keys)
            raise ValueError(
                "inference-only or canonical benchmark NPZ keys must match exactly; "
                f"missing={missing}, extra={extra}"
            )
        energy = np.array(source["energy"], copy=True)
        noisy = np.array(source["noisy"], copy=True)
        sample_id_values = np.array(source["sample_ids"], copy=True)
        group_id_values = np.array(source["group_ids"], copy=True)
        energy_unit_value = np.array(source["energy_unit"], copy=True)

    for name, array in (("energy", energy), ("noisy", noisy)):
        if array.dtype.fields is not None:
            raise ValueError(f"source NPZ {name} must not use a structured dtype")
        if array.dtype.kind not in "fiu":
            raise ValueError(f"source NPZ {name} must be a real numeric array")
    energy = _canonical_float64(energy, name="energy")
    noisy = _canonical_float64(noisy, name="noisy")
    for name, array in (
        ("sample_ids", sample_id_values),
        ("group_ids", group_id_values),
    ):
        if array.dtype.fields is not None or array.dtype.kind != "U":
            raise ValueError(f"source NPZ {name} must be a plain string array")
    if (
        energy_unit_value.dtype.fields is not None
        or energy_unit_value.dtype.kind != "U"
        or energy_unit_value.shape != ()
    ):
        raise ValueError("source NPZ energy_unit must be one plain string")

    sample_ids = tuple(str(value) for value in sample_id_values)
    group_ids = tuple(str(value) for value in group_id_values)
    energy_unit = str(energy_unit_value)
    if is_canonical_benchmark and energy.ndim == 1 and noisy.ndim == 2:
        energy = np.broadcast_to(energy, noisy.shape).copy()
    malformed = (
        energy.ndim != 2
        or noisy.ndim != 2
        or energy.shape != noisy.shape
        or energy.shape[0] == 0
        or energy.shape[1] < 2
        or sample_id_values.ndim != 1
        or group_id_values.ndim != 1
        or len(sample_ids) != energy.shape[0]
        or len(group_ids) != energy.shape[0]
        or any(not value.strip() for value in sample_ids)
        or any(not value.strip() for value in group_ids)
        or len(sample_ids) != len(set(sample_ids))
        or energy_unit != "eV"
    )
    if malformed:
        raise ValueError("source NPZ contains malformed noisy-input arrays")
    differences = np.diff(energy, axis=1)
    if not np.all(
        np.logical_or(
            np.all(differences > 0.0, axis=1),
            np.all(differences < 0.0, axis=1),
        )
    ):
        raise ValueError("source NPZ energy rows must be strictly monotonic")
    index_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    if any(sample_id not in index_by_id for sample_id in selected):
        raise ValueError("source NPZ does not contain the complete selected input set")
    energy.setflags(write=False)
    noisy.setflags(write=False)
    return _InferenceSource(
        energy=energy,
        noisy=noisy,
        sample_ids=sample_ids,
        group_ids=group_ids,
        energy_unit=energy_unit,
    )


def _canonical_float64(array: Any, *, name: str) -> Any:
    """Detach real numeric input as float64 without overflow or precision loss."""

    canonical = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(canonical).all():
        raise ValueError(f"source NPZ {name} must contain only finite values")
    if array.dtype.kind in "iu":
        exact = np.equal(array.astype(object), canonical.astype(object)).all()
    else:
        exact = np.array_equal(array, canonical.astype(array.dtype))
    if not exact:
        raise ValueError(f"source NPZ {name} cannot be represented exactly as float64")
    return canonical


def _savgol_parameters(plan: RunPlanType) -> tuple[int, int]:
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
    plan: RunPlanType,
    selected: tuple[str, ...],
    results: Sequence[object],
    resolved_tool: _ResolvedSavGol,
) -> PredictionBundleV2 | PredictionBundleV3:
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
    common_provenance = {
        "model_digest": plan.model_digest,
        "tool_digest": plan.tool_digest,
        "implementation_digest": plan.implementation_digest,
        "environment_digest": plan.environment_digest,
        "weight_digest": plan.weight_digest,
        "plan_digest": plan.plan_digest,
        "plan_schema_version": plan.schema_version,
        "dataset_code": plan.dataset_code,
        "dataset_version": plan.dataset_version,
        "backend": plan.backend,
        "data_origin": plan.data_origin,
        "parameters": plan.model_dump(mode="json")["parameters"],
    }
    if isinstance(plan, RunPlanV2):
        bundle: PredictionBundleV2 | PredictionBundleV3 = (
            PredictionBundleV2.model_validate(
                {
                    "schema_version": "hyperspectrum-prediction/v2",
                    "run_id": run_id,
                    "task_id": plan.task.id,
                    "predictions": predictions,
                    "failures": failures,
                    "provenance": {
                        **common_provenance,
                        "data_digest": plan.data_digest,
                    },
                }
            )
        )
        run_identity: dict[str, object] = {"data_digest": plan.data_digest}
        run_schema_version = "hyperspectrum-run/v1"
    else:
        bundle = PredictionBundleV3.model_validate(
            {
                "schema_version": "hyperspectrum-prediction/v3",
                "run_id": run_id,
                "task_id": plan.task.id,
                "predictions": predictions,
                "failures": failures,
                "provenance": {
                    **common_provenance,
                    "source_dataset_digest": plan.source_dataset_digest,
                    "source_content_manifest_digest": (
                        plan.source_content_manifest_digest
                    ),
                    "benchmark_asset_digest": plan.benchmark_asset_digest,
                },
            }
        )
        run_identity = {
            "source_dataset_digest": plan.source_dataset_digest,
            "source_content_manifest_digest": (
                plan.source_content_manifest_digest
            ),
            "benchmark_asset_digest": plan.benchmark_asset_digest,
        }
        run_schema_version = "hyperspectrum-run/v2"
    _write_json(directory / "predictions.json", bundle.model_dump(mode="json"))
    _write_json(
        directory / "run.json",
        {
            "schema_version": run_schema_version,
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
            **run_identity,
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
