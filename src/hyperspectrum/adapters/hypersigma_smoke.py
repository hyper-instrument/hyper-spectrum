"""Explicit MAT-to-contract smoke runner for pinned HyperSIGMA assets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from hyperspectrum.adapters.hypersigma import (
    HyperSIGMADenoiseAdapter,
    WeightVariant,
)
from hyperspectrum.denoising import (
    DenoisingModel,
    DenoisingPair,
    SpectrumAxis,
    SpectrumSample,
    SplitEntry,
    SplitManifest,
    evaluate_denoising_suite,
)
from hyperspectrum.plugins.hsi.arrays import resolve_hsi_layout

FixtureKind = Literal["synthetic", "research-use-only"]


def _real_cube(value: object, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional cube")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(f"{name} must be numeric")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def load_mat_pair(
    path: Path,
    *,
    input_key: str,
    target_key: str,
    axis_names: tuple[str, str, str],
    axis_units: tuple[str, str, str],
    signal_unit: str,
    fixture_kind: FixtureKind,
    sample_id: str,
    group_id: str,
) -> DenoisingPair:
    """Load two explicitly named real cubes without inferring keys or axes."""

    payload = loadmat(path)
    if input_key not in payload or target_key not in payload:
        raise ValueError(f"MAT keys must include {input_key!r} and {target_key!r}")
    noisy_values = _real_cube(payload[input_key], name=input_key)
    clean_values = _real_cube(payload[target_key], name=target_key)
    if noisy_values.shape != clean_values.shape:
        raise ValueError("MAT input and target shapes must match")
    resolve_hsi_layout(axis_names, axis_units, noisy_values.shape)
    axes = tuple(
        SpectrumAxis(
            name=name,
            unit=unit,
            direction="increasing",
            values=np.arange(length, dtype=np.float64),
        )
        for name, unit, length in zip(
            axis_names, axis_units, noisy_values.shape, strict=True
        )
    )

    def make_sample(values: NDArray[np.float64]) -> SpectrumSample:
        return SpectrumSample(
            sample_id=sample_id,
            group_id=group_id,
            modality="hyperspectral",
            representation="dense",
            axes=axes,
            signal=values,
            valid_mask=np.ones(values.shape, dtype=np.bool_),
            signal_unit=signal_unit,
            metadata={
                "fixture_kind": fixture_kind,
                "source_filename": path.name,
                "input_key": input_key,
                "target_key": target_key,
            },
        )

    noisy = make_sample(noisy_values)
    clean = make_sample(clean_values)
    manifest = SplitManifest(
        schema_version="hyperspectrum-split-manifest/v1",
        entries=(SplitEntry(sample_id, group_id, "test"),),
    )
    return DenoisingPair(
        noisy=noisy,
        clean=clean,
        assignment=manifest.assignment_for(sample_id, group_id),
    )


def run_smoke(
    *,
    source_root: Path,
    weight_path: Path,
    variant: WeightVariant,
    mat_path: Path,
    input_key: str,
    target_key: str,
    axis_names: tuple[str, str, str],
    axis_units: tuple[str, str, str],
    signal_unit: str,
    fixture_kind: FixtureKind,
    sample_id: str,
    group_id: str,
    device: str,
    _model: DenoisingModel | None = None,
) -> Mapping[str, object]:
    """Run one diagnostic sample through the shared denoising evaluator."""

    pair = load_mat_pair(
        mat_path,
        input_key=input_key,
        target_key=target_key,
        axis_names=axis_names,
        axis_units=axis_units,
        signal_unit=signal_unit,
        fixture_kind=fixture_kind,
        sample_id=sample_id,
        group_id=group_id,
    )
    model = (
        _model
        if _model is not None
        else HyperSIGMADenoiseAdapter(
            source_root,
            weight_path,
            variant=variant,
            device=device,
        )
    )
    report = evaluate_denoising_suite(model, (pair,))
    return {
        "schema_version": "hyperspectrum-hypersigma-smoke/v1",
        "diagnostic_only": True,
        "fixture_kind": fixture_kind,
        "variant": variant,
        "sample_id": pair.noisy.sample_id,
        "axis_names": list(axis_names),
        "axis_units": list(axis_units),
        "device": device,
        "evaluation": report.to_dict(),
    }


def _three_nonblank(value: str, *, name: str) -> tuple[str, str, str]:
    parts = value.split(",")
    if len(parts) != 3 or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(f"{name} must contain three non-empty values")
    return parts[0], parts[1], parts[2]


def _axis_order(value: str) -> tuple[str, str, str]:
    parts = _three_nonblank(value, name="axis order")
    spectral = {part for part in parts if part in {"band", "wavelength"}}
    if len(spectral) != 1 or set(parts) != {*spectral, "y", "x"}:
        raise argparse.ArgumentTypeError(
            "axis order must contain exactly band or wavelength, y, and x"
        )
    return parts


def _axis_units(value: str) -> tuple[str, str, str]:
    return _three_nonblank(value, name="axis units")


def _nonblank(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must be non-empty")
    return value


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one diagnostic-only HyperSIGMA research-data smoke."
    )
    parser.add_argument("--source-root", type=_absolute_path, required=True)
    parser.add_argument("--weight-path", type=_absolute_path, required=True)
    parser.add_argument(
        "--variant", choices=("gaussian", "complex"), required=True
    )
    parser.add_argument("--mat-path", type=_absolute_path, required=True)
    parser.add_argument("--input-key", type=_nonblank, required=True)
    parser.add_argument("--target-key", type=_nonblank, required=True)
    parser.add_argument("--axis-order", type=_axis_order, required=True)
    parser.add_argument("--axis-units", type=_axis_units, required=True)
    parser.add_argument("--signal-unit", type=_nonblank, required=True)
    parser.add_argument("--sample-id", type=_nonblank, required=True)
    parser.add_argument("--group-id", type=_nonblank, required=True)
    parser.add_argument("--device", type=_nonblank, required=True)
    return parser


def _diagnostic_succeeded(result: Mapping[str, object]) -> bool:
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return False
    coverage = evaluation.get("coverage")
    if not isinstance(coverage, Mapping):
        return False
    return (
        coverage.get("value") == 1.0
        and coverage.get("evaluated_count") == 1
        and coverage.get("skipped_count") == 0
        and coverage.get("failed_count") == 0
        and coverage.get("total_count") == 1
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse explicit physical declarations and emit exactly one JSON result."""

    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        resolve_hsi_layout(arguments.axis_order, arguments.axis_units, (1, 1, 1))
    except ValueError as error:
        parser.error(str(error))
    result = run_smoke(
        source_root=arguments.source_root,
        weight_path=arguments.weight_path,
        variant=cast(WeightVariant, arguments.variant),
        mat_path=arguments.mat_path,
        input_key=arguments.input_key,
        target_key=arguments.target_key,
        axis_names=arguments.axis_order,
        axis_units=arguments.axis_units,
        signal_unit=arguments.signal_unit,
        fixture_kind="research-use-only",
        sample_id=arguments.sample_id,
        group_id=arguments.group_id,
        device=arguments.device,
    )
    print(
        json.dumps(
            result,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    return 0 if _diagnostic_succeeded(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
