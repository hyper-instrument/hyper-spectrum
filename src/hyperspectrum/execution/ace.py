"""Thin ACE execution facade over HyperSpectrum's canonical XAS contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from hyperspectrum.contracts import PredictionBundleV3, TaskSpec
from hyperspectrum.contracts.json import freeze_json_mapping
from hyperspectrum.datasets import load_benchmark_asset_identity, load_denoising_pairs
from hyperspectrum.datasets.xanes_spec import BenchmarkManifest
from hyperspectrum.hyperdata.models import DatasetCandidate
from hyperspectrum.registry import ResourceBudget, ToolRegistry
from hyperspectrum.tasks.recommend import ReadinessVerdict

from .local import execute_local_run
from .plan import RunPlanV3, build_run_plan
from .tools import load_tool_by_id

_PSEUDO_CLEAN_LIMITATION = (
    "xas_pseudo_clean_frozen_measurement_not_physical_noiseless_ground_truth"
)


@dataclass(frozen=True, slots=True)
class AceXasExecutionResult:
    """A complete prediction run plus the exact files ACE may collect."""

    plan: RunPlanV3
    bundle: PredictionBundleV3
    artifact_paths: tuple[Path, ...]

    @property
    def n_samples(self) -> int:
        """Return the number of complete predictions in this successful run."""

        return len(self.bundle.predictions)


def execute_ace_xas_denoising(
    *,
    benchmark_directory: Path,
    output_directory: Path,
    task_id: str,
    tool_id: str,
    parameters: Mapping[str, object],
    expected_dose_fraction: float,
    max_samples: int | None = None,
    weight_files: Sequence[Path] = (),
) -> AceXasExecutionResult:
    """Execute one ACE XAS variant without moving domain contracts into ACE."""

    if task_id != "xas-denoising":
        raise ValueError("ACE XAS execution supports only task 'xas-denoising'")
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples < 1
    ):
        raise ValueError("max_samples must be a positive integer or None")
    manifest_path = benchmark_directory / "manifest.json"
    try:
        manifest = BenchmarkManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"benchmark manifest is invalid: {error}") from error
    if (
        isinstance(expected_dose_fraction, bool)
        or not isinstance(expected_dose_fraction, (int, float))
        or not math.isfinite(float(expected_dose_fraction))
        or not math.isclose(
            manifest.noise.dose_fraction,
            float(expected_dose_fraction),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "benchmark dose fraction does not match the requested ACE track"
        )

    benchmark_asset = load_benchmark_asset_identity(manifest_path)
    pairs = load_denoising_pairs(benchmark_directory)
    test_sample_ids = tuple(
        pair.noisy.sample_id for pair in pairs if pair.assignment.split == "test"
    )
    selected_sample_ids = (
        test_sample_ids if max_samples is None else test_sample_ids[:max_samples]
    )
    if not selected_sample_ids:
        raise ValueError("canonical benchmark test split must not be empty")

    task = TaskSpec(
        schema_version="hyperspectrum-task/v1",
        id=task_id,
        modality="xas",
        task_type="denoising",
        input_roles=("raw_signal",),
        output_kind="dense_array",
        ground_truth_roles=("clean_spectrum",),
        split_group_keys=("compound_id",),
        metrics=(),
    )
    dataset = DatasetCandidate(
        dataset_code=benchmark_asset.dataset_code,
        dataset_version=benchmark_asset.dataset_version,
        content_digest=benchmark_asset.declared_manifest_digest,
        title=manifest.dataset.id,
        description="ACE-admitted materialized XANES denoising benchmark",
        file_count=manifest.counts.source_file_count,
        parsed_file_count=manifest.counts.spectrum_candidate_count,
        formats=("SPEC", "NPZ"),
        license=None,
        evidence=freeze_json_mapping(
            {"admission": "validated_materialized_xanes_benchmark"}
        ),
    )
    verdict = ReadinessVerdict(
        dataset_code=benchmark_asset.dataset_code,
        dataset_version=benchmark_asset.dataset_version,
        content_digest=benchmark_asset.declared_manifest_digest,
        status="scoreable",
        reasons=("validated_materialized_xanes_denoising_pair",),
        candidate_tasks=("denoising",),
        ground_truth_roles=("clean_spectrum",),
        split_group_keys=("compound_id",),
        limitations=(_PSEUDO_CLEAN_LIMITATION,),
    )
    tool = load_tool_by_id(tool_id)
    availability = ToolRegistry((tool,)).availability(tool)
    weights = tuple(weight_files)
    device = parameters.get("device")
    plan = build_run_plan(
        task=task,
        dataset=dataset,
        verdict=verdict,
        benchmark_asset=benchmark_asset,
        tool=tool,
        availability=availability,
        backend="local",
        resources=ResourceBudget(
            cpu=tool.resources.cpu,
            memory_gb=tool.resources.memory_gb,
            gpu_available=isinstance(device, str) and device.startswith("cuda"),
        ),
        max_samples=(
            len(selected_sample_ids) if max_samples is None else max_samples
        ),
        selected_sample_ids=selected_sample_ids,
        output_directory=output_directory,
        dry_run=False,
        parameters=parameters,
        data_origin="real",
    )
    bundle = cast(
        PredictionBundleV3,
        execute_local_run(
            plan,
            tool=tool,
            selected_sample_ids=selected_sample_ids,
            source_npz=benchmark_directory / "benchmark.npz",
            weight_files=weights,
        ),
    )
    if bundle.failures or len(bundle.predictions) != len(selected_sample_ids):
        raise RuntimeError(
            "ACE XAS execution is incomplete: "
            f"predictions={len(bundle.predictions)}, "
            f"failures={len(bundle.failures)}, "
            f"planned={len(selected_sample_ids)}"
        )
    artifact_paths = (
        output_directory / "predictions.json",
        output_directory / "run.json",
        *(output_directory / prediction.uri for prediction in bundle.predictions),
    )
    return AceXasExecutionResult(
        plan=plan,
        bundle=bundle,
        artifact_paths=artifact_paths,
    )
