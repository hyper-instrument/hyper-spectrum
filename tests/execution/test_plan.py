"""Behavior tests for immutable, fail-closed execution planning."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from hyperspectrum.contracts import MetricSpec, TaskSpec
from hyperspectrum.datasets import BenchmarkAssetIdentity
from hyperspectrum.execution.plan import RunPlanV3, build_run_plan, parse_run_plan
from hyperspectrum.hyperdata.models import DatasetCandidate
from hyperspectrum.registry.loader import load_tool_manifest
from hyperspectrum.registry.models import ResourceBudget, ToolAvailability, ToolManifest
from hyperspectrum.tasks.recommend import ReadinessVerdict

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIGEST = sha256(
    (ROOT / "tests/fixtures/xas/denoising-pairs.npz").read_bytes()
).hexdigest()
CATALOG_DIGEST = "9" * 64
SOURCE_DATASET_DIGEST = "1" * 64
SOURCE_CONTENT_MANIFEST_DIGEST = "2" * 64


def task() -> TaskSpec:
    return TaskSpec(
        schema_version="hyperspectrum-task/v1",
        id="xas-denoising",
        modality="xas",
        task_type="denoising",
        input_roles=("raw_signal",),
        output_kind="dense_array",
        ground_truth_roles=("clean_spectrum",),
        split_group_keys=("sample_id", "compound_id"),
        metrics=(
            MetricSpec(
                key="nrmse",
                direction="min",
                aggregation="group_mean",
                primary=True,
            ),
        ),
    )


def dataset(
    *,
    dataset_version: str | None = "fixture-v1",
    content_digest: str | None = CATALOG_DIGEST,
) -> DatasetCandidate:
    return DatasetCandidate(
        dataset_code="synthetic-xas-denoising-fixture",
        dataset_version=dataset_version,
        content_digest=content_digest,
        title="Synthetic XAS fixture",
        description="Test-only deterministic spectra",
        file_count=1,
        parsed_file_count=1,
        formats=("NPZ",),
        license="test-only",
        evidence={},
    )


def verdict(*, status: str = "scoreable") -> ReadinessVerdict:
    if status != "scoreable":
        return ReadinessVerdict(
            dataset_code="synthetic-xas-denoising-fixture",
            dataset_version="fixture-v1",
            content_digest=CATALOG_DIGEST,
            status=status,  # type: ignore[arg-type]
            reasons=("test_non_scoreable",),
            candidate_tasks=(),
        )
    return ReadinessVerdict(
        dataset_code="synthetic-xas-denoising-fixture",
        dataset_version="fixture-v1",
        content_digest=CATALOG_DIGEST,
        status="scoreable",
        reasons=("xas_verified_noisy_clean_pair",),
        candidate_tasks=("denoising",),
        ground_truth_roles=("clean_spectrum",),
        split_group_keys=("sample_id", "compound_id"),
    )


def savgol() -> ToolManifest:
    return load_tool_manifest(ROOT / "tools/xas/savgol/tool.yaml")


def xasdenoise() -> ToolManifest:
    return load_tool_manifest(ROOT / "tools/xas/xasdenoise/tool.yaml")


def xasdenoise_parameters() -> dict[str, object]:
    return {
        "normalization_method": "identity_raw",
        "model_normalization_method": None,
        "preprocessing": {
            "schema_version": "hyperspectrum-xasdenoise-step-baseline/v1",
            "method": "symmetric_tanh_step",
            "inverse": "add_same_fitted_baseline",
        },
        "device": "auto",
    }


def resources(**changes: object) -> ResourceBudget:
    values: dict[str, object] = {"cpu": 2, "memory_gb": 4.0, "gpu_available": False}
    values.update(changes)
    return ResourceBudget.model_validate(values)


def benchmark_asset(
    *, benchmark_asset_digest: str = FIXTURE_DIGEST
) -> BenchmarkAssetIdentity:
    return BenchmarkAssetIdentity(
        dataset_code="synthetic-xas-denoising-fixture",
        dataset_version="fixture-v1",
        declared_manifest_digest=CATALOG_DIGEST,
        source_dataset_digest=SOURCE_DATASET_DIGEST,
        source_content_manifest_digest=SOURCE_CONTENT_MANIFEST_DIGEST,
        benchmark_asset_digest=benchmark_asset_digest,
    )


def plan(tmp_path: Path, **changes: object) -> RunPlanV3:
    values: dict[str, object] = {
        "task": task(),
        "dataset": dataset(),
        "verdict": verdict(),
        "benchmark_asset": benchmark_asset(),
        "tool": savgol(),
        "availability": ToolAvailability(available=True),
        "backend": "local",
        "resources": resources(),
        "max_samples": 3,
        "selected_sample_ids": ("feo-1", "feo-2", "fe2o3-1"),
        "output_directory": tmp_path / "run",
        "dry_run": False,
        "parameters": {"window_length": 5, "polyorder": 2},
        "data_origin": "synthetic-test",
    }
    values.update(changes)
    selected_dataset = values["dataset"]
    if "benchmark_asset" not in changes and isinstance(
        selected_dataset, DatasetCandidate
    ):
        test_asset_digest = selected_dataset.evidence.get("test_benchmark_asset_digest")
        if isinstance(test_asset_digest, str):
            values["benchmark_asset"] = benchmark_asset(
                benchmark_asset_digest=test_asset_digest
            )
    return build_run_plan(**values)  # type: ignore[arg-type]


def test_plan_preserves_every_reproducibility_input_and_detaches_parameters(
    tmp_path: Path,
) -> None:
    # Break caught: planning could drop or alias a version, digest, resource, or algorithm setting.
    parameters = {"window_length": 5, "polyorder": 2, "nested": {"mode": "interp"}}

    result = plan(tmp_path, parameters=parameters)
    parameters["nested"]["mode"] = "mirror"  # type: ignore[index]

    assert result.schema_version == "hyperspectrum-run-plan/v3"
    assert result.task == task()
    assert result.dataset_code == "synthetic-xas-denoising-fixture"
    assert result.dataset_version == "fixture-v1"
    assert result.source_dataset_digest == SOURCE_DATASET_DIGEST
    assert result.source_content_manifest_digest == SOURCE_CONTENT_MANIFEST_DIGEST
    assert result.benchmark_asset_digest == FIXTURE_DIGEST
    assert (
        len(
            {
                result.source_dataset_digest,
                result.source_content_manifest_digest,
                result.benchmark_asset_digest,
                dataset().content_digest,
            }
        )
        == 4
    )
    assert result.tool_digest == savgol().tool_digest
    assert len(result.implementation_digest) == 64
    assert result.weight_digest == "none"
    assert result.backend == "local"
    assert result.resources == resources()
    assert result.max_samples == 3
    assert result.selection_policy == "explicit_order"
    assert result.selection_policy_version == "1"
    assert result.selected_sample_ids == ("feo-1", "feo-2", "fe2o3-1")
    assert result.output_directory == tmp_path / "run"
    assert result.dry_run is False
    assert result.data_origin == "synthetic-test"
    assert result.parameters["nested"]["mode"] == "interp"  # type: ignore[index]
    assert len(result.model_digest) == len(result.environment_digest) == 64
    assert len(result.plan_digest) == 64
    with pytest.raises(TypeError):
        result.parameters["new"] = "forbidden"  # type: ignore[index]


def test_plan_digest_binds_the_exact_ordered_sample_selection(tmp_path: Path) -> None:
    # Break caught: two different three-sample selections could share a plan/run identity.
    first = plan(
        tmp_path,
        selected_sample_ids=("sample-1", "sample-2", "sample-3"),
    )
    different_member = plan(
        tmp_path,
        selected_sample_ids=("sample-1", "sample-2", "sample-4"),
    )
    different_order = plan(
        tmp_path,
        selected_sample_ids=("sample-3", "sample-2", "sample-1"),
    )

    assert first.plan_digest != different_member.plan_digest
    assert first.plan_digest != different_order.plan_digest


def test_v2_plan_remains_parseable_for_legacy_execution(tmp_path: Path) -> None:
    current = plan(tmp_path)
    raw = current.model_dump(mode="json")
    raw["schema_version"] = "hyperspectrum-run-plan/v2"
    raw["data_digest"] = raw.pop("benchmark_asset_digest")
    del raw["source_dataset_digest"]
    del raw["source_content_manifest_digest"]

    legacy = parse_run_plan(raw)

    assert legacy.schema_version == "hyperspectrum-run-plan/v2"
    assert legacy.data_digest == FIXTURE_DIGEST


@pytest.mark.parametrize("field", ["dataset_code", "dataset_version"])
def test_planning_rejects_benchmark_for_different_dataset_identity(
    tmp_path: Path, field: str
) -> None:
    values = {
        "dataset_code": "synthetic-xas-denoising-fixture",
        "dataset_version": "fixture-v1",
        "declared_manifest_digest": CATALOG_DIGEST,
        "source_dataset_digest": SOURCE_DATASET_DIGEST,
        "source_content_manifest_digest": SOURCE_CONTENT_MANIFEST_DIGEST,
        "benchmark_asset_digest": FIXTURE_DIGEST,
    }
    values[field] = "different"

    with pytest.raises(ValueError, match="benchmark asset dataset identity"):
        plan(tmp_path, benchmark_asset=BenchmarkAssetIdentity(**values))


def test_planning_binds_catalog_content_to_the_declared_source_manifest(
    tmp_path: Path,
) -> None:
    mismatched = BenchmarkAssetIdentity(
        dataset_code="synthetic-xas-denoising-fixture",
        dataset_version="fixture-v1",
        declared_manifest_digest="f" * 64,
        source_dataset_digest=SOURCE_DATASET_DIGEST,
        source_content_manifest_digest=SOURCE_CONTENT_MANIFEST_DIGEST,
        benchmark_asset_digest=FIXTURE_DIGEST,
    )

    with pytest.raises(ValueError, match="declared manifest digest"):
        plan(tmp_path, benchmark_asset=mismatched)


@pytest.mark.parametrize(
    "selected",
    [
        (),
        ("sample-1", "sample-1"),
        ("sample-1", ""),
        ("sample-1", "sample-2", "sample-3", "sample-4"),
    ],
)
def test_planning_rejects_invalid_or_over_limit_sample_selection(
    tmp_path: Path, selected: tuple[str, ...]
) -> None:
    # Break caught: an ambiguous or oversized selection could be admitted into identity.
    with pytest.raises(ValueError, match="selected sample"):
        plan(tmp_path, selected_sample_ids=selected)


def test_implementation_digest_covers_entrypoint_and_local_dependency_bytes(
    tmp_path: Path,
) -> None:
    # Break caught: changing arrays.py could retain the implementation identity of code that imports it.
    paths = (
        "src/hyperspectrum/plugins/xas/arrays.py",
        "src/hyperspectrum/plugins/xas/baselines.py",
    )
    modules = [
        {
            "module": f"hyperspectrum.plugins.xas.{Path(path).stem}",
            "path": path,
            "sha256": sha256((ROOT / path).read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    expected = sha256(
        json.dumps(
            {"modules": modules},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    result = plan(tmp_path)

    assert result.implementation_digest == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_code", "another-dataset"),
        ("dataset_version", "v2"),
        ("content_digest", "f" * 64),
    ],
)
def test_planning_rejects_verdict_for_a_different_dataset_identity(
    tmp_path: Path, field: str, value: str
) -> None:
    # Break caught: readiness evidence from dataset A could authorize execution of dataset B.
    mismatched = verdict().model_copy(update={field: value})

    with pytest.raises(ValueError, match="dataset identity"):
        plan(tmp_path, verdict=mismatched)


def test_model_digest_is_canonical_and_identifies_exact_algorithm_configuration(
    tmp_path: Path,
) -> None:
    # Break caught: key order could change identity or a setting change could retain the same model ID.
    first = plan(tmp_path, parameters={"window_length": 5, "polyorder": 2})
    reordered = plan(tmp_path, parameters={"polyorder": 2, "window_length": 5})
    changed = plan(tmp_path, parameters={"window_length": 7, "polyorder": 2})

    assert first.model_digest == reordered.model_digest
    assert first.plan_digest == reordered.plan_digest
    assert first.model_digest != changed.model_digest
    assert first.plan_digest != changed.plan_digest


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (dataset(dataset_version=None), "dataset version"),
        (dataset(content_digest=None), "content digest"),
        (dataset(content_digest="digest"), "content digest"),
    ],
)
def test_planning_rejects_unpinned_dataset_identity(
    tmp_path: Path, candidate: DatasetCandidate, message: str
) -> None:
    # Break caught: a run could be planned against mutable or unverifiable dataset contents.
    with pytest.raises(ValueError, match=message):
        plan(tmp_path, dataset=candidate)


@pytest.mark.parametrize("status", ["inference_only", "blocked"])
def test_planning_fails_closed_for_non_scoreable_verdicts(
    tmp_path: Path, status: str
) -> None:
    # Break caught: an inference-only or blocked discovery verdict could enter evaluation execution.
    with pytest.raises(ValueError, match="scoreable"):
        plan(tmp_path, verdict=verdict(status=status))


def test_planning_rejects_backend_resource_mismatch(tmp_path: Path) -> None:
    # Break caught: a selected backend could admit a tool it cannot execute reliably.
    with pytest.raises(ValueError, match="memory"):
        plan(tmp_path, resources=resources(memory_gb=0.5))


def test_planning_requires_one_dense_denoised_signal_output(tmp_path: Path) -> None:
    # Break caught: a nonempty but task-incompatible output could pass a vacuous kind-only check.
    source = savgol().model_dump(mode="json", exclude_none=True)
    source["outputs"] = [{"role": "normalized_signal", "kind": "dense_array"}]
    incompatible = ToolManifest.model_validate(source)

    with pytest.raises(ValueError, match="denoised_signal"):
        plan(tmp_path, tool=incompatible)


def test_planning_rejects_tool_output_kind_that_differs_from_task(
    tmp_path: Path,
) -> None:
    # Break caught: an image task could authorize a tool that publishes dense-array artifacts.
    image_task = task().model_copy(update={"output_kind": "image"})

    with pytest.raises(ValueError, match="output kind"):
        plan(tmp_path, task=image_task)


def test_planning_rejects_unverified_mounted_model_weights(
    tmp_path: Path,
) -> None:
    # Break caught: model execution could begin before checkpoint integrity was verified.
    with pytest.raises(ValueError, match="weights-unverified"):
        plan(
            tmp_path,
            tool=xasdenoise(),
            availability=ToolAvailability(
                available=False, reasons=("weights-unverified",)
            ),
            resources=resources(cpu=4, memory_gb=16),
            parameters=xasdenoise_parameters(),
        )


def test_verified_xasdenoise_plan_records_exact_model_semantics(tmp_path: Path) -> None:
    # Break caught: the official null normalization could be silently replaced by
    # range scaling, or the model-native baseline transform could go unversioned.
    result = plan(
        tmp_path,
        tool=xasdenoise(),
        availability=ToolAvailability(available=True),
        resources=resources(cpu=4, memory_gb=16),
        parameters=xasdenoise_parameters(),
    )

    assert result.weight_digest == (
        "09620ee9ea0c96585f534d76ce42aa72edf2cf71e481f5737e43e93116e24160"
    )
    assert result.parameters == xasdenoise_parameters()

    changed = deepcopy(xasdenoise_parameters())
    changed["normalization_method"] = "per_spectrum_range"
    with pytest.raises(ValueError, match="identity_raw"):
        plan(
            tmp_path,
            tool=xasdenoise(),
            availability=ToolAvailability(available=True),
            resources=resources(cpu=4, memory_gb=16),
            parameters=changed,
        )
