"""Behavior tests for immutable, fail-closed execution planning."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from hyperspectrum.contracts import MetricSpec, TaskSpec
from hyperspectrum.execution.plan import RunPlan, build_run_plan
from hyperspectrum.hyperdata.models import DatasetCandidate
from hyperspectrum.registry.loader import load_tool_manifest
from hyperspectrum.registry.models import ResourceBudget, ToolAvailability, ToolManifest
from hyperspectrum.tasks.recommend import ReadinessVerdict

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIGEST = sha256(
    (ROOT / "tests/fixtures/xas/denoising-pairs.npz").read_bytes()
).hexdigest()


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
    content_digest: str | None = FIXTURE_DIGEST,
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
            status=status,  # type: ignore[arg-type]
            reasons=("test_non_scoreable",),
            candidate_tasks=(),
        )
    return ReadinessVerdict(
        status="scoreable",
        reasons=("xas_verified_noisy_clean_pair",),
        candidate_tasks=("denoising",),
        ground_truth_roles=("clean_spectrum",),
        split_group_keys=("sample_id", "compound_id"),
    )


def savgol() -> ToolManifest:
    return load_tool_manifest(ROOT / "tools/xas/savgol/tool.yaml")


def resources(**changes: object) -> ResourceBudget:
    values: dict[str, object] = {"cpu": 2, "memory_gb": 4.0, "gpu_available": False}
    values.update(changes)
    return ResourceBudget.model_validate(values)


def plan(tmp_path: Path, **changes: object) -> RunPlan:
    values: dict[str, object] = {
        "task": task(),
        "dataset": dataset(),
        "verdict": verdict(),
        "tool": savgol(),
        "availability": ToolAvailability(available=True),
        "backend": "local",
        "resources": resources(),
        "max_samples": 3,
        "output_directory": tmp_path / "run",
        "dry_run": False,
        "parameters": {"window_length": 5, "polyorder": 2},
        "data_origin": "synthetic-test",
    }
    values.update(changes)
    return build_run_plan(**values)  # type: ignore[arg-type]


def test_plan_preserves_every_reproducibility_input_and_detaches_parameters(
    tmp_path: Path,
) -> None:
    # Break caught: planning could drop or alias a version, digest, resource, or algorithm setting.
    parameters = {"window_length": 5, "polyorder": 2, "nested": {"mode": "interp"}}

    result = plan(tmp_path, parameters=parameters)
    parameters["nested"]["mode"] = "mirror"  # type: ignore[index]

    assert result.task == task()
    assert result.dataset_code == "synthetic-xas-denoising-fixture"
    assert result.dataset_version == "fixture-v1"
    assert result.data_digest == FIXTURE_DIGEST
    assert result.tool_digest == savgol().tool_digest
    assert result.weight_digest == "none"
    assert result.backend == "local"
    assert result.resources == resources()
    assert result.max_samples == 3
    assert result.output_directory == tmp_path / "run"
    assert result.dry_run is False
    assert result.data_origin == "synthetic-test"
    assert result.parameters["nested"]["mode"] == "interp"  # type: ignore[index]
    assert len(result.model_digest) == len(result.environment_digest) == 64
    assert len(result.plan_digest) == 64
    with pytest.raises(TypeError):
        result.parameters["new"] = "forbidden"  # type: ignore[index]


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


def test_planning_rejects_required_missing_and_unverified_model_weights(
    tmp_path: Path,
) -> None:
    # Break caught: model execution could begin before checkpoint integrity was verified.
    missing = load_tool_manifest(ROOT / "tools/xas/xasdenoise/tool.yaml")
    present_data = deepcopy(missing.model_dump(mode="json", exclude_none=True))
    present_data["weights"] = {
        "required": True,
        "state": "present",
        "allow_download": False,
        "digest": "b" * 64,
    }
    present = ToolManifest.model_validate(present_data)

    with pytest.raises(ValueError, match="required weights are missing"):
        plan(tmp_path, tool=missing, availability=ToolAvailability(available=True))
    with pytest.raises(ValueError, match="weights-unverified"):
        plan(
            tmp_path,
            tool=present,
            availability=ToolAvailability(
                available=False, reasons=("weights-unverified",)
            ),
        )


def test_verified_model_weights_retain_their_exact_digest(tmp_path: Path) -> None:
    # Break caught: a model plan could erase its checkpoint identity or mislabel it as classical.
    source = savgol().model_dump(mode="json", exclude_none=True)
    source["weights"] = {
        "required": True,
        "state": "present",
        "allow_download": False,
        "digest": "b" * 64,
    }
    model = ToolManifest.model_validate(source)

    result = plan(tmp_path, tool=model, availability=ToolAvailability(available=True))

    assert result.weight_digest == "b" * 64
