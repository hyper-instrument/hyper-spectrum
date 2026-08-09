"""Focused integration tests for atomic local Savitzky-Golay smoke runs."""

from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hyperspectrum.adapters.xasdenoise import (
    XASDenoisePrediction,
    preprocess_step_baseline,
)
from hyperspectrum.contracts import PredictionBundleV2, PredictionBundleV3
from hyperspectrum.execution import local
from hyperspectrum.execution.local import execute_local_run
from hyperspectrum.execution.plan import RunPlanV2, RunPlanV3
from hyperspectrum.hyperdata.models import DatasetCandidate
from hyperspectrum.plugins.xas.arrays import XASSpectrum
from hyperspectrum.registry.models import ResourceBudget, ToolAvailability, ToolManifest
from hyperspectrum.tasks.recommend import ReadinessVerdict

from .test_plan import (
    SOURCE_CONTENT_MANIFEST_DIGEST,
    SOURCE_DATASET_DIGEST,
    dataset,
    plan,
    savgol,
    verdict,
    xasdenoise,
    xasdenoise_parameters,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/xas/denoising-pairs.npz"
FIXTURE_DIGEST = sha256(FIXTURE.read_bytes()).hexdigest()


def fixture_ids(limit: int = 3) -> tuple[str, ...]:
    with np.load(FIXTURE, allow_pickle=False) as data:
        return tuple(str(value) for value in data["sample_ids"][:limit])


def dataset_for_source(path: Path) -> DatasetCandidate:
    return dataset().model_copy(
        update={
            "evidence": {
                "test_benchmark_asset_digest": sha256(path.read_bytes()).hexdigest()
            }
        }
    )


def verdict_for_source(path: Path) -> ReadinessVerdict:
    _ = path
    return verdict()


def noisy_source(path: Path) -> Path:
    with np.load(FIXTURE, allow_pickle=False) as data:
        np.savez(
            path,
            energy=data["energy"],
            noisy=data["noisy"],
            sample_ids=data["sample_ids"],
            group_ids=data["group_ids"],
            energy_unit=data["energy_unit"],
        )
    return path


def canonical_benchmark_source(
    path: Path,
    *,
    sample_count: int | None = None,
    updates: dict[str, np.ndarray] | None = None,
    extras: dict[str, np.ndarray] | None = None,
) -> Path:
    with np.load(FIXTURE, allow_pickle=False) as data:
        sample_ids = data["sample_ids"]
        group_ids = data["group_ids"]
        source_energy = data["energy"]
        noisy = data["noisy"]
        point_count = noisy.shape[1]
        fixture_sample_count = noisy.shape[0]
        i0 = np.full((fixture_sample_count, point_count), 100_000.0)
        ketek = np.maximum(noisy * i0, 0.0)
        arrays = {
            "energy": source_energy[0],
            "energy_unit": np.array("eV"),
            "experiments": np.array(["Exp2"] * fixture_sample_count),
            "group_ids": group_ids,
            "i0_counts": i0,
            "ketek_counts": ketek,
            "noisy": noisy,
            "noisy_i0_counts": i0,
            "noisy_ketek_counts": ketek,
            "pseudo_clean": data["clean"],
            "sample_ids": sample_ids,
            "sample_seeds": np.arange(fixture_sample_count, dtype=np.uint64),
            "source_energy": source_energy,
            "source_paths": np.array(
                [f"sample-{index}.dat" for index in range(fixture_sample_count)]
            ),
            "source_sha256": np.array(
                [f"{index + 1:064x}" for index in range(fixture_sample_count)]
            ),
            "splits": np.array(["train"] * fixture_sample_count),
        }
    selected_count = sample_count
    if selected_count is not None:
        for key in (
            "experiments",
            "group_ids",
            "i0_counts",
            "ketek_counts",
            "noisy",
            "noisy_i0_counts",
            "noisy_ketek_counts",
            "pseudo_clean",
            "sample_ids",
            "sample_seeds",
            "source_energy",
            "source_paths",
            "source_sha256",
            "splits",
        ):
            arrays[key] = arrays[key][:selected_count]
    if updates is not None:
        arrays.update(updates)
    if extras is not None:
        arrays.update(extras)
    np.savez(path, **arrays)
    return path


def inference_plan(tmp_path: Path, **changes: object) -> tuple[Path, RunPlanV3]:
    source = canonical_benchmark_source(tmp_path / "benchmark-input.npz")
    values: dict[str, object] = {
        "dataset": dataset_for_source(source),
        "verdict": verdict_for_source(source),
        "selected_sample_ids": fixture_ids(),
    }
    values.update(changes)
    return source, plan(tmp_path, **values)


def assert_no_scoring_fields(value: object) -> None:
    forbidden = {
        "ground_truth",
        "nrmse",
        "metric",
        "metrics",
        "leaderboard",
        "ranking",
        "report",
    }
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for nested in value.values():
            assert_no_scoring_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_no_scoring_fields(nested)


def test_local_savgol_smoke_publishes_a_complete_prediction_bundle_without_scoring(
    tmp_path: Path,
) -> None:
    # Break caught: local smoke execution could publish incomplete artifacts or ACE-owned scores.
    source, run_plan = inference_plan(tmp_path)
    selected = fixture_ids()

    returned = execute_local_run(
        run_plan,
        tool=savgol(),
        selected_sample_ids=selected,
        source_npz=source,
    )

    assert run_plan.output_directory.is_dir()
    predictions_data = json.loads(
        (run_plan.output_directory / "predictions.json").read_text()
    )
    bundle = PredictionBundleV3.model_validate(predictions_data)
    assert bundle.schema_version == "hyperspectrum-prediction/v3"
    assert bundle.provenance["plan_schema_version"] == "hyperspectrum-run-plan/v3"
    run_data = json.loads((run_plan.output_directory / "run.json").read_text())
    assert returned == bundle
    assert len(bundle.predictions) == 3
    assert bundle.failures == ()
    assert bundle.provenance["model_digest"] == run_plan.model_digest
    assert bundle.provenance["tool_digest"] == run_plan.tool_digest
    assert bundle.provenance["implementation_digest"] == run_plan.implementation_digest
    assert bundle.provenance["source_dataset_digest"] == SOURCE_DATASET_DIGEST
    assert bundle.provenance["source_content_manifest_digest"] == (
        SOURCE_CONTENT_MANIFEST_DIGEST
    )
    assert bundle.provenance["benchmark_asset_digest"] == (
        run_plan.benchmark_asset_digest
    )
    assert (
        len(
            {
                run_plan.source_dataset_digest,
                run_plan.source_content_manifest_digest,
                run_plan.benchmark_asset_digest,
            }
        )
        == 3
    )
    assert "data_digest" not in bundle.provenance
    assert bundle.provenance["environment_digest"] == run_plan.environment_digest
    assert bundle.provenance["weight_digest"] == "none"
    assert bundle.provenance["data_origin"] == "synthetic-test"
    assert run_data == {
        "schema_version": "hyperspectrum-run/v2",
        "run_id": bundle.run_id,
        "status": "completed",
        "planned_sample_count": 3,
        "prediction_count": 3,
        "failure_count": 0,
        "data_origin": "synthetic-test",
        "plan_digest": run_plan.plan_digest,
        "model_digest": run_plan.model_digest,
        "tool_digest": run_plan.tool_digest,
        "implementation_digest": run_plan.implementation_digest,
        "weight_digest": "none",
        "source_dataset_digest": run_plan.source_dataset_digest,
        "source_content_manifest_digest": run_plan.source_content_manifest_digest,
        "benchmark_asset_digest": run_plan.benchmark_asset_digest,
        "environment_digest": run_plan.environment_digest,
    }
    for artifact in bundle.predictions:
        path = run_plan.output_directory / artifact.uri
        assert path.is_file()
        assert path.resolve().is_relative_to(run_plan.output_directory.resolve())
        assert artifact.sha256 is not None
        with np.load(path, allow_pickle=False) as output:
            assert set(output.files) == {
                "sample_id",
                "group_id",
                "energy",
                "intensity",
                "energy_unit",
            }
            assert str(output["sample_id"]) in selected
    assert_no_scoring_fields(predictions_data)
    assert_no_scoring_fields(run_data)


def test_local_xasdenoise_uses_one_mounted_asset_and_records_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Break caught: OCI/Bohr could need a second adapter, or model-native baseline
    # state could be omitted from the existing v3 PredictionBundle provenance.
    payload = b"deterministic-fake-state-dict"
    weights_path = tmp_path / "mounted.pth"
    weights_path.write_bytes(payload)
    raw_tool = xasdenoise().model_dump(mode="json", exclude_none=True)
    raw_tool["weights"] = {
        "required": True,
        "state": "present",
        "allow_download": False,
        "digest": sha256(payload).hexdigest(),
        "asset_id": "test-only-asset",
        "source_url": "https://example.invalid/model.pth",
        "filename": "model.pth",
        "size_bytes": len(payload),
        "license": "test-only",
    }
    tool = ToolManifest.model_validate(raw_tool)
    source = canonical_benchmark_source(tmp_path / "benchmark-input.npz")
    run_plan = plan(
        tmp_path,
        dataset=dataset_for_source(source),
        verdict=verdict_for_source(source),
        selected_sample_ids=fixture_ids(2),
        tool=tool,
        availability=ToolAvailability(available=True),
        resources=ResourceBudget(cpu=4, memory_gb=16, gpu_available=False),
        parameters=xasdenoise_parameters(),
    )
    _, _, preprocessing_state = preprocess_step_baseline(
        np.linspace(5693.0, 5801.4, 135), np.linspace(0.0, 1.0, 135)
    )

    def fake_runner(
        spectra: tuple[object, ...], *, weights_path: Path, device: str
    ) -> tuple[XASDenoisePrediction, ...]:
        assert weights_path == tmp_path / "mounted.pth"
        assert device == "auto"
        return tuple(
            XASDenoisePrediction(
                spectrum=item,  # type: ignore[arg-type]
                method="xasdenoise",
                optimization_kind="no_training",
                preprocessing_state=preprocessing_state,
                device="cpu",
            )
            for item in spectra
        )

    monkeypatch.setattr(
        local,
        "_load_tool_callable",
        lambda resolved, tool_id: SimpleNamespace(
            runner=fake_runner,
            prediction_type=XASDenoisePrediction,
            failure_type=type("NeverFailure", (), {}),
            spectrum_type=XASSpectrum,
            isolated_module_names=(),
        ),
    )

    bundle = execute_local_run(
        run_plan,
        tool=tool,
        selected_sample_ids=fixture_ids(2),
        source_npz=source,
        weight_files=(weights_path,),
    )

    assert isinstance(bundle, PredictionBundleV3)
    assert len(bundle.predictions) == 2
    assert bundle.provenance["weight_digest"] == sha256(payload).hexdigest()
    preprocessing = bundle.provenance["preprocessing"]
    assert preprocessing["schema_version"] == (  # type: ignore[index]
        "hyperspectrum-xasdenoise-step-baseline/v1"
    )
    assert preprocessing["normalization_method"] == "identity_raw"  # type: ignore[index]
    assert [state["sample_id"] for state in preprocessing["states"]] == [  # type: ignore[index]
        "feo-1",
        "feo-2",
    ]
    assert bundle.provenance["runtime"]["device"] == "cpu"  # type: ignore[index]
    assert_no_scoring_fields(bundle.model_dump(mode="json"))


def test_legacy_v2_plan_still_executes_and_emits_v2_prediction(
    tmp_path: Path,
) -> None:
    source = noisy_source(tmp_path / "legacy-noisy-input.npz")
    current = plan(
        tmp_path,
        dataset=dataset_for_source(source),
        verdict=verdict_for_source(source),
        selected_sample_ids=fixture_ids(),
    )
    raw = current.model_dump(mode="json")
    raw["schema_version"] = "hyperspectrum-run-plan/v2"
    raw["data_digest"] = raw.pop("benchmark_asset_digest")
    del raw["source_dataset_digest"]
    del raw["source_content_manifest_digest"]
    legacy = RunPlanV2.model_validate(raw)

    bundle = execute_local_run(
        legacy,
        tool=savgol(),
        selected_sample_ids=fixture_ids(),
        source_npz=source,
    )

    assert isinstance(bundle, PredictionBundleV2)
    assert bundle.schema_version == "hyperspectrum-prediction/v2"
    assert bundle.provenance["data_digest"] == legacy.data_digest


def test_v3_plan_rejects_legacy_five_key_inference_npz(tmp_path: Path) -> None:
    source = noisy_source(tmp_path / "legacy-noisy-input.npz")
    run_plan = plan(
        tmp_path,
        dataset=dataset_for_source(source),
        verdict=verdict_for_source(source),
        selected_sample_ids=fixture_ids(),
    )

    with pytest.raises(ValueError, match="v3.*canonical benchmark"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            source_npz=source,
        )

    assert not run_plan.output_directory.exists()


def test_executor_rejects_selection_that_differs_from_the_plan(tmp_path: Path) -> None:
    # Break caught: callers could reuse one plan/run identity for a different sample set.
    planned = fixture_ids(2)
    supplied = fixture_ids(3)[1:]
    source, run_plan = inference_plan(
        tmp_path,
        selected_sample_ids=planned,
    )

    with pytest.raises(ValueError, match="does not match the run plan"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=supplied,
            source_npz=source,
        )

    assert not run_plan.output_directory.exists()


def test_executor_runs_fresh_verified_source_not_a_preimported_callable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Break caught: execution could ignore entrypoint source and call a preimported hardcoded function.
    from hyperspectrum.plugins.xas import baselines

    def wrong_callable(*args: object, **kwargs: object) -> object:
        raise AssertionError("preimported callable must not run")

    monkeypatch.setattr(baselines, "savgol_filter", wrong_callable)

    source, run_plan = inference_plan(tmp_path, selected_sample_ids=fixture_ids(1))
    bundle = execute_local_run(
        run_plan,
        tool=savgol(),
        selected_sample_ids=fixture_ids(1),
        source_npz=source,
    )

    assert len(bundle.predictions) == 1


def test_executor_uses_verified_dependency_bytes_not_cached_module_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Break caught: replacing arrays.XASSpectrum after planning could change output under one digest.
    from hyperspectrum.plugins.xas import arrays

    class WrongSpectrum:
        def __init__(self, **values: object) -> None:
            self.sample_id = values["sample_id"]
            self.group_id = values["group_id"]
            self.energy = values["energy"]
            self.intensity = np.zeros_like(values["intensity"])
            self.energy_unit = values["energy_unit"]

    source, run_plan = inference_plan(tmp_path, selected_sample_ids=fixture_ids(1))
    monkeypatch.setattr(arrays, "XASSpectrum", WrongSpectrum)

    bundle = execute_local_run(
        run_plan,
        tool=savgol(),
        selected_sample_ids=fixture_ids(1),
        source_npz=source,
    )

    with np.load(
        run_plan.output_directory / bundle.predictions[0].uri, allow_pickle=False
    ) as output:
        assert not np.allclose(output["intensity"], 0.0)


def test_executor_rejects_changed_local_dependency_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Break caught: arrays.py could change after planning while every recorded digest stayed fixed.
    dependency = (ROOT / "src/hyperspectrum/plugins/xas/arrays.py").resolve()
    original_read_bytes = Path.read_bytes
    source, run_plan = inference_plan(tmp_path, selected_sample_ids=fixture_ids(1))

    def changed_read_bytes(path: Path) -> bytes:
        contents = original_read_bytes(path)
        if path.resolve() == dependency:
            return contents + b"\n# changed after planning\n"
        return contents

    monkeypatch.setattr(Path, "read_bytes", changed_read_bytes)

    with pytest.raises(ValueError, match="implementation digest"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(1),
            source_npz=source,
        )

    assert not run_plan.output_directory.exists()


def test_partial_input_materialization_aborts_without_a_final_run_directory(
    tmp_path: Path,
) -> None:
    # Break caught: execution could silently filter missing selected samples and publish partial output.
    partial = canonical_benchmark_source(tmp_path / "partial.npz", sample_count=2)
    run_plan = plan(
        tmp_path,
        dataset=dataset_for_source(partial),
        verdict=verdict_for_source(partial),
    )

    with pytest.raises(ValueError, match="complete selected input set"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            source_npz=partial,
        )

    assert not run_plan.output_directory.exists()
    assert not tuple(tmp_path.glob(".run.*"))


def test_materialization_exception_aborts_without_a_final_run_directory(
    tmp_path: Path,
) -> None:
    # Break caught: a source read error could leave a directory that looks like a completed run.
    malformed = tmp_path / "malformed.npz"
    malformed.write_bytes(b"not-an-npz")
    run_plan = plan(
        tmp_path,
        dataset=dataset_for_source(malformed),
        verdict=verdict_for_source(malformed),
    )

    with pytest.raises((OSError, ValueError)):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            source_npz=malformed,
        )

    assert not run_plan.output_directory.exists()
    assert not tuple(tmp_path.glob(".run.*"))


def test_model_failures_are_recorded_one_for_one_without_disappearing(
    tmp_path: Path,
) -> None:
    # Break caught: sample-level model failures could be dropped or abort an otherwise complete run record.
    source, run_plan = inference_plan(
        tmp_path, parameters={"window_length": 4, "polyorder": 2}
    )

    bundle = execute_local_run(
        run_plan,
        tool=savgol(),
        selected_sample_ids=fixture_ids(),
        source_npz=source,
    )

    assert bundle.predictions == ()
    assert [failure["sample_id"] for failure in bundle.failures] == list(fixture_ids())
    assert {failure["error_type"] for failure in bundle.failures} == {
        "unsupported_setting"
    }
    run_data = json.loads((run_plan.output_directory / "run.json").read_text())
    assert run_data["status"] == "completed_with_failures"
    assert run_data["planned_sample_count"] == run_data["failure_count"] == 3
    assert run_data["prediction_count"] == 0


def test_sample_ids_cannot_escape_the_array_artifact_directory(tmp_path: Path) -> None:
    # Break caught: using a sample ID as a filename could overwrite files outside the run.
    malicious_id = "../../escape"
    source = canonical_benchmark_source(
        tmp_path / "malicious-id.npz",
        sample_count=1,
        updates={
            "energy": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
            "noisy": np.array([[0.0, 0.2, 0.8, 0.3, 0.1]]),
            "sample_ids": np.array([malicious_id]),
            "group_ids": np.array(["synthetic"]),
        },
    )
    run_plan = plan(
        tmp_path,
        max_samples=1,
        selected_sample_ids=(malicious_id,),
        dataset=dataset_for_source(source),
        verdict=verdict_for_source(source),
    )

    bundle = execute_local_run(
        run_plan,
        tool=savgol(),
        selected_sample_ids=(malicious_id,),
        source_npz=source,
    )

    artifact = bundle.predictions[0]
    assert ".." not in Path(artifact.uri).parts
    assert (
        (run_plan.output_directory / artifact.uri)
        .resolve()
        .is_relative_to((run_plan.output_directory / "arrays").resolve())
    )
    assert not (tmp_path / "escape").exists()


def test_dry_run_plan_cannot_execute_or_materialize_inputs(tmp_path: Path) -> None:
    # Break caught: a planning-only command could cause scientific data reads or filesystem writes.
    run_plan = plan(tmp_path, dry_run=True)
    missing = tmp_path / "must-not-be-read.npz"

    with pytest.raises(ValueError, match="dry-run"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            source_npz=missing,
        )

    assert not run_plan.output_directory.exists()


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "clean",
        "target",
        "ground_truth",
        "labels",
        "metric",
        "score",
        "ground_truth_roles",
    ],
)
def test_inference_source_rejects_every_extra_array_before_tool_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forbidden_key: str
) -> None:
    source = canonical_benchmark_source(
        tmp_path / f"forbidden-{forbidden_key}.npz",
        extras={forbidden_key: np.array(["must-not-cross-boundary"])},
    )
    run_plan = plan(
        tmp_path,
        selected_sample_ids=fixture_ids(1),
        dataset=dataset_for_source(source),
        verdict=verdict_for_source(source),
    )

    def forbidden_loader(*args: object, **kwargs: object) -> object:
        raise AssertionError("tool code must not load")

    monkeypatch.setattr(local, "_load_tool_callable", forbidden_loader)

    with pytest.raises(ValueError, match="v3.*canonical benchmark"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(1),
            source_npz=source,
        )

    assert not run_plan.output_directory.exists()


def test_legacy_v2_separately_digested_noisy_only_source_executes_successfully(
    tmp_path: Path,
) -> None:
    source = tmp_path / "noisy-only.npz"
    with np.load(FIXTURE, allow_pickle=False) as data:
        np.savez(
            source,
            energy=data["energy"],
            noisy=data["noisy"],
            sample_ids=data["sample_ids"],
            group_ids=data["group_ids"],
            energy_unit=data["energy_unit"],
        )
    current = plan(
        tmp_path,
        selected_sample_ids=fixture_ids(2),
        dataset=dataset_for_source(source),
        verdict=verdict_for_source(source),
    )
    raw = current.model_dump(mode="json")
    raw["schema_version"] = "hyperspectrum-run-plan/v2"
    raw["data_digest"] = raw.pop("benchmark_asset_digest")
    del raw["source_dataset_digest"]
    del raw["source_content_manifest_digest"]
    run_plan = RunPlanV2.model_validate(raw)

    bundle = execute_local_run(
        run_plan,
        tool=savgol(),
        selected_sample_ids=fixture_ids(2),
        source_npz=source,
    )

    assert len(bundle.predictions) == 2


def test_canonical_materializer_bundle_executes_without_a_format_bridge(
    tmp_path: Path,
) -> None:
    source = canonical_benchmark_source(tmp_path / "benchmark.npz")
    run_plan = plan(
        tmp_path,
        selected_sample_ids=fixture_ids(2),
        dataset=dataset_for_source(source),
        verdict=verdict_for_source(source),
    )

    bundle = execute_local_run(
        run_plan,
        tool=savgol(),
        selected_sample_ids=fixture_ids(2),
        source_npz=source,
    )

    assert isinstance(bundle, PredictionBundleV3)
    assert len(bundle.predictions) == 2


def test_structured_noisy_array_is_rejected_before_tool_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with np.load(FIXTURE, allow_pickle=False) as data:
        noisy = np.empty(
            data["noisy"].shape,
            dtype=[("signal", np.float64), ("ground_truth", np.float64)],
        )
        noisy["signal"] = data["noisy"]
        noisy["ground_truth"] = data["clean"]
    source = canonical_benchmark_source(
        tmp_path / "structured-noisy.npz", updates={"noisy": noisy}
    )
    run_plan = plan(
        tmp_path,
        selected_sample_ids=fixture_ids(1),
        dataset=dataset_for_source(source),
        verdict=verdict_for_source(source),
    )

    def forbidden_loader(*args: object, **kwargs: object) -> object:
        raise AssertionError("tool code must not load")

    monkeypatch.setattr(local, "_load_tool_callable", forbidden_loader)

    with pytest.raises(ValueError, match="structured"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(1),
            source_npz=source,
        )

    assert not run_plan.output_directory.exists()


@pytest.mark.parametrize(
    "energy",
    [
        np.array([[2, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.uint64),
        np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7, np.iinfo(np.uint64).max]],
            dtype=np.uint64,
        ),
        np.array(
            [
                [
                    2**53,
                    2**53 + 1,
                    2**53 + 2,
                    2**53 + 3,
                    2**53 + 4,
                    2**53 + 5,
                    2**53 + 6,
                    2**53 + 7,
                    2**53 + 8,
                ]
            ],
            dtype=np.int64,
        ),
        np.array([[False, True, True, True, True, True, True, True, True]]),
        np.array([[1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0]]),
        np.array([[1.0, 2.0, 3.0, 4.0, np.inf, 6.0, 7.0, 8.0, 9.0]]),
        np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=object),
        np.zeros((1, 9), dtype=[("energy", np.float64), ("target", np.float64)]),
        np.array(
            [[1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j]]
        ),
    ],
    ids=[
        "uint64-nonmonotonic-underflow",
        "uint64-extreme-overflow",
        "int64-inexact-float64",
        "bool",
        "nan",
        "infinity",
        "object",
        "structured",
        "complex",
    ],
)
def test_unsafe_energy_is_rejected_before_entrypoint_or_tool_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, energy: np.ndarray
) -> None:
    source = canonical_benchmark_source(
        tmp_path / "unsafe-energy.npz",
        sample_count=1,
        updates={"energy": energy},
    )
    run_plan = plan(
        tmp_path,
        selected_sample_ids=fixture_ids(1),
        dataset=dataset_for_source(source),
        verdict=verdict_for_source(source),
    )

    def forbidden_boundary(*args: object, **kwargs: object) -> object:
        raise AssertionError("entrypoint or tool loading must not run")

    monkeypatch.setattr(local, "_verify_tool_identity", forbidden_boundary)
    monkeypatch.setattr(local, "_load_tool_callable", forbidden_boundary)

    with pytest.raises((TypeError, ValueError)):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(1),
            source_npz=source,
        )

    assert not run_plan.output_directory.exists()


@pytest.mark.parametrize(
    "energy",
    [
        np.arange(1, 10, dtype=np.uint32)[None, :],
        np.arange(9, 0, -1, dtype=np.int32)[None, :],
        np.arange(2**53, 2**53 + 18, 2, dtype=np.int64)[None, :],
    ],
    ids=[
        "increasing-uint32",
        "decreasing-int32",
        "exact-int64-above-two-to-the-53",
    ],
)
def test_exact_integer_energy_axes_execute_after_float64_canonicalization(
    tmp_path: Path, energy: np.ndarray
) -> None:
    source = canonical_benchmark_source(
        tmp_path / "integer-energy.npz",
        sample_count=1,
        updates={"energy": energy},
    )
    run_plan = plan(
        tmp_path,
        selected_sample_ids=fixture_ids(1),
        dataset=dataset_for_source(source),
        verdict=verdict_for_source(source),
    )

    bundle = execute_local_run(
        run_plan,
        tool=savgol(),
        selected_sample_ids=fixture_ids(1),
        source_npz=source,
    )

    assert len(bundle.predictions) == 1
    prediction = run_plan.output_directory / bundle.predictions[0].uri
    with np.load(prediction, allow_pickle=False) as output:
        assert output["energy"].dtype == np.dtype(np.float64)
        assert np.array_equal(output["energy"], energy.astype(np.float64)[0])


def test_executor_rejects_a_changed_implementation_digest(tmp_path: Path) -> None:
    # Break caught: source-code changes after planning could execute under stale model provenance.
    source, valid_plan = inference_plan(tmp_path)
    run_plan = valid_plan.model_copy(update={"implementation_digest": "f" * 64})

    with pytest.raises(ValueError, match="implementation digest"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            source_npz=source,
        )

    assert not run_plan.output_directory.exists()


def test_executor_recomputes_and_rejects_a_stale_environment_digest(
    tmp_path: Path,
) -> None:
    # Break caught: output could copy a planned environment identity never used for execution.
    run_plan = plan(tmp_path).model_copy(update={"environment_digest": "f" * 64})

    with pytest.raises(ValueError, match="environment digest"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            source_npz=FIXTURE,
        )

    assert not run_plan.output_directory.exists()


def test_savgol_policy_rejects_a_different_manifest_entrypoint(tmp_path: Path) -> None:
    # Break caught: the executor could attribute its hardcoded callable to a changed entrypoint.
    changed = savgol().model_copy(
        update={"entrypoint": "hyperspectrum.plugins.xas.baselines:identity_filter"}
    )
    source, run_plan = inference_plan(tmp_path, tool=changed)

    with pytest.raises(ValueError, match="canonical savgol entrypoint"):
        execute_local_run(
            run_plan,
            tool=changed,
            selected_sample_ids=fixture_ids(),
            source_npz=source,
        )


def test_atomic_rename_noreplace_preserves_both_directories_on_collision(
    tmp_path: Path,
) -> None:
    # Break caught: publication could replace an empty existing run directory on POSIX.
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    with pytest.raises(FileExistsError):
        local._atomic_rename_noreplace(source, destination)

    assert source.is_dir()
    assert destination.is_dir()


def test_destination_created_immediately_before_publish_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Break caught: a check-then-rename TOCTOU could overwrite a concurrent empty directory.
    source, run_plan = inference_plan(tmp_path)
    real_publish = local._atomic_rename_noreplace

    def collide(source: Path, destination: Path) -> None:
        destination.mkdir()
        real_publish(source, destination)

    monkeypatch.setattr(local, "_atomic_rename_noreplace", collide)

    with pytest.raises(FileExistsError):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            source_npz=source,
        )

    assert run_plan.output_directory.is_dir()
    assert tuple(run_plan.output_directory.iterdir()) == ()
    assert not tuple(tmp_path.glob(".run.*"))


def test_executor_refuses_to_overwrite_an_existing_run_directory(
    tmp_path: Path,
) -> None:
    # Break caught: rerunning a plan could replace immutable prior evidence.
    run_plan = plan(tmp_path)
    run_plan.output_directory.mkdir()
    marker = run_plan.output_directory / "keep.txt"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            source_npz=tmp_path / "must-not-be-read.npz",
        )

    assert marker.read_text() == "original"


def test_executor_hashes_source_bytes_and_rejects_a_different_asset_digest(
    tmp_path: Path,
) -> None:
    # Break caught: a caller could assert a planned digest while supplying unrelated input bytes.
    run_plan = plan(tmp_path)
    changed = tmp_path / "changed.npz"
    shutil.copyfile(FIXTURE, changed)
    changed.write_bytes(changed.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="benchmark asset digest"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            source_npz=changed,
        )

    assert not run_plan.output_directory.exists()


def test_executor_rejects_a_manifest_that_does_not_match_the_immutable_plan(
    tmp_path: Path,
) -> None:
    # Break caught: an executor could run changed tool semantics under an old provenance digest.
    source, valid_plan = inference_plan(tmp_path)
    run_plan = valid_plan.model_copy(update={"tool_digest": "f" * 64})

    with pytest.raises(ValueError, match="tool digest"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            source_npz=source,
        )

    assert not run_plan.output_directory.exists()
