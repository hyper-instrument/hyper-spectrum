"""Focused integration tests for atomic local Savitzky-Golay smoke runs."""

from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from hyperspectrum.contracts import PredictionBundle
from hyperspectrum.execution import local
from hyperspectrum.execution.local import execute_local_run
from hyperspectrum.execution.plan import RunPlan
from hyperspectrum.hyperdata.models import DatasetCandidate
from hyperspectrum.tasks.recommend import ReadinessVerdict

from .test_plan import dataset, plan, savgol, verdict

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/xas/denoising-pairs.npz"
FIXTURE_DIGEST = sha256(FIXTURE.read_bytes()).hexdigest()


def fixture_ids(limit: int = 3) -> tuple[str, ...]:
    with np.load(FIXTURE, allow_pickle=False) as data:
        return tuple(str(value) for value in data["sample_ids"][:limit])


def dataset_for_source(path: Path) -> DatasetCandidate:
    return dataset(content_digest=sha256(path.read_bytes()).hexdigest())


def verdict_for_source(path: Path) -> ReadinessVerdict:
    return verdict().model_copy(
        update={"content_digest": sha256(path.read_bytes()).hexdigest()}
    )


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


def inference_plan(tmp_path: Path, **changes: object) -> tuple[Path, RunPlan]:
    source = noisy_source(tmp_path / "noisy-input.npz")
    values: dict[str, object] = {
        "dataset": dataset_for_source(source),
        "verdict": verdict_for_source(source),
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
    bundle = PredictionBundle.model_validate(predictions_data)
    run_data = json.loads((run_plan.output_directory / "run.json").read_text())
    assert returned == bundle
    assert len(bundle.predictions) == 3
    assert bundle.failures == ()
    assert bundle.provenance["model_digest"] == run_plan.model_digest
    assert bundle.provenance["tool_digest"] == run_plan.tool_digest
    assert bundle.provenance["implementation_digest"] == run_plan.implementation_digest
    assert bundle.provenance["data_digest"] == run_plan.data_digest
    assert bundle.provenance["environment_digest"] == run_plan.environment_digest
    assert bundle.provenance["weight_digest"] == "none"
    assert bundle.provenance["data_origin"] == "synthetic-test"
    assert run_data == {
        "schema_version": "hyperspectrum-run/v1",
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
        "data_digest": run_plan.data_digest,
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


def test_executor_runs_fresh_verified_source_not_a_preimported_callable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Break caught: execution could ignore entrypoint source and call a preimported hardcoded function.
    from hyperspectrum.plugins.xas import baselines

    def wrong_callable(*args: object, **kwargs: object) -> object:
        raise AssertionError("preimported callable must not run")

    monkeypatch.setattr(baselines, "savgol_filter", wrong_callable)

    source, run_plan = inference_plan(tmp_path)
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

    source, run_plan = inference_plan(tmp_path)
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
    source, run_plan = inference_plan(tmp_path)

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
    partial = tmp_path / "partial.npz"
    with np.load(FIXTURE, allow_pickle=False) as data:
        np.savez(
            partial,
            energy=data["energy"][:2],
            noisy=data["noisy"][:2],
            sample_ids=data["sample_ids"][:2],
            group_ids=data["group_ids"][:2],
            energy_unit=data["energy_unit"],
        )
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
    source = tmp_path / "malicious-id.npz"
    np.savez(
        source,
        energy=np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        noisy=np.array([[0.0, 0.2, 0.8, 0.3, 0.1]]),
        sample_ids=np.array([malicious_id]),
        group_ids=np.array(["synthetic"]),
        energy_unit=np.array("eV"),
    )
    run_plan = plan(
        tmp_path,
        max_samples=1,
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
    source = tmp_path / f"forbidden-{forbidden_key}.npz"
    with np.load(FIXTURE, allow_pickle=False) as data:
        values = {
            key: data[key]
            for key in ("energy", "noisy", "sample_ids", "group_ids", "energy_unit")
        }
    values[forbidden_key] = np.array(["must-not-cross-boundary"])
    np.savez(source, **values)
    run_plan = plan(
        tmp_path,
        dataset=dataset_for_source(source),
        verdict=verdict_for_source(source),
    )

    def forbidden_loader(*args: object, **kwargs: object) -> object:
        raise AssertionError("tool code must not load")

    monkeypatch.setattr(local, "_load_savgol_callable", forbidden_loader)

    with pytest.raises(ValueError, match="inference-only"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(1),
            source_npz=source,
        )

    assert not run_plan.output_directory.exists()


def test_separately_digested_noisy_only_source_executes_successfully(
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
    run_plan = plan(
        tmp_path,
        dataset=dataset_for_source(source),
        verdict=verdict_for_source(source),
    )

    bundle = execute_local_run(
        run_plan,
        tool=savgol(),
        selected_sample_ids=fixture_ids(2),
        source_npz=source,
    )

    assert len(bundle.predictions) == 2


def test_structured_noisy_array_is_rejected_before_tool_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "structured-noisy.npz"
    with np.load(FIXTURE, allow_pickle=False) as data:
        noisy = np.empty(
            data["noisy"].shape,
            dtype=[("signal", np.float64), ("ground_truth", np.float64)],
        )
        noisy["signal"] = data["noisy"]
        noisy["ground_truth"] = data["clean"]
        np.savez(
            source,
            energy=data["energy"],
            noisy=noisy,
            sample_ids=data["sample_ids"],
            group_ids=data["group_ids"],
            energy_unit=data["energy_unit"],
        )
    run_plan = plan(
        tmp_path,
        dataset=dataset_for_source(source),
        verdict=verdict_for_source(source),
    )

    def forbidden_loader(*args: object, **kwargs: object) -> object:
        raise AssertionError("tool code must not load")

    monkeypatch.setattr(local, "_load_savgol_callable", forbidden_loader)

    with pytest.raises(ValueError, match="structured"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(1),
            source_npz=source,
        )

    assert not run_plan.output_directory.exists()


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

    with pytest.raises(ValueError, match="SavGol entrypoint"):
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


def test_executor_hashes_source_bytes_and_rejects_a_different_content_digest(
    tmp_path: Path,
) -> None:
    # Break caught: a caller could assert a planned digest while supplying unrelated input bytes.
    run_plan = plan(tmp_path)
    changed = tmp_path / "changed.npz"
    shutil.copyfile(FIXTURE, changed)
    changed.write_bytes(changed.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="data digest"):
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
