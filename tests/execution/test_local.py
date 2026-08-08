"""Focused integration tests for atomic local Savitzky-Golay smoke runs."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from hyperspectrum.contracts import PredictionBundle
from hyperspectrum.execution.local import MaterializedInputSet, execute_local_run
from hyperspectrum.plugins.xas.arrays import XASSpectrum

from .test_plan import plan, savgol

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/xas/denoising-pairs.npz"
FIXTURE_DIGEST = sha256(FIXTURE.read_bytes()).hexdigest()


def fixture_ids(limit: int = 3) -> tuple[str, ...]:
    with np.load(FIXTURE, allow_pickle=False) as data:
        return tuple(str(value) for value in data["sample_ids"][:limit])


def materialize_fixture(selected_ids: tuple[str, ...]) -> MaterializedInputSet:
    """Read only noisy inputs and explicit coordinates from the synthetic fixture."""
    with np.load(FIXTURE, allow_pickle=False) as data:
        sample_ids = [str(value) for value in data["sample_ids"]]
        index_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        return MaterializedInputSet(
            data_digest=FIXTURE_DIGEST,
            spectra=tuple(
                XASSpectrum(
                    sample_id=sample_id,
                    group_id=str(data["group_ids"][index_by_id[sample_id]]),
                    energy=data["energy"][index_by_id[sample_id]],
                    intensity=data["noisy"][index_by_id[sample_id]],
                    energy_unit=str(data["energy_unit"]),  # type: ignore[arg-type]
                )
                for sample_id in selected_ids
            ),
        )


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
    run_plan = plan(tmp_path)
    selected = fixture_ids()

    returned = execute_local_run(
        run_plan,
        tool=savgol(),
        selected_sample_ids=selected,
        materialize=materialize_fixture,
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


@pytest.mark.parametrize("partial", [(), fixture_ids(2)])
def test_partial_input_materialization_aborts_without_a_final_run_directory(
    tmp_path: Path, partial: tuple[str, ...]
) -> None:
    # Break caught: execution could silently filter missing selected samples and publish partial output.
    run_plan = plan(tmp_path)

    def materialize(_: tuple[str, ...]) -> MaterializedInputSet:
        return materialize_fixture(partial)

    with pytest.raises(ValueError, match="complete selected input set"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            materialize=materialize,
        )

    assert not run_plan.output_directory.exists()
    assert not tuple(tmp_path.glob(".run.*"))


def test_materialization_exception_aborts_without_a_final_run_directory(
    tmp_path: Path,
) -> None:
    # Break caught: a source read error could leave a directory that looks like a completed run.
    run_plan = plan(tmp_path)

    def fail(_: tuple[str, ...]) -> MaterializedInputSet:
        raise OSError("fixture read failed")

    with pytest.raises(OSError, match="fixture read failed"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            materialize=fail,
        )

    assert not run_plan.output_directory.exists()
    assert not tuple(tmp_path.glob(".run.*"))


def test_model_failures_are_recorded_one_for_one_without_disappearing(
    tmp_path: Path,
) -> None:
    # Break caught: sample-level model failures could be dropped or abort an otherwise complete run record.
    run_plan = plan(tmp_path, parameters={"window_length": 4, "polyorder": 2})

    bundle = execute_local_run(
        run_plan,
        tool=savgol(),
        selected_sample_ids=fixture_ids(),
        materialize=materialize_fixture,
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
    run_plan = plan(tmp_path, max_samples=1)
    malicious_id = "../../escape"

    def materialize(_: tuple[str, ...]) -> MaterializedInputSet:
        return MaterializedInputSet(
            data_digest=FIXTURE_DIGEST,
            spectra=(
                XASSpectrum(
                    sample_id=malicious_id,
                    group_id="synthetic",
                    energy=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                    intensity=np.array([0.0, 0.2, 0.8, 0.3, 0.1]),
                    energy_unit="eV",
                ),
            ),
        )

    bundle = execute_local_run(
        run_plan,
        tool=savgol(),
        selected_sample_ids=(malicious_id,),
        materialize=materialize,
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
    called = False

    def materialize(_: tuple[str, ...]) -> MaterializedInputSet:
        nonlocal called
        called = True
        return materialize_fixture(fixture_ids())

    with pytest.raises(ValueError, match="dry-run"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            materialize=materialize,
        )

    assert called is False
    assert not run_plan.output_directory.exists()


def test_executor_refuses_to_overwrite_an_existing_run_directory(
    tmp_path: Path,
) -> None:
    # Break caught: rerunning a plan could replace immutable prior evidence.
    run_plan = plan(tmp_path)
    run_plan.output_directory.mkdir()
    marker = run_plan.output_directory / "keep.txt"
    marker.write_text("original")
    called = False

    def materialize(_: tuple[str, ...]) -> MaterializedInputSet:
        nonlocal called
        called = True
        return materialize_fixture(fixture_ids())

    with pytest.raises(FileExistsError):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            materialize=materialize,
        )

    assert called is False
    assert marker.read_text() == "original"


def test_executor_rejects_materialized_data_with_a_different_content_digest(
    tmp_path: Path,
) -> None:
    # Break caught: a materializer could silently serve different bytes than the immutable plan selected.
    run_plan = plan(tmp_path)
    materialized = materialize_fixture(fixture_ids())

    def materialize(_: tuple[str, ...]) -> MaterializedInputSet:
        return MaterializedInputSet(data_digest="f" * 64, spectra=materialized.spectra)

    with pytest.raises(ValueError, match="data digest"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            materialize=materialize,
        )

    assert not run_plan.output_directory.exists()


def test_executor_rejects_a_manifest_that_does_not_match_the_immutable_plan(
    tmp_path: Path,
) -> None:
    # Break caught: an executor could run changed tool semantics under an old provenance digest.
    run_plan = plan(tmp_path).model_copy(update={"tool_digest": "f" * 64})
    called = False

    def materialize(_: tuple[str, ...]) -> MaterializedInputSet:
        nonlocal called
        called = True
        return materialize_fixture(fixture_ids())

    with pytest.raises(ValueError, match="tool digest"):
        execute_local_run(
            run_plan,
            tool=savgol(),
            selected_sample_ids=fixture_ids(),
            materialize=materialize,
        )

    assert called is False
    assert not run_plan.output_directory.exists()
