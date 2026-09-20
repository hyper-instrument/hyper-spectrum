from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import hyperspectrum
import hyperspectrum.fewshot.solver as solver_module
from hyperspectrum.fewshot.delivery import DELIVERY_FILES
from hyperspectrum.fewshot.knn import GRID_POINTS, PairIndex, predict_many
from hyperspectrum.fewshot.solver import (
    SolverError,
    _select_validation_pool,
    main,
    predict,
    prepare,
    train,
)

ENERGY = np.linspace(-3.0, 30.0, GRID_POINTS, dtype=np.float32)
QUERY_IDS = [
    "sim2exp_0000",
    "exp2sim_0000",
    "cycle_sim_only_0000",
    "cycle_exp_only_0000",
]


def paired_pool(
    prefix: str, atomic_numbers: list[int], edge_codes: list[int], seed: int
) -> dict[str, Any]:
    count = len(atomic_numbers)
    rng = np.random.default_rng(seed)
    return {
        "sample_id": np.asarray([f"{prefix}_{index:04d}" for index in range(count)]),
        "group_id": np.asarray(
            [f"{prefix}-group-{index % 2}" for index in range(count)]
        ),
        "energy": ENERGY,
        "simulation": rng.random((count, GRID_POINTS), dtype=np.float32),
        "experiment": rng.random((count, GRID_POINTS), dtype=np.float32),
        "absorber_atomic_number": np.asarray(atomic_numbers, dtype=np.int16),
        "edge_code": np.asarray(edge_codes, dtype=np.int8),
    }


def query_pool(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "sample_id": np.asarray(QUERY_IDS),
        "direction": np.asarray(["sim2exp", "exp2sim", "sim2exp", "exp2sim"]),
        "energy": ENERGY,
        "spectrum": rng.random((len(QUERY_IDS), GRID_POINTS), dtype=np.float32),
        "absorber_atomic_number": np.asarray([29, 26, 0, 0], dtype=np.int16),
        "edge_code": np.asarray([1, 1, 0, 0], dtype=np.int8),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_release(
    root: Path,
    *,
    grid_points: int = GRID_POINTS,
    grid_step: float = 0.25,
    validation_required: bool = True,
    extra_paired: bool = False,
    validation_energy_offset: float = 0.0,
    queries_energy_offset: float = 0.0,
    shared_id: bool = False,
) -> Path:
    release = root / "release"
    release.mkdir(parents=True)
    pools: dict[str, dict[str, Any]] = {
        "train.npz": paired_pool(
            "train", [29, 29, 26, 26, 29, 26], [1, 1, 1, 1, 2, 2], 1
        ),
        "validation.npz": paired_pool("valid", [29, 26], [1, 1], 2),
    }
    if extra_paired:
        pools["extra_pairs.npz"] = paired_pool("extra", [29, 26], [1, 2], 4)
    if validation_energy_offset:
        pools["validation.npz"]["energy"] = ENERGY + np.float32(
            validation_energy_offset
        )
    if shared_id:
        pools["validation.npz"]["sample_id"] = np.asarray(["train_0000", "valid_0001"])
    required = {"train.npz", *(["validation.npz"] if validation_required else [])}
    for name, pool in pools.items():
        np.savez(release / name, **pool)
    theory_only = {
        "sample_id": np.asarray(["theory_0000"]),
        "absorber": np.asarray(["Cu"]),
        "edge": np.asarray(["K"]),
        "energy": ENERGY,
        "simulation": np.zeros((1, GRID_POINTS), dtype=np.float32),
    }
    np.savez(release / "theory_only_public_pool.npz", **theory_only)
    queries = query_pool(3)
    if queries_energy_offset:
        queries["energy"] = ENERGY + np.float32(queries_energy_offset)
    np.savez(release / "queries.npz", **queries)

    (release / "README.md").write_text("# Release\n")
    files: dict[str, Any] = {}
    for name in (*pools, "theory_only_public_pool.npz", "README.md"):
        path = release / name
        files[name] = {
            "sha256": sha256(path),
            "size": path.stat().st_size,
            "records": len(pools[name]["sample_id"]) if name in pools else None,
            "required": name in required,
        }
    manifest = {
        "schema": "hyperdata-fewshot-release/v1",
        "task": "xas-bidirectional-mapping",
        "release_id": "f" * 32,
        "grid": {"points": grid_points, "start_ev": -3.0, "step_ev": grid_step},
        "counts": {"queries": len(QUERY_IDS)},
        "paired_queries": "disjoint",
        "query_metadata": "full",
        "files": files,
        "queries_sha256": sha256(release / "queries.npz"),
    }
    (release / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    (release / "instruction.md").write_text("# Not read by the solver\n")
    return release


@pytest.fixture
def release(tmp_path: Path) -> Path:
    return build_release(tmp_path)


def run_all(
    release: Path, root: Path, opts: dict[str, str] | None = None
) -> tuple[Path, Path, Path, dict[str, Any]]:
    work, model, out = root / "work", root / "model", root / "out"
    prepare(release, work, opts or {})
    train(work, model, {})
    result = predict(release, work, model, out, {})
    return work, model, out, result


def test_pipeline_delivers_seven_files_with_query_ids_in_order(
    release: Path, tmp_path: Path
) -> None:
    work, model, out, result = run_all(release, tmp_path)

    for name in DELIVERY_FILES:
        assert (out / name).is_file(), name
    assert (work / "prepare.json").is_file()
    assert (model / "index.npz").is_file()
    assert (model / "train.json").is_file()
    assert json.loads((out / "predict.json").read_text()) == result
    assert result["samples"] == 4
    assert result["directions"] == {"sim2exp": 2, "exp2sim": 2}
    assert set(result["files"]) == set(DELIVERY_FILES)

    queries = query_pool(3)
    with np.load(out / "predictions.npz", allow_pickle=False) as loaded:
        assert list(loaded["sample_id"]) == QUERY_IDS
        predictions = loaded["prediction"]
    with np.load(out / "cycle_predictions.npz", allow_pickle=False) as loaded:
        assert list(loaded["sample_id"]) == QUERY_IDS
        cycle = loaded["prediction"]
    assert predictions.shape == cycle.shape == (4, GRID_POINTS)
    assert np.isfinite(predictions).all() and np.isfinite(cycle).all()
    # The cycle is a genuine reverse mapping, not the query echoed back.
    assert not np.array_equal(cycle, queries["spectrum"])
    assert not np.array_equal(cycle, predictions)
    # Positively: the reverse direction applied to our own predictions ...
    index = PairIndex.load(model / "index.npz")
    reversed_direction = np.asarray(["exp2sim", "sim2exp", "exp2sim", "sim2exp"])
    on_predictions = dict(queries, direction=reversed_direction, spectrum=predictions)
    np.testing.assert_array_equal(cycle, predict_many(on_predictions, index))
    # ... and not the reverse direction applied to the query spectra.
    on_queries = dict(queries, direction=reversed_direction)
    assert not np.array_equal(cycle, predict_many(on_queries, index))


def test_data_usage_lists_every_paired_pool_as_influencing(
    release: Path, tmp_path: Path
) -> None:
    _, _, out, _ = run_all(release, tmp_path)
    manifest = json.loads((release / "data_manifest.json").read_text())

    # Both pools are required in the release, and both end up in the final
    # index, so the evaluator's "required => influencing > 0" rule holds.
    usage = json.loads((out / "data_usage.json").read_text())["files"]
    assert usage["train.npz"] == {
        "scanned": 6,
        "influencing": 6,
        "stage": "index",
        "sha256": manifest["files"]["train.npz"]["sha256"],
    }
    assert usage["validation.npz"] == {
        "scanned": 2,
        "influencing": 2,
        "stage": "index",
        "sha256": manifest["files"]["validation.npz"]["sha256"],
    }
    assert "theory_only_public_pool.npz" not in usage

    log_lines = (out / "iteration_log.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in log_lines] == [
        "prepare",
        "index",
        "predict",
        "cycle",
    ]
    summary = json.loads((out / "method_summary.json").read_text())
    assert summary["name"] == "knn-baseline" and summary["neighbors"] == 8
    run_manifest = json.loads((out / "run_manifest.json").read_text())
    assert run_manifest["seed"] == "42"
    report = (out / "method_report.md").read_text()
    assert "validation.npz" in report and "train.npz" in report
    assert "leak-free" in report and "re-admitted" in report


def load_pool(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def mean_absolute_error(
    index: PairIndex, pool: dict[str, Any], direction: str
) -> float:
    source, target = (
        ("simulation", "experiment")
        if direction == "sim2exp"
        else ("experiment", "simulation")
    )
    queries = {
        "direction": np.full(len(pool["sample_id"]), direction),
        "spectrum": pool[source],
        "absorber_atomic_number": pool["absorber_atomic_number"],
        "edge_code": pool["edge_code"],
    }
    predicted = predict_many(queries, index).astype(np.float64)
    return float(np.mean(np.abs(predicted - pool[target].astype(np.float64))))


def test_train_scores_validation_leak_free_then_readmits_it(
    release: Path, tmp_path: Path
) -> None:
    work, model, _, _ = run_all(release, tmp_path)

    prepared = json.loads((work / "prepare.json").read_text())
    assert prepared["release_id"] == "f" * 32
    assert prepared["grid"]["points"] == GRID_POINTS
    assert prepared["validation_pool"] == "validation.npz"
    assert [
        (pool["name"], pool["required"], pool["role"], pool["diagnostic"])
        for pool in prepared["pools"]
    ] == [
        ("train.npz", True, "index", False),
        ("validation.npz", True, "index", True),
    ]
    # Single-domain pools and sidecars are skipped, not staged.
    assert {entry["name"] for entry in prepared["skipped"]} == {
        "theory_only_public_pool.npz",
        "README.md",
    }

    trained = json.loads((model / "train.json").read_text())
    assert trained["seed"] == 42
    assert trained["pools"] == ["train.npz", "validation.npz"]
    assert trained["records"] == 8
    validation = trained["validation"]
    assert validation["name"] == "validation.npz"
    assert validation["records"] == 2
    assert validation["diagnostic"] is True

    # The diagnostic MAE comes from an index WITHOUT the pool: it equals an
    # independent train-only computation and is not the self-match of 0.
    train_only = PairIndex.from_pools([load_pool(release / "train.npz")])
    held_out = load_pool(release / "validation.npz")
    for direction in ("sim2exp", "exp2sim"):
        expected = mean_absolute_error(train_only, held_out, direction)
        assert expected > 0.0
        assert validation[f"mae_{direction}"] == pytest.approx(expected)

    # The final index re-admits every required pool; through it the same
    # pool would self-match exactly (MAE 0), which is why it is not scored so.
    final = PairIndex.load(model / "index.npz")
    assert len(final) == 8
    assert sorted(final.sample_id) == sorted(
        [*train_only.sample_id, *held_out["sample_id"]]
    )
    assert mean_absolute_error(final, held_out, "sim2exp") == 0.0


def test_train_seed_opt_is_recorded(release: Path, tmp_path: Path) -> None:
    work, model = tmp_path / "work", tmp_path / "model"
    prepare(release, work, {})

    train(work, model, {"seed": "7"})

    assert json.loads((model / "train.json").read_text())["seed"] == 7
    with pytest.raises(SolverError, match="seed"):
        train(work, tmp_path / "model-2", {"seed": "seven"})


def test_empty_validation_pool_opt_skips_the_diagnostic(
    release: Path, tmp_path: Path
) -> None:
    work, model, out, _ = run_all(release, tmp_path, {"validation_pool": ""})

    prepared = json.loads((work / "prepare.json").read_text())
    assert prepared["validation_pool"] is None
    assert not any(pool["diagnostic"] for pool in prepared["pools"])
    trained = json.loads((model / "train.json").read_text())
    assert trained["pools"] == ["train.npz", "validation.npz"]
    assert trained["records"] == 8
    assert trained["validation"] is None
    usage = json.loads((out / "data_usage.json").read_text())["files"]
    assert usage["validation.npz"]["influencing"] == 2
    assert usage["validation.npz"]["stage"] == "index"
    assert "No diagnostic pool" in (out / "method_report.md").read_text()


def test_validation_pool_opt_overrides_default(tmp_path: Path) -> None:
    release = build_release(tmp_path, extra_paired=True)
    _, model, _, _ = run_all(release, tmp_path, {"validation_pool": "extra_pairs.npz"})

    trained = json.loads((model / "train.json").read_text())
    assert trained["pools"] == ["train.npz", "validation.npz", "extra_pairs.npz"]
    assert trained["records"] == 10
    assert trained["validation"]["name"] == "extra_pairs.npz"
    assert trained["validation"]["records"] == 2

    with pytest.raises(SolverError, match="nope.npz"):
        prepare(release, tmp_path / "work-2", {"validation_pool": "nope.npz"})

    # A required pool may be the diagnostic pool; it is still re-admitted.
    _, model_2, out_2, _ = run_all(
        release, tmp_path / "second", {"validation_pool": "train.npz"}
    )
    trained_2 = json.loads((model_2 / "train.json").read_text())
    assert trained_2["validation"]["name"] == "train.npz"
    assert trained_2["validation"]["records"] == 6
    expected = mean_absolute_error(
        PairIndex.from_pools(
            [
                load_pool(release / "validation.npz"),
                load_pool(release / "extra_pairs.npz"),
            ]
        ),
        load_pool(release / "train.npz"),
        "exp2sim",
    )
    assert trained_2["validation"]["mae_exp2sim"] == pytest.approx(expected)
    assert trained_2["records"] == 10
    usage = json.loads((out_2 / "data_usage.json").read_text())["files"]
    assert usage["train.npz"]["influencing"] == 6


def test_optional_validation_pool_is_staged_scored_and_readmitted(
    tmp_path: Path,
) -> None:
    release = build_release(tmp_path, validation_required=False)
    work, model, out, _ = run_all(release, tmp_path)

    prepared = json.loads((work / "prepare.json").read_text())
    assert prepared["validation_pool"] == "validation.npz"
    assert {pool["name"]: pool["required"] for pool in prepared["pools"]} == {
        "train.npz": True,
        "validation.npz": False,
    }
    trained = json.loads((model / "train.json").read_text())
    assert trained["pools"] == ["train.npz", "validation.npz"]
    assert trained["validation"]["diagnostic"] is True
    usage = json.loads((out / "data_usage.json").read_text())["files"]
    assert usage["validation.npz"] == {
        "scanned": 2,
        "influencing": 2,
        "stage": "index",
        "sha256": json.loads((release / "data_manifest.json").read_text())["files"][
            "validation.npz"
        ]["sha256"],
    }


def test_diagnostic_pool_needs_another_pool_to_score_against() -> None:
    assert _select_validation_pool(["train.npz", "validation.npz"], {}) == (
        "validation.npz"
    )
    assert _select_validation_pool(["train.npz"], {}) is None
    assert _select_validation_pool(["validation.npz"], {}) is None
    with pytest.raises(SolverError, match="only paired pool"):
        _select_validation_pool(
            ["validation.npz"], {"validation_pool": "validation.npz"}
        )


def test_tampered_pool_makes_prepare_exit_with_sha_message(
    release: Path, tmp_path: Path
) -> None:
    pool = release / "train.npz"
    pool.write_bytes(pool.read_bytes() + b"\0")

    with pytest.raises(SolverError, match="sha256.*train.npz") as excinfo:
        prepare(release, tmp_path / "work", {})
    assert excinfo.value.exit_status == 1


def test_grid_other_than_133_points_is_refused(tmp_path: Path) -> None:
    release = build_release(tmp_path, grid_points=100)

    with pytest.raises(SolverError, match="grid"):
        prepare(release, tmp_path / "work", {})


def test_prepare_rejects_energy_axes_off_the_manifest_grid(tmp_path: Path) -> None:
    shifted = build_release(tmp_path / "shifted", validation_energy_offset=0.5)
    with pytest.raises(SolverError, match="validation.npz energy"):
        prepare(shifted, tmp_path / "work-shifted", {})

    # The pools agree with each other but not with the manifest's step.
    wrong_step = build_release(tmp_path / "step", grid_step=0.3)
    with pytest.raises(SolverError, match="train.npz energy.*manifest grid"):
        prepare(wrong_step, tmp_path / "work-step", {})


def test_predict_rejects_query_energy_off_the_release_grid(tmp_path: Path) -> None:
    release = build_release(tmp_path, queries_energy_offset=0.25)
    work, model = tmp_path / "work", tmp_path / "model"
    prepare(release, work, {})
    train(work, model, {})

    with pytest.raises(SolverError, match="queries.npz energy"):
        predict(release, work, model, tmp_path / "out", {})


def test_prepare_rejects_sample_ids_shared_across_pools(tmp_path: Path) -> None:
    release = build_release(tmp_path, shared_id=True)

    with pytest.raises(SolverError, match="duplicate sample ids.*train_0000"):
        prepare(release, tmp_path / "work", {})


def test_predict_verifies_queries_sha_and_index_size(
    release: Path, tmp_path: Path
) -> None:
    work, model, _, _ = run_all(release, tmp_path)

    trained = json.loads((model / "train.json").read_text())
    trained["records"] = 99
    (model / "train.json").write_text(json.dumps(trained))
    with pytest.raises(SolverError, match="99.*rerun train"):
        predict(release, work, model, tmp_path / "out-2", {})
    trained["records"] = 8
    (model / "train.json").write_text(json.dumps(trained))

    queries = release / "queries.npz"
    queries.write_bytes(queries.read_bytes() + b"\0")
    with pytest.raises(SolverError, match="sha256 mismatch for queries.npz"):
        predict(release, work, model, tmp_path / "out-3", {})


def test_malformed_prepare_json_is_reported_without_traceback(
    release: Path, tmp_path: Path
) -> None:
    work, model, _, _ = run_all(release, tmp_path)
    prepared = json.loads((work / "prepare.json").read_text())
    del prepared["pools"][0]["records"]
    (work / "prepare.json").write_text(json.dumps(prepared))

    with pytest.raises(SolverError, match="rerun prepare"):
        train(work, tmp_path / "model-2", {})
    with pytest.raises(SolverError, match="rerun prepare"):
        predict(release, work, model, tmp_path / "out-2", {})

    result = module_command(
        "predict",
        "--release-dir",
        str(release),
        "--work-dir",
        str(work),
        "--model-dir",
        str(model),
        "--out-dir",
        str(tmp_path / "out-3"),
    )
    assert result.returncode == 1
    assert "rerun prepare" in result.stderr
    assert "Traceback" not in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1
    assert result.stdout == ""


def test_predict_rejects_manifest_without_files_object(
    release: Path, tmp_path: Path
) -> None:
    work, model, _, _ = run_all(release, tmp_path)
    manifest = json.loads((release / "data_manifest.json").read_text())
    del manifest["files"]
    (release / "data_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(SolverError, match="files"):
        predict(release, work, model, tmp_path / "out-2", {})


def test_main_backstop_reports_unexpected_errors_on_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom\nsecond line")

    monkeypatch.setattr(solver_module, "prepare", boom)

    status = main(
        ["prepare", "--release-dir", str(tmp_path), "--work-dir", str(tmp_path / "w")]
    )

    assert status == 1
    captured = capsys.readouterr()
    assert captured.err == "prepare: RuntimeError: boom second line\n"
    assert captured.out == ""


def test_unknown_opt_exits_2_naming_the_key(release: Path, tmp_path: Path) -> None:
    work, model = tmp_path / "work", tmp_path / "model"

    with pytest.raises(SolverError, match="bogus") as excinfo:
        prepare(release, work, {"bogus": "1"})
    assert excinfo.value.exit_status == 2

    prepare(release, work, {})
    with pytest.raises(SolverError, match="validation_pool") as excinfo:
        train(work, model, {"validation_pool": "train.npz"})
    assert excinfo.value.exit_status == 2

    train(work, model, {})
    with pytest.raises(SolverError, match="seed") as excinfo:
        predict(release, work, model, tmp_path / "out", {"seed": "1"})
    assert excinfo.value.exit_status == 2


def test_predict_refuses_model_from_another_release(
    release: Path, tmp_path: Path
) -> None:
    work, model, _, _ = run_all(release, tmp_path)
    other = build_release(tmp_path / "other")
    manifest = json.loads((other / "data_manifest.json").read_text())
    manifest["release_id"] = "e" * 32
    (other / "data_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(SolverError, match="release_id"):
        predict(other, work, model, tmp_path / "out-2", {})


def module_command(*arguments: str) -> subprocess.CompletedProcess[str]:
    package_root = Path(hyperspectrum.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(package_root)}
    return subprocess.run(
        [sys.executable, "-m", "hyperspectrum.fewshot.solver", *arguments],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_module_cli_runs_all_three_steps_and_prints_json_last(
    release: Path, tmp_path: Path
) -> None:
    work, model, out = tmp_path / "work", tmp_path / "model", tmp_path / "out"

    prepared = module_command(
        "prepare", "--release-dir", str(release), "--work-dir", str(work)
    )
    assert prepared.returncode == 0, prepared.stderr
    last_line = json.loads(prepared.stdout.splitlines()[-1])
    assert last_line == json.loads((work / "prepare.json").read_text())
    assert last_line["release_id"] == "f" * 32

    trained = module_command(
        "train", "--work-dir", str(work), "--model-dir", str(model), "--opt", "seed=9"
    )
    assert trained.returncode == 0, trained.stderr
    assert json.loads(trained.stdout.splitlines()[-1])["seed"] == 9

    predicted = module_command(
        "predict",
        "--release-dir",
        str(release),
        "--work-dir",
        str(work),
        "--model-dir",
        str(model),
        "--out-dir",
        str(out),
    )
    assert predicted.returncode == 0, predicted.stderr
    assert json.loads(predicted.stdout.splitlines()[-1])["samples"] == 4
    for name in DELIVERY_FILES:
        assert (out / name).is_file(), name
    assert json.loads((out / "run_manifest.json").read_text())["seed"] == "9"


def test_module_cli_reports_expected_errors_without_traceback(
    release: Path, tmp_path: Path
) -> None:
    unknown = module_command(
        "prepare",
        "--release-dir",
        str(release),
        "--work-dir",
        str(tmp_path / "work"),
        "--opt",
        "bogus=1",
    )
    assert unknown.returncode == 2
    assert "bogus" in unknown.stderr
    assert "Traceback" not in unknown.stderr
    assert len(unknown.stderr.strip().splitlines()) == 1

    malformed = module_command(
        "prepare",
        "--release-dir",
        str(release),
        "--work-dir",
        str(tmp_path / "work"),
        "--opt",
        "no-equals-sign",
    )
    assert malformed.returncode == 2
    assert "no-equals-sign" in malformed.stderr

    pool = release / "train.npz"
    pool.write_bytes(pool.read_bytes() + b"\0")
    tampered = module_command(
        "prepare", "--release-dir", str(release), "--work-dir", str(tmp_path / "work")
    )
    assert tampered.returncode == 1
    assert "sha256" in tampered.stderr and "train.npz" in tampered.stderr
    assert "Traceback" not in tampered.stderr
    assert len(tampered.stderr.strip().splitlines()) == 1
