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
from hyperspectrum.fewshot.delivery import DELIVERY_FILES
from hyperspectrum.fewshot.knn import GRID_POINTS, PairIndex
from hyperspectrum.fewshot.solver import SolverError, predict, prepare, train

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
    validation_required: bool = False,
    extra_paired: bool = False,
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
    np.savez(release / "queries.npz", **query_pool(3))

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
        "grid": {"points": grid_points, "start_ev": -3.0, "step_ev": 0.25},
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


def test_data_usage_lists_index_pool_as_influencing_and_validation_as_not(
    release: Path, tmp_path: Path
) -> None:
    _, _, out, _ = run_all(release, tmp_path)
    manifest = json.loads((release / "data_manifest.json").read_text())

    usage = json.loads((out / "data_usage.json").read_text())["files"]
    assert usage["train.npz"] == {
        "scanned": 6,
        "influencing": 6,
        "stage": "index",
        "sha256": manifest["files"]["train.npz"]["sha256"],
    }
    assert usage["validation.npz"] == {
        "scanned": 2,
        "influencing": 0,
        "stage": "validation",
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


def test_train_excludes_validation_pool_and_reports_its_mae(
    release: Path, tmp_path: Path
) -> None:
    work, model, _, _ = run_all(release, tmp_path)

    prepared = json.loads((work / "prepare.json").read_text())
    assert prepared["release_id"] == "f" * 32
    assert prepared["grid"]["points"] == GRID_POINTS
    assert prepared["validation_pool"] == "validation.npz"
    assert {pool["name"]: pool["role"] for pool in prepared["pools"]} == {
        "train.npz": "index",
        "validation.npz": "validation",
    }
    assert {pool["name"]: pool["required"] for pool in prepared["pools"]} == {
        "train.npz": True,
        "validation.npz": False,
    }
    # Single-domain pools and sidecars are skipped, not staged.
    assert {entry["name"] for entry in prepared["skipped"]} == {
        "theory_only_public_pool.npz",
        "README.md",
    }

    trained = json.loads((model / "train.json").read_text())
    assert trained["seed"] == 42
    assert trained["pools"] == ["train.npz"]
    assert trained["records"] == 6
    validation = trained["validation"]
    assert validation["name"] == "validation.npz"
    assert validation["mae_sim2exp"] >= 0.0 and validation["mae_exp2sim"] >= 0.0

    index = PairIndex.load(model / "index.npz")
    assert len(index) == 6
    assert all(sample_id.startswith("train_") for sample_id in index.sample_id)


def test_train_seed_opt_is_recorded(release: Path, tmp_path: Path) -> None:
    work, model = tmp_path / "work", tmp_path / "model"
    prepare(release, work, {})

    train(work, model, {"seed": "7"})

    assert json.loads((model / "train.json").read_text())["seed"] == 7
    with pytest.raises(SolverError, match="seed"):
        train(work, tmp_path / "model-2", {"seed": "seven"})


def test_empty_validation_pool_opt_indexes_every_paired_pool(
    release: Path, tmp_path: Path
) -> None:
    work, model, out, _ = run_all(release, tmp_path, {"validation_pool": ""})

    assert json.loads((work / "prepare.json").read_text())["validation_pool"] is None
    trained = json.loads((model / "train.json").read_text())
    assert trained["pools"] == ["train.npz", "validation.npz"]
    assert trained["records"] == 8
    assert trained["validation"] is None
    usage = json.loads((out / "data_usage.json").read_text())["files"]
    assert usage["validation.npz"]["influencing"] == 2
    assert usage["validation.npz"]["stage"] == "index"


def test_validation_pool_opt_overrides_default(tmp_path: Path) -> None:
    release = build_release(tmp_path, extra_paired=True)
    _, model, _, _ = run_all(release, tmp_path, {"validation_pool": "extra_pairs.npz"})

    trained = json.loads((model / "train.json").read_text())
    assert trained["pools"] == ["train.npz", "validation.npz"]
    assert trained["records"] == 8
    assert trained["validation"]["name"] == "extra_pairs.npz"

    with pytest.raises(SystemExit, match="nope.npz"):
        prepare(release, tmp_path / "work-2", {"validation_pool": "nope.npz"})
    # A required pool must influence the delivery, so it cannot be held out.
    with pytest.raises(SystemExit, match="train.npz.*required"):
        prepare(release, tmp_path / "work-3", {"validation_pool": "train.npz"})


def test_required_validation_pool_stays_in_the_index_by_default(
    tmp_path: Path,
) -> None:
    release = build_release(tmp_path, validation_required=True)
    work, model, out, _ = run_all(release, tmp_path)

    assert json.loads((work / "prepare.json").read_text())["validation_pool"] is None
    trained = json.loads((model / "train.json").read_text())
    assert trained["pools"] == ["train.npz", "validation.npz"]
    assert trained["validation"] is None
    usage = json.loads((out / "data_usage.json").read_text())["files"]
    assert usage["validation.npz"]["influencing"] == 2

    with pytest.raises(SystemExit, match="validation.npz.*required"):
        prepare(release, tmp_path / "work-2", {"validation_pool": "validation.npz"})


def test_tampered_pool_makes_prepare_exit_with_sha_message(
    release: Path, tmp_path: Path
) -> None:
    pool = release / "train.npz"
    pool.write_bytes(pool.read_bytes() + b"\0")

    with pytest.raises(SystemExit, match="sha256.*train.npz") as excinfo:
        prepare(release, tmp_path / "work", {})
    assert excinfo.value.exit_status == 1
    assert "Traceback" not in str(excinfo.value)


def test_grid_other_than_133_points_is_refused(tmp_path: Path) -> None:
    release = build_release(tmp_path, grid_points=100)

    with pytest.raises(SystemExit, match="grid"):
        prepare(release, tmp_path / "work", {})


def test_unknown_opt_exits_2_naming_the_key(release: Path, tmp_path: Path) -> None:
    work, model = tmp_path / "work", tmp_path / "model"

    with pytest.raises(SystemExit, match="bogus") as excinfo:
        prepare(release, work, {"bogus": "1"})
    assert excinfo.value.exit_status == 2

    prepare(release, work, {})
    with pytest.raises(SystemExit, match="validation_pool") as excinfo:
        train(work, model, {"validation_pool": "train.npz"})
    assert excinfo.value.exit_status == 2

    train(work, model, {})
    with pytest.raises(SystemExit, match="seed") as excinfo:
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

    with pytest.raises(SystemExit, match="release_id"):
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
