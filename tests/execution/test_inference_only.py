"""The inference-only sibling of the ACE execution facade.

Every test here is about one boundary: this entry point is handed a single
noisy-input asset and no ground truth, and it must never be able to reach one.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from hyperspectrum.execution import execute_ace_xas_inference_only

SAVGOL_PARAMETERS = {"window_length": 5, "polyorder": 2}
DATASET_CODE = "zenodo-10159154"
DATASET_VERSION = "synthetic-version-id"


def _rows(count: int = 3, points: int = 21) -> dict[str, np.ndarray]:
    energy = np.stack(
        [np.linspace(8_800.0, 9_000.0, points) for _ in range(count)]
    )
    noisy = np.stack(
        [
            np.sin(np.linspace(0.0, 3.0, points)) + 0.01 * index
            for index in range(count)
        ]
    )
    return {
        "energy": energy,
        "energy_unit": np.asarray("eV"),
        "group_ids": np.asarray(
            [f"High-Cu_exposure_g{index}/seg" for index in range(count)]
        ),
        "noisy": noisy,
        "sample_ids": np.asarray(
            [f"data_txt/High-Cu_exposure_g{index}/{index}_at_200C_seg" for index in range(count)]
        ),
    }


def _inference_asset(path: Path, **overrides: np.ndarray) -> Path:
    arrays = {**_rows(), **overrides}
    np.savez(path, **arrays)
    return path


def _execute(tmp_path: Path, asset: Path, **changes: object) -> object:
    request: dict[str, object] = {
        "inference_asset": asset,
        "output_directory": tmp_path / "run",
        "task_id": "xas-denoising",
        "tool_id": "savgol",
        "parameters": SAVGOL_PARAMETERS,
        "dataset_code": DATASET_CODE,
        "dataset_version": DATASET_VERSION,
    }
    request.update(changes)
    return execute_ace_xas_inference_only(**request)  # type: ignore[arg-type]


def test_facade_answers_every_mounted_row_from_the_asset_alone(tmp_path: Path) -> None:
    # Break caught: a candidate that answered a subset, or that needed a
    # benchmark directory to answer at all, cannot be run against the one file a
    # dual-asset candidate mount contains.
    asset = _inference_asset(tmp_path / "inference.npz")
    expected_ids = [str(value) for value in _rows()["sample_ids"]]

    result = _execute(tmp_path, asset)

    assert result.answered_sample_ids == tuple(expected_ids)
    assert result.inference_sha256 == sha256(asset.read_bytes()).hexdigest()
    assert result.n_samples == len(expected_ids)
    assert result.bundle.schema_version == "hyperspectrum-prediction/v2"
    assert not result.bundle.failures

    bundle = json.loads((tmp_path / "run/predictions.json").read_text(encoding="utf-8"))
    run = json.loads((tmp_path / "run/run.json").read_text(encoding="utf-8"))
    provenance = bundle["provenance"]
    # The asset's own digest is the plan's data identity. There is no benchmark
    # asset digest to name, which is the point: this plan shape cannot carry one.
    assert provenance["data_digest"] == result.inference_sha256
    assert provenance["plan_schema_version"] == "hyperspectrum-run-plan/v2"
    assert provenance["dataset_code"] == DATASET_CODE
    assert provenance["dataset_version"] == DATASET_VERSION
    assert provenance["weight_digest"] == "none"
    assert bundle["run_id"] == f"run-{result.plan.plan_digest}"
    assert run["schema_version"] == "hyperspectrum-run/v1"
    assert run["status"] == "completed"
    assert run["planned_sample_count"] == len(expected_ids)
    assert run["data_digest"] == result.inference_sha256
    assert all(path.is_file() for path in result.artifact_paths)


def test_facade_ignores_a_benchmark_sitting_beside_the_asset(tmp_path: Path) -> None:
    # Break caught: the self-scored sibling walks a directory for a benchmark. If
    # any of that reached this path, a mount that happened to carry the answers
    # would change what the candidate produced — and the whole boundary is that
    # it cannot.
    isolated = _inference_asset(tmp_path / "alone.npz")
    crowded = tmp_path / "crowded"
    crowded.mkdir()
    _inference_asset(crowded / "inference.npz")
    (crowded / "manifest.json").write_text("{}", encoding="utf-8")
    (crowded / "split.json").write_text("{}", encoding="utf-8")
    np.savez(
        crowded / "benchmark.npz",
        proxy_full_count=_rows()["noisy"] * 0.0,
        splits=np.asarray(["test", "train", "val"]),
    )

    isolated_result = _execute(tmp_path, isolated, output_directory=tmp_path / "a")
    crowded_result = _execute(
        tmp_path, crowded / "inference.npz", output_directory=tmp_path / "b"
    )

    assert isolated_result.answered_sample_ids == crowded_result.answered_sample_ids
    for left, right in zip(
        sorted((tmp_path / "a/arrays").iterdir()),
        sorted((tmp_path / "b/arrays").iterdir()),
        strict=True,
    ):
        assert left.read_bytes() == right.read_bytes()


def test_facade_refuses_an_asset_that_carries_a_target(tmp_path: Path) -> None:
    # Break caught: an asset with a target is a scorer asset, and a facade that
    # quietly accepted one would be a candidate marking its own work.
    asset = tmp_path / "leaky.npz"
    np.savez(asset, **{**_rows(), "proxy_full_count": _rows()["noisy"]})

    with pytest.raises(ValueError, match="exact inference-only input contract"):
        _execute(tmp_path, asset)


def test_facade_refuses_an_asset_that_declares_splits(tmp_path: Path) -> None:
    # Break caught: `splits` is a scorer-only array. Its presence means the mount
    # was built from the scorer half, which is the failure this whole boundary
    # exists to make impossible rather than unlikely.
    asset = tmp_path / "split-aware.npz"
    np.savez(
        asset, **{**_rows(), "splits": np.asarray(["test", "train", "val"])}
    )

    with pytest.raises(ValueError, match="exact inference-only input contract"):
        _execute(tmp_path, asset)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"task_id": "xas-classification"}, "only task 'xas-denoising'"),
        ({"tool_id": "unknown-tool"}, "unknown tool"),
        ({"max_samples": 0}, "max_samples"),
        ({"dataset_code": "  "}, "dataset identity"),
        ({"dataset_version": ""}, "dataset identity"),
    ],
)
def test_facade_refuses_identity_it_cannot_stand_behind(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    asset = _inference_asset(tmp_path / "inference.npz")

    with pytest.raises(ValueError, match=message):
        _execute(tmp_path, asset, **changes)

    assert not (tmp_path / "run").exists()


def test_facade_refuses_mounted_weights_for_an_unweighted_tool(tmp_path: Path) -> None:
    asset = _inference_asset(tmp_path / "inference.npz")
    weight = tmp_path / "weights.pt"
    weight.write_bytes(b"not a checkpoint")

    with pytest.raises(ValueError, match="unweighted tools"):
        _execute(tmp_path, asset, weight_files=(weight,))


def test_facade_caps_the_answer_at_max_samples(tmp_path: Path) -> None:
    # Kept symmetric with the self-scored sibling: a capped run is a partial
    # answer, and it is ACE's completeness check — not this facade — that decides
    # whether a partial answer is admissible for a given mount.
    asset = _inference_asset(tmp_path / "inference.npz")

    result = _execute(tmp_path, asset, max_samples=2)

    assert len(result.answered_sample_ids) == 2
    assert result.answered_sample_ids == tuple(
        str(value) for value in _rows()["sample_ids"][:2]
    )
