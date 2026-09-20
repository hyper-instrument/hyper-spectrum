from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hyperspectrum.fewshot.delivery import (
    DELIVERY_FILES,
    LIMITS,
    Usage,
    write_delivery,
)
from hyperspectrum.fewshot.knn import GRID_POINTS

TRAIN_SHA = "a" * 64
VALIDATION_SHA = "b" * 64
AUX_SHA = "c" * 64


def manifest() -> dict[str, Any]:
    return {
        "schema": "xas-fewshot/v1",
        "release_id": "0" * 32,
        "files": {
            "train.npz": {
                "sha256": TRAIN_SHA,
                "size": 100,
                "records": 6,
                "required": True,
            },
            "validation.npz": {
                "sha256": VALIDATION_SHA,
                "size": 40,
                "records": 2,
                "required": False,
            },
            "auxiliary_experiments.npz": {
                "sha256": AUX_SHA,
                "size": 10,
                "records": None,
                "required": False,
            },
        },
    }


def usage() -> dict[str, Usage]:
    return {
        "train.npz": Usage(scanned=6, influencing=6, stage="index"),
        "validation.npz": Usage(scanned=2, influencing=0, stage="validation"),
    }


def log(count: int = 4) -> list[dict[str, Any]]:
    return [{"step": f"step-{index}", "count": index} for index in range(count)]


SAMPLE_IDS = np.asarray(["sim2exp_0001", "exp2sim_0000", "cycle_sim_only_0000"])


def predictions(value: float = 0.5) -> np.ndarray:
    return np.full((len(SAMPLE_IDS), GRID_POINTS), value, dtype=np.float32)


def write(out_dir: Path, **overrides: Any) -> dict[str, int]:
    arguments: dict[str, Any] = {
        "manifest": manifest(),
        "usage": usage(),
        "sample_ids": SAMPLE_IDS,
        "predictions": predictions(0.5),
        "cycle_predictions": predictions(0.25),
        "method": {"name": "knn-baseline", "neighbors": 8},
        "report_markdown": "# Method\n\nk-NN baseline.\n",
        "seed": 42,
        "log": log(),
    }
    arguments.update(overrides)
    return write_delivery(out_dir, **arguments)


def test_writes_all_seven_files_and_reports_their_sizes(tmp_path: Path) -> None:
    sizes = write(tmp_path)

    assert set(sizes) == set(DELIVERY_FILES)
    assert set(DELIVERY_FILES) == set(LIMITS)
    for name, size in sizes.items():
        assert (tmp_path / name).is_file()
        assert (tmp_path / name).stat().st_size == size
        assert size > 0


def test_npz_files_carry_exactly_sample_id_and_prediction_in_query_order(
    tmp_path: Path,
) -> None:
    write(tmp_path)

    for name, value in (("predictions.npz", 0.5), ("cycle_predictions.npz", 0.25)):
        with zipfile.ZipFile(tmp_path / name) as archive:
            assert sorted(archive.namelist()) == ["prediction.npy", "sample_id.npy"]
        with np.load(tmp_path / name, allow_pickle=False) as loaded:
            assert list(loaded["sample_id"]) == list(SAMPLE_IDS)
            assert loaded["sample_id"].dtype.kind == "U"
            assert loaded["prediction"].dtype == np.float32
            assert loaded["prediction"].shape == (len(SAMPLE_IDS), GRID_POINTS)
            np.testing.assert_array_equal(loaded["prediction"], value)


def test_data_usage_takes_sha_from_manifest(tmp_path: Path) -> None:
    write(tmp_path)

    usage_json = json.loads((tmp_path / "data_usage.json").read_text())
    assert usage_json == {
        "files": {
            "train.npz": {
                "scanned": 6,
                "influencing": 6,
                "stage": "index",
                "sha256": TRAIN_SHA,
            },
            "validation.npz": {
                "scanned": 2,
                "influencing": 0,
                "stage": "validation",
                "sha256": VALIDATION_SHA,
            },
        }
    }


def test_optional_pool_with_unknown_record_count_may_be_listed(
    tmp_path: Path,
) -> None:
    listed = usage()
    listed["auxiliary_experiments.npz"] = Usage(
        scanned=5, influencing=0, stage="ignored"
    )

    write(tmp_path, usage=listed)

    usage_json = json.loads((tmp_path / "data_usage.json").read_text())
    assert usage_json["files"]["auxiliary_experiments.npz"]["sha256"] == AUX_SHA


def test_usage_rules_are_enforced(tmp_path: Path) -> None:
    unknown = usage()
    unknown["mystery.npz"] = Usage(scanned=1, influencing=1, stage="index")
    with pytest.raises(ValueError, match="mystery.npz"):
        write(tmp_path, usage=unknown)

    missing_required = usage()
    del missing_required["train.npz"]
    with pytest.raises(ValueError, match="train.npz"):
        write(tmp_path, usage=missing_required)

    optional_unlisted = usage()
    del optional_unlisted["validation.npz"]
    write(tmp_path, usage=optional_unlisted)

    required_without_influence = usage()
    required_without_influence["train.npz"] = Usage(
        scanned=6, influencing=0, stage="index"
    )
    with pytest.raises(ValueError, match="train.npz"):
        write(tmp_path, usage=required_without_influence)

    over_scanned = usage()
    over_scanned["train.npz"] = Usage(scanned=7, influencing=6, stage="index")
    with pytest.raises(ValueError, match="records"):
        write(tmp_path, usage=over_scanned)


def test_usage_rejects_negative_or_inverted_counts_and_non_string_stage() -> None:
    with pytest.raises(ValueError, match="influencing"):
        Usage(scanned=1, influencing=2, stage="index")
    with pytest.raises(ValueError, match="scanned"):
        Usage(scanned=-1, influencing=-1, stage="index")
    with pytest.raises(TypeError, match="stage"):
        Usage(scanned=1, influencing=1, stage=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="scanned"):
        Usage(scanned=1.5, influencing=1, stage="index")  # type: ignore[arg-type]


def test_iteration_log_has_one_json_object_per_line(tmp_path: Path) -> None:
    write(tmp_path, log=log(5))

    lines = (tmp_path / "iteration_log.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == log(5)


def test_fewer_than_four_log_entries_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="iteration_log"):
        write(tmp_path, log=log(3))


def test_run_manifest_has_string_seed_and_dependency_array(tmp_path: Path) -> None:
    write(tmp_path, seed=7)

    run_manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert run_manifest["seed"] == "7"
    assert isinstance(run_manifest["dependencies"], list)
    assert f"numpy=={np.__version__}" in run_manifest["dependencies"]
    assert any(
        dependency.startswith("hyperspectrum==")
        for dependency in run_manifest["dependencies"]
    )
    assert run_manifest["solver"].startswith("hyperspectrum.fewshot/")


def test_method_summary_and_report_are_written_verbatim(tmp_path: Path) -> None:
    write(tmp_path, method={"name": "x", "k": 3}, report_markdown="# Report\n")

    assert json.loads((tmp_path / "method_summary.json").read_text()) == {
        "name": "x",
        "k": 3,
    }
    assert (tmp_path / "method_report.md").read_text() == "# Report\n"


def test_non_finite_and_oversized_predictions_are_rejected(tmp_path: Path) -> None:
    nan = predictions()
    nan[1, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        write(tmp_path, predictions=nan)

    huge = predictions()
    huge[0, 0] = 2e6
    with pytest.raises(ValueError, match="1e\\+?0?6|1000000"):
        write(tmp_path, cycle_predictions=huge)


def test_prediction_shape_must_match_sample_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="predictions"):
        write(tmp_path, predictions=predictions()[:-1])
    with pytest.raises(ValueError, match="cycle_predictions"):
        write(tmp_path, cycle_predictions=predictions()[:, :-1])


def test_byte_limit_violation_names_the_file(tmp_path: Path) -> None:
    oversized = log()
    oversized[0]["payload"] = "x" * (LIMITS["iteration_log.jsonl"] + 1)

    with pytest.raises(ValueError, match="iteration_log.jsonl"):
        write(tmp_path, log=oversized)
