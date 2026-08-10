"""The vendored adapter, driven by path against this checkout's runtime.

There is no Docker on the machines this suite runs on, so "the image works"
cannot be demonstrated here. What can be demonstrated is the half that actually
carries risk: the vendored copies and `src/hyperspectrum/` are two things that
move independently, and the image is the only place they meet. A candidate
mount is assembled on disk exactly as ACE publishes one, `run.py` is imported
the way `/opt/acebench` arranges for it to be, and it is left to call the real
inference-only facade.

That is a narrower claim than a built image and a much cheaper one to keep
true. What it does not cover is the layer underneath — the interpreter digest
and the frozen dependency lock — which is why
`test_ace_candidate_image.py` exists beside this file.

`answer.py` is driven directly as well, mirroring how ACE's own suite drives it
(`tests/leaderboard/test_xas_candidate_answer.py`): the file the Dockerfile
COPYs, not a convenience copy beside it.
"""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = ROOT / "docker" / "ace-candidate" / "vendor"

TRACK_ID = "dose-0.25"
DATASET_ID = "cu-cha-xas-denoising"
CANDIDATE_MANIFEST_SCHEMA = "ace-candidate-input-manifest/v1"
CANDIDATE_INPUT_KIND = "hyperspectrum-xas-inference/v3"


@pytest.fixture(scope="module")
def adapter() -> Iterator[ModuleType]:
    """`run.py`, imported the way the image imports it.

    The vendor directory goes on `sys.path` rather than each module being
    loaded from an isolated spec, because `run.py` does `from answer import ...`
    and `from entry import ...` — resolved in the image off the script's own
    directory and off `PYTHONPATH`, both of which are `/opt/acebench`. Loading
    it any other way would be testing an import graph the image does not have.
    """

    import importlib

    sys.path.insert(0, str(VENDOR_DIR))
    imported = [name for name in ("run", "answer", "entry") if name in sys.modules]
    assert not imported, f"names already taken by this checkout: {imported}"
    try:
        yield importlib.import_module("run")
    finally:
        sys.path.remove(str(VENDOR_DIR))
        for name in ("run", "answer", "entry"):
            sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def answer(adapter: ModuleType) -> ModuleType:
    return sys.modules["answer"]


# -- a candidate package shaped exactly as ACE publishes one -----------------


def inference_rows(count: int = 3, points: int = 21) -> dict[str, np.ndarray]:
    return {
        "energy": np.stack([np.linspace(8_800.0, 9_000.0, points) for _ in range(count)]),
        "energy_unit": np.asarray("eV"),
        "group_ids": np.asarray([f"High-Cu_exposure_g{index}" for index in range(count)]),
        "noisy": np.stack(
            [np.sin(np.linspace(0.0, 3.0, points)) + 0.01 * index for index in range(count)]
        ),
        "sample_ids": np.asarray([f"sample-{index:03d}" for index in range(count)]),
    }


def candidate_package(
    root: Path,
    *,
    track_id: str = TRACK_ID,
    input_kind: str = CANDIDATE_INPUT_KIND,
    count: int = 3,
    mutate: Any = None,
) -> Path:
    """Write `candidate-input-manifest.json` + `inputs/inference.npz`.

    Nothing else: no target, no split, no profile. That is the whole of what a
    dual-asset candidate receives, and a fixture that quietly added a benchmark
    file beside it would be testing the other lane.
    """

    (root / "inputs").mkdir(parents=True, exist_ok=True)
    asset = root / "inputs" / "inference.npz"
    rows = inference_rows(count=count)
    np.savez(asset, **rows)
    payload = asset.read_bytes()
    manifest: dict[str, Any] = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA,
        "input_kind": input_kind,
        "track_id": track_id,
        "dataset_id": DATASET_ID,
        "artifacts": [
            {
                "path": "inputs/inference.npz",
                "sha256": sha256(payload).hexdigest(),
                "size": len(payload),
            }
        ],
        "samples": [
            {"sample_id": str(value), "group_id": str(group)}
            for value, group in zip(rows["sample_ids"], rows["group_ids"])
        ],
    }
    if mutate is not None:
        mutate(manifest)
    (root / "candidate-input-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


def run_candidate(
    adapter: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    track_id: str | None = TRACK_ID,
    name: str = "candidate",
    **package: Any,
) -> Path:
    """Run the adapter over a candidate mount and hand back the output dir."""

    data = candidate_package(tmp_path / name / "data", **package)
    out = tmp_path / name / "out"
    out.mkdir(parents=True)
    if track_id is None:
        monkeypatch.delenv(adapter.TRACK_ID_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(adapter.TRACK_ID_VARIABLE, track_id)
    exit_code = adapter.main(
        ["--data", str(data), "--out", str(out / "candidate-output.json")]
    )
    assert exit_code == 0
    return out


# -- the run the image performs ----------------------------------------------


def test_the_adapter_answers_a_candidate_mount_with_this_checkouts_runtime(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The one test that would catch the vendored copies and the runtime drifting
    apart: no stub executor, no injected facade, the real call the image makes."""

    out = run_candidate(adapter, monkeypatch, tmp_path)

    answer_document = json.loads((out / "candidate-output.json").read_text("utf-8"))
    assert answer_document["schema_version"] == "ace-xas-candidate-answer/v1"
    assert answer_document["task_id"] == "xas-denoising"
    assert answer_document["energy_unit"] == "eV"
    assert answer_document["failures"] == []
    assert [item["sample_id"] for item in answer_document["predictions"]] == [
        "sample-000",
        "sample-001",
        "sample-002",
    ]
    for prediction in answer_document["predictions"]:
        assert set(prediction) == {"sample_id", "group_id", "energy", "intensity"}
        assert len(prediction["intensity"]) == 21
        assert all(np.isfinite(prediction["intensity"]))

    bundle = json.loads((out / "hyperspectrum" / "predictions.json").read_text("utf-8"))
    assert bundle["schema_version"] == "hyperspectrum-prediction/v2"
    assert bundle["task_id"] == "xas-denoising"


def test_the_completion_envelope_lands_beside_the_answer_and_carries_no_metrics(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ACE marks remote completion off `metrics.json`, and the number comes from
    the scoring container. A candidate that scored itself would be the thing the
    whole dual-asset boundary exists to prevent."""

    out = run_candidate(adapter, monkeypatch, tmp_path)

    envelope = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    # camelCase because the envelope is a wire model, not a Python one.
    assert set(envelope) == {"metrics", "nSamples", "durationS", "artifacts"}
    assert envelope["metrics"] == {}
    assert envelope["nSamples"] == 3
    recorded = {record["path"] for record in envelope["artifacts"]}
    assert recorded
    for path in recorded:
        assert (out / path).is_file()
        assert not path.startswith("/") and ".." not in path


def test_two_runs_over_the_same_bytes_produce_the_same_answer(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The candidate half of the digit-identical claim, at the only scale this
    machine can check it: one box, twice. A method that could not reproduce
    itself here would never have been worth comparing across two boxes."""

    first = run_candidate(adapter, monkeypatch, tmp_path, name="first")
    second = run_candidate(adapter, monkeypatch, tmp_path, name="second")

    assert (first / "candidate-output.json").read_bytes() == (
        second / "candidate-output.json"
    ).read_bytes()


# -- what the adapter refuses ------------------------------------------------


def test_a_container_told_no_track_refuses_before_it_opens_the_mount(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The one environment name ACE is entitled to send. Its absence is a
    platform fault, and typed apart from a disagreement about the mount."""

    with pytest.raises(adapter.CandidateEnvironmentError, match="TRACK_ID"):
        run_candidate(adapter, monkeypatch, tmp_path, track_id=None)


def test_a_package_published_for_another_track_is_refused(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unchecked, a mount prepared for one dose and answered under a run
    declaring another files a complete, well-formed answer against the wrong
    track, with every digest in the chain agreeing with every other."""

    with pytest.raises(adapter.CandidateMountError, match="dose-0.50"):
        run_candidate(adapter, monkeypatch, tmp_path, track_id="dose-0.50")


def test_a_package_in_the_older_input_kind_is_refused(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`/v2` packages carried every split's rows. This image answers every row it
    is given, so accepting one would mean answering for train and validation
    samples — which the scorer refuses, correctly, as leakage."""

    with pytest.raises(adapter.CandidateMountError, match="input kind"):
        run_candidate(
            adapter,
            monkeypatch,
            tmp_path,
            input_kind="hyperspectrum-xas-inference/v2",
        )


def test_a_mount_whose_bytes_are_not_the_ones_ace_published_is_refused(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = candidate_package(tmp_path / "data")
    manifest = json.loads((data / "candidate-input-manifest.json").read_text("utf-8"))
    manifest["artifacts"][0]["sha256"] = "0" * 64
    (data / "candidate-input-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv(adapter.TRACK_ID_VARIABLE, TRACK_ID)

    with pytest.raises(adapter.CandidateMountError, match="SHA-256"):
        adapter.main(["--data", str(data), "--out", str(out / "candidate-output.json")])


def test_a_mount_holding_rows_ace_never_declared_is_refused(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A digest cannot catch this: a re-materialization that went back to
    carrying every split's rows would be re-pinned end to end and internally
    consistent. ACE named the rows it published; the adapter compares."""

    def drop_a_sample(manifest: dict[str, Any]) -> None:
        manifest["samples"] = manifest["samples"][:-1]

    with pytest.raises(adapter.CandidateMountError, match="undeclared"):
        run_candidate(adapter, monkeypatch, tmp_path, mutate=drop_a_sample)


def test_an_out_path_ace_does_not_name_is_refused(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The two basenames are a contract with the platform; a typo that silently
    relocated the answer would be a run ACE collects nothing scoreable from."""

    data = candidate_package(tmp_path / "data")
    monkeypatch.setenv(adapter.TRACK_ID_VARIABLE, TRACK_ID)

    with pytest.raises(ValueError, match="--out must name"):
        adapter.main(["--data", str(data), "--out", str(tmp_path / "out/answer.json")])


def test_a_mount_that_is_neither_shape_is_refused(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(adapter.TRACK_ID_VARIABLE, TRACK_ID)

    with pytest.raises(adapter.CandidateMountError, match="neither"):
        adapter.main(["--data", str(empty), "--out", str(tmp_path / "candidate-output.json")])


# -- the projection, driven directly -----------------------------------------


def prediction_bundle(root: Path, *, samples: tuple[tuple[str, str], ...]) -> Path:
    """A `hyperspectrum-prediction/v2` bundle where the executor leaves one."""

    (root / "arrays").mkdir(parents=True, exist_ok=True)
    references = []
    for sample_id, group_id in samples:
        buffer = io.BytesIO()
        np.savez(
            buffer,
            sample_id=np.asarray(sample_id),
            group_id=np.asarray(group_id),
            energy=np.linspace(8_970.0, 8_990.0, 4, dtype=np.float64),
            intensity=np.linspace(0.10, 0.90, 4, dtype=np.float64),
            energy_unit=np.asarray("eV"),
        )
        payload = buffer.getvalue()
        uri = f"arrays/{sample_id}.npz"
        (root / uri).write_bytes(payload)
        references.append(
            {
                "role": "denoised_signal",
                "kind": "dense_array",
                "uri": uri,
                "sha256": sha256(payload).hexdigest(),
                "axes": [
                    {
                        "name": "energy",
                        "unit": "eV",
                        "direction": "increasing",
                        "values_uri": uri,
                    }
                ],
            }
        )
    (root / "predictions.json").write_text(
        json.dumps(
            {
                "schema_version": "hyperspectrum-prediction/v2",
                "run_id": "run-" + "a" * 64,
                "task_id": "xas-denoising",
                "predictions": references,
                "failures": [],
                "provenance": {"tool_digest": "b" * 64},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_the_projection_is_canonical_so_equal_predictions_digest_alike(
    answer: ModuleType, tmp_path: Path
) -> None:
    """ACE content-addresses this document. Two runs that predicted the same
    numbers in a different order must still produce the same bytes, or the
    attempt id stops naming the attempt."""

    first = answer.build_candidate_answer(
        prediction_bundle(tmp_path / "a", samples=(("s-002", "g-b"), ("s-001", "g-a")))
    )
    second = answer.build_candidate_answer(
        prediction_bundle(tmp_path / "b", samples=(("s-001", "g-a"), ("s-002", "g-b")))
    )

    assert first == second
    assert first == json.dumps(
        json.loads(first), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def test_an_artifact_that_does_not_match_its_declared_digest_is_refused(
    answer: ModuleType, tmp_path: Path
) -> None:
    """The container checks its own output before signing its name to it — the
    difference between a truncated write being caught here and being caught as a
    scientific disagreement two containers later."""

    root = prediction_bundle(tmp_path / "bundle", samples=(("s-001", "g-a"),))
    (root / "arrays" / "s-001.npz").write_bytes(b"not the array it published")

    with pytest.raises(answer.AnswerError, match="digest"):
        answer.build_candidate_answer(root)
