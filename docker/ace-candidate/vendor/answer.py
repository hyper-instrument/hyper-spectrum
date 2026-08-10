"""Project this container's own prediction bundle into the scorer's one document.

The scoring container reads exactly one candidate file,
``/scorer/candidate-output/candidate-output.json``, in
``ace-xas-candidate-answer/v1``, and it has to be self-contained: the mount is a
single *file*, so a ``hyperspectrum-prediction/v2`` bundle that references
sibling ``arrays/*.npz`` artifacts names files the scoring container cannot open.
The arrays therefore travel inside the document.

**Why this lives in the image and not in ACE.** ACE holds the same bundle after
collection and could assemble the document from it in twenty lines. It must not.
The candidate's answer is the thing being scored; an ACE-authored projection of
it would make ACE the author of the answer, and every refusal downstream — split
leakage, drifted energy grid, non-finite prediction — would be ACE checking its
own work rather than someone else's. So the container writes its own answer, out
of its own arrays, and ACE only measures the bytes.

**It is a projection, not a computation.** Every number here is read from an
array this container wrote; nothing is derived, resampled, reordered by value, or
defaulted. Where the bundle does not contain what the answer needs, this module
refuses — a candidate that cannot say what it predicted has not predicted.

The bounds below are the container's own, applied before numpy is asked to
allocate. They mirror the scorer's (``ace_xas_scorer``) so that a document this
module is willing to write is a document that side is willing to read; the two
files never import each other, which is the point, so the numbers are restated
rather than shared.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

#: The document this module writes, and the name ACE collects it under.
CANDIDATE_ANSWER_SCHEMA = "ace-xas-candidate-answer/v1"
CANDIDATE_ANSWER_FILENAME = "candidate-output.json"

#: The bundle this module reads. HyperSpectrum owns it; this is a consumer.
#:
#: One version, and it is the inference-only lane's, deliberately. HyperSpectrum
#: writes `/v2` for a `RunPlanV2` — one data identity, the mounted asset's own
#: digest — and `/v3` for a `RunPlanV3`, which carries three data identity
#: classes derived from a benchmark. A candidate has no benchmark, so the lane
#: that produces an answer produces `/v2` and always will; the lane that produces
#: `/v3` is the self-scored one, which ACE never asks for an answer from and
#: which `run.py` now refuses to write one for by name.
#:
#: Accepting `/v3` here would therefore not add a capability — it would add a
#: path by which a bundle built against ground truth could be projected into the
#: document that is scored against that same ground truth.
PREDICTION_SCHEMA = "hyperspectrum-prediction/v2"
PREDICTION_BUNDLE_FILENAME = "predictions.json"

TASK_ID = "xas-denoising"
ENERGY_UNIT = "eV"

MAX_BUNDLE_BYTES = 2 << 20
MAX_ARTIFACT_BYTES = 64 << 20
MAX_SAMPLES = 100_000
MAX_POINTS_PER_SAMPLE = 1_000_000
MAX_ARTIFACT_MEMBERS = 8


class AnswerError(RuntimeError):
    """The bundle does not support an answer. Nothing is written."""


def _read_bounded(path: Path, *, max_bytes: int, label: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise AnswerError(f"{label} is not a regular file")
        size = path.stat().st_size
        if size > max_bytes:
            # Before the read, not after: a bound enforced by reading the file
            # first is not a bound.
            raise AnswerError(f"{label} exceeds its {max_bytes} byte limit")
        return path.read_bytes()
    except OSError as error:
        raise AnswerError(f"{label} could not be read") from error


def _relative_within(root: Path, uri: object) -> Path:
    if not isinstance(uri, str) or not uri or "\\" in uri:
        raise AnswerError("a prediction artifact uri must be a normalized relative path")
    pure = PurePosixPath(uri)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise AnswerError(f"prediction artifact uri escapes the bundle: {uri!r}")
    return root.joinpath(*pure.parts)


def _npz_arrays(payload: bytes, *, label: str) -> dict[str, np.ndarray]:
    import io
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARTIFACT_MEMBERS:
                raise AnswerError(f"{label} holds more members than an artifact may")
            if sum(member.file_size for member in members) > MAX_ARTIFACT_BYTES:
                raise AnswerError(f"{label} expands past its byte limit")
    except zipfile.BadZipFile as error:
        raise AnswerError(f"{label} is not a readable NPZ") from error
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
            return {name: loaded[name] for name in loaded.files}
    except (OSError, ValueError, EOFError) as error:
        raise AnswerError(f"{label} could not be decoded") from error


def _scalar_text(arrays: dict[str, np.ndarray], name: str, *, label: str) -> str:
    value = arrays.get(name)
    if value is None or value.shape != () or value.dtype.kind not in "US":
        raise AnswerError(f"{label} {name} must be one plain string")
    text = str(value)
    if not text.strip():
        raise AnswerError(f"{label} {name} must be non-blank")
    return text


def _float_row(arrays: dict[str, np.ndarray], name: str, *, label: str) -> list[float]:
    value = arrays.get(name)
    if value is None or value.ndim != 1 or value.size == 0:
        raise AnswerError(f"{label} {name} must be a non-empty one-dimensional array")
    if value.size > MAX_POINTS_PER_SAMPLE:
        raise AnswerError(f"{label} {name} exceeds its point limit")
    if value.dtype.kind not in "fiu":
        raise AnswerError(f"{label} {name} must be numeric")
    row = np.asarray(value, dtype=np.float64)
    if not np.isfinite(row).all():
        # The container is the earlier and better place to refuse this than the
        # scorer: nothing is written, so there is no document to be tempted by.
        raise AnswerError(f"{label} {name} contains a value that is not finite")
    return [float(number) for number in row.tolist()]


def _bundle_document(bundle_directory: Path) -> dict[str, Any]:
    payload = _read_bounded(
        bundle_directory / PREDICTION_BUNDLE_FILENAME,
        max_bytes=MAX_BUNDLE_BYTES,
        label="the prediction bundle",
    )
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnswerError("the prediction bundle is not valid JSON") from error
    if not isinstance(document, dict):
        raise AnswerError("the prediction bundle must be a JSON object")
    if document.get("schema_version") != PREDICTION_SCHEMA:
        raise AnswerError(
            f"the prediction bundle is {document.get('schema_version')!r}; this "
            f"document is projected from {PREDICTION_SCHEMA!r}, which is what "
            "the inference-only lane writes"
        )
    if document.get("task_id") != TASK_ID:
        raise AnswerError("the prediction bundle names another task")
    return document


def _predictions(bundle_directory: Path, references: object) -> list[dict[str, Any]]:
    if not isinstance(references, list):
        raise AnswerError("the prediction bundle's predictions must be an array")
    if len(references) > MAX_SAMPLES:
        raise AnswerError("the prediction bundle exceeds its sample limit")
    predictions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reference in references:
        if not isinstance(reference, dict):
            raise AnswerError("each prediction reference must be an object")
        artifact = _relative_within(bundle_directory, reference.get("uri"))
        payload = _read_bounded(
            artifact, max_bytes=MAX_ARTIFACT_BYTES, label="a prediction artifact"
        )
        declared = reference.get("sha256")
        if not isinstance(declared, str) or hashlib.sha256(payload).hexdigest() != declared:
            # This container checking its own output. Cheap, and the difference
            # between a truncated write being caught here and being caught as a
            # scientific disagreement two containers later.
            raise AnswerError("a prediction artifact does not match its declared digest")
        arrays = _npz_arrays(payload, label="a prediction artifact")
        label = "a prediction artifact"
        sample_id = _scalar_text(arrays, "sample_id", label=label)
        if _scalar_text(arrays, "energy_unit", label=label) != ENERGY_UNIT:
            raise AnswerError(f"prediction {sample_id!r} is in another energy unit")
        if sample_id in seen:
            raise AnswerError(f"duplicate prediction for {sample_id!r}")
        seen.add(sample_id)
        energy = _float_row(arrays, "energy", label=f"prediction {sample_id!r}")
        intensity = _float_row(arrays, "intensity", label=f"prediction {sample_id!r}")
        if len(energy) != len(intensity):
            raise AnswerError(
                f"prediction {sample_id!r} has an energy grid and an intensity of "
                "different lengths"
            )
        predictions.append(
            {
                "sample_id": sample_id,
                "group_id": _scalar_text(arrays, "group_id", label=label),
                "energy": energy,
                "intensity": intensity,
            }
        )
    return predictions


def _failures(entries: object) -> list[dict[str, str]]:
    if not isinstance(entries, list):
        raise AnswerError("the prediction bundle's failures must be an array")
    failures: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise AnswerError("each failure must be an object")
        identity = {}
        for name in ("sample_id", "group_id"):
            value = entry.get(name)
            if not isinstance(value, str) or not value.strip():
                # Projected down to two fields, and refused when either is
                # missing: a failure the scorer cannot attribute to a sample is
                # not a failure it can account for.
                raise AnswerError(f"a failure does not name its {name}")
            identity[name] = value
        failures.append(identity)
    return failures


def build_candidate_answer(bundle_directory: Path) -> bytes:
    """The scorer's document, projected from this container's own bundle.

    Canonical bytes — sorted keys, no spaces, ``allow_nan=False``. ACE
    content-addresses this file into the attempt id, so two runs that predicted
    the same numbers have to produce the same digest; anything that let encoding
    vary would make one execution answer to two names.
    """

    root = Path(bundle_directory)
    document = _bundle_document(root)
    answer = {
        "schema_version": CANDIDATE_ANSWER_SCHEMA,
        "task_id": TASK_ID,
        "energy_unit": ENERGY_UNIT,
        "predictions": sorted(
            _predictions(root, document.get("predictions")),
            key=lambda item: item["sample_id"],
        ),
        "failures": sorted(
            _failures(document.get("failures")), key=lambda item: item["sample_id"]
        ),
    }
    if len(answer["predictions"]) + len(answer["failures"]) > MAX_SAMPLES:
        raise AnswerError("the answer exceeds its sample limit")
    try:
        return json.dumps(
            answer,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except ValueError as error:  # pragma: no cover - _float_row refuses first
        if not all(
            math.isfinite(number)
            for prediction in answer["predictions"]
            for number in prediction["intensity"]
        ):
            raise AnswerError("a prediction is not finite") from error
        raise


def write_candidate_answer(bundle_directory: Path, destination: Path) -> int:
    """Build the answer, then write it. Nothing lands unless all of it is valid.

    Built whole in memory first on purpose: a partially written answer is a
    document ACE would collect, measure and score. A refusal has to leave the
    output mount as it found it.
    """

    payload = build_candidate_answer(bundle_directory)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return len(payload)
