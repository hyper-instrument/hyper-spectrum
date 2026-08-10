"""Run the unweighted Savitzky-Golay XAS denoising baseline.

HyperSpectrum owns the scientific input and prediction contracts.  This thin
adapter only validates the ACE board identity, invokes that runtime, and emits
what the run was asked for.

Two documents, and which of them are written is decided by the name ACE puts on
``--out`` — never guessed from the environment:

``--out <dir>/metrics.json``
    The self-scored path, unchanged.  One completion envelope with an empty
    metrics map; ACE's ingester scores the canonical prediction bundle under
    ``<dir>/hyperspectrum/`` afterwards, so every execution backend shares one
    evaluator implementation.

``--out <dir>/candidate-output.json``
    The dual-asset path.  The same completion envelope still lands at
    ``<dir>/metrics.json``, and *beside* it this adapter projects its own
    prediction bundle into ``ace-xas-candidate-answer/v1`` — the single
    self-contained document the scoring container mounts.  ``answer.py`` says
    why that projection has to happen in here rather than in ACE.

Any other ``--out`` basename is a refusal.  The two names are a contract with
the platform (``research/candidate_launch.py``), and a typo that silently
relocated the answer would be a run ACE collects nothing scoreable from.

**Two mount shapes, and which one this is is read, never assumed.**

``--data <dir>`` holding a HyperSpectrum benchmark directory
    The self-scored lane.  ``manifest.json`` + ``split.json`` +
    ``benchmark.npz``; HyperSpectrum reads the manifest, loads pairs, and
    selects the test split itself.  Unchanged.

``--data <dir>`` holding ACE's candidate package
    The dual-asset lane.  ``candidate-input-manifest.json`` — ACE's own
    reserved document — plus ``inputs/inference.npz``, which is the whole of
    what a candidate receives: no target, no split, no profile.  This adapter
    verifies the asset — its track, its bytes and its rows — against what ACE
    said it published, and calls HyperSpectrum's inference-only facade.

Neither shape, or both, is a :class:`CandidateMountError`.  Both is the one
worth stating: whichever lane this adapter picked, it would be running one
nobody asked for, and one of the two has the answers in it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from answer import CANDIDATE_ANSWER_FILENAME, write_candidate_answer
from entry import parse_args, write_metrics

METRICS_FILENAME = "metrics.json"
#: Where the executor is told to leave its prediction bundle, relative to the
#: ACE output directory. Named once: the answer projection reads the same tree
#: the completion envelope's artifact records point into.
HYPERSPECTRUM_SUBDIR = "hyperspectrum"

EXPECTED_VARIANT_ID = "savitzky-golay"
EXPECTED_TASK_ID = "xas-denoising"
EXPECTED_TOOL_ID = "savgol"
EXPECTED_PARAMETERS = {"polyorder": 2, "window_length": 5}
EXPECTED_TRACK_DOSES = {
    "dose-0.10": 0.1,
    "dose-0.25": 0.25,
    "dose-0.50": 0.5,
}
BENCHMARK_FILENAMES = frozenset({"manifest.json", "split.json", "benchmark.npz"})

#: ACE's reserved document inside a candidate package, and the one artifact it
#: publishes beside it. Both names are ACE's (``research/xas_candidate_views.py``,
#: ``research/materializers.py``); they are restated rather than imported because
#: this file runs inside an image that has no ACE on its path, which is the point.
CANDIDATE_MANIFEST_FILENAME = "candidate-input-manifest.json"
CANDIDATE_MANIFEST_SCHEMA = "ace-candidate-input-manifest/v1"
CANDIDATE_ASSET_PATH = "inputs/inference.npz"
#: The mount contract this image is built for. ``/v2`` packages carried every
#: split's rows; this image answers every row it is given, so accepting one
#: would mean answering for train and validation samples — which the scorer
#: refuses, correctly, as leakage. Matched exactly.
CANDIDATE_INPUT_KIND = "hyperspectrum-xas-inference/v3"
MAX_CANDIDATE_MANIFEST_BYTES = 8 << 20
MAX_CANDIDATE_ASSET_BYTES = 512 << 20


class CandidateEnvironmentError(ValueError):
    """ACE did not tell this container the one thing it is entitled to be told.

    A candidate is entitled to its track identity and nothing else — everything
    else it needs is in the mount, or in ACE's manifest beside it, or compiled
    into this image. So this is always a platform fault: the boundary stage
    resolves the track from a signed isolation receipt, and its absence means
    that did not happen. Typed apart from `CandidateMountError` for exactly that
    reason: one is "ACE told me nothing", the other is "ACE and I disagree about
    what was handed over", and they are fixed in different places.
    """


class CandidateMountError(ValueError):
    """The data mount is not a shape this adapter can honestly run.

    Typed because a candidate mount is the one input this container does not
    author and cannot re-derive. Every refusal raised as this class is "ACE and
    I disagree about what was handed over" — which is an operator's problem to
    look at, and a different thing from the identity-drift refusals above (a
    board pointing the wrong image at a track) or a scientific refusal inside
    HyperSpectrum. A caller that wants to tell those apart can.
    """


def _default_executor(**kwargs: object) -> Any:
    # Import lazily: unit tests exercise the image boundary without installing a
    # sibling checkout, while the image installs the pinned HyperSpectrum sdist.
    from hyperspectrum.execution import execute_ace_xas_denoising

    return execute_ace_xas_denoising(**kwargs)


def _default_inference_executor(**kwargs: object) -> Any:
    from hyperspectrum.execution import execute_ace_xas_inference_only

    return execute_ace_xas_inference_only(**kwargs)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"missing required environment variable {name}")
    return value


#: The one environment name ACE's boundary stage is entitled to send, resolved
#: there from a signed isolation receipt (`research/candidate_launch.py`).
TRACK_ID_VARIABLE = "ACEBENCH_RESEARCH_TRACK_ID"


def _required_track_id() -> str:
    value = os.environ.get(TRACK_ID_VARIABLE)
    if value is None or not value.strip():
        raise CandidateEnvironmentError(
            f"ACE did not set {TRACK_ID_VARIABLE}; a candidate is entitled to be "
            "told which track it is answering, and cannot proceed without it"
        )
    return value


def _require_identity(name: str, expected: str, label: str) -> str:
    actual = _required_environment(name)
    if actual != expected:
        raise ValueError(f"{label} identity drift: expected {expected!r}, got {actual!r}")
    return actual


def _parameters() -> Mapping[str, object]:
    raw = _required_environment("ACEBENCH_HYPERSPECTRUM_PARAMETERS_JSON")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("parameters must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("parameters must be a JSON object")
    if parsed != EXPECTED_PARAMETERS or any(
        type(parsed.get(name)) is not type(expected) for name, expected in EXPECTED_PARAMETERS.items()
    ):
        raise ValueError(f"parameters identity drift: expected {EXPECTED_PARAMETERS!r}, got {parsed!r}")
    return parsed


def _dose_fraction(track_id: str) -> float:
    expected = EXPECTED_TRACK_DOSES.get(track_id)
    if expected is None:
        raise ValueError(f"track identity drift: unsupported track {track_id!r}")
    raw = _required_environment("ACEBENCH_XAS_DOSE_FRACTION")
    try:
        dose = float(raw)
    except ValueError as exc:
        raise ValueError("dose fraction must be a finite positive number") from exc
    if not math.isfinite(dose) or dose <= 0:
        raise ValueError("dose fraction must be a finite positive number")
    if dose != expected:
        raise ValueError(f"dose identity drift for {track_id}: expected {expected}, got {dose}")
    return dose


def _benchmark_directories(data_root: Path) -> list[Path]:
    if not data_root.is_dir():
        raise CandidateMountError(f"data root is not a directory: {data_root}")
    matches = {
        manifest.parent
        for manifest in data_root.rglob("manifest.json")
        if all((manifest.parent / name).is_file() for name in BENCHMARK_FILENAMES)
    }
    return sorted(matches)


def _read_bounded(path: Path, *, max_bytes: int, label: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise CandidateMountError(f"{label} is not a regular file")
        size = path.stat().st_size
        if size > max_bytes:
            # Before the read, not after: a bound enforced by reading the file
            # first is not a bound.
            raise CandidateMountError(f"{label} exceeds its {max_bytes} byte limit")
        return path.read_bytes()
    except OSError as error:
        raise CandidateMountError(f"{label} could not be read") from error


def _candidate_manifest(data_root: Path, *, track_id: str) -> tuple[dict[str, Any], str]:
    payload = _read_bounded(
        data_root / CANDIDATE_MANIFEST_FILENAME,
        max_bytes=MAX_CANDIDATE_MANIFEST_BYTES,
        label="the candidate input manifest",
    )
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateMountError(
            "the candidate input manifest is not valid JSON"
        ) from error
    if not isinstance(document, dict):
        raise CandidateMountError("the candidate input manifest must be a JSON object")
    if document.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA:
        raise CandidateMountError(
            "the candidate input manifest uses an unsupported schema: "
            f"{document.get('schema_version')!r}"
        )
    if document.get("input_kind") != CANDIDATE_INPUT_KIND:
        raise CandidateMountError(
            "this image is built for candidate input kind "
            f"{CANDIDATE_INPUT_KIND!r}, and was handed "
            f"{document.get('input_kind')!r}"
        )
    if document.get("track_id") != track_id:
        # On the self-scored lane the dose is held against the benchmark
        # manifest, which is in the mount. Here there is no such document — the
        # profile is scorer-only — so ACE's own statement of which track these
        # bytes are is the only one there is. Unchecked, a mount prepared for one
        # dose and answered under a run declaring another produces a complete,
        # well-formed answer filed against the wrong track, with every digest in
        # the chain agreeing with every other.
        raise CandidateMountError(
            f"this run is track {track_id!r}, and the mounted candidate package "
            f"was published for {document.get('track_id')!r}"
        )
    # Returned beside the document because it is provenance, not parsing: ACE
    # holds the same value in its materialization receipt as
    # `candidate_manifest_document_digest`.
    return document, hashlib.sha256(payload).hexdigest()


def _declared_asset(manifest: dict[str, Any]) -> tuple[str, int]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise CandidateMountError("the candidate input manifest declares no artifacts")
    declared = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("path") == CANDIDATE_ASSET_PATH
    ]
    if len(declared) != 1:
        raise CandidateMountError(
            f"the candidate input manifest must declare exactly one "
            f"{CANDIDATE_ASSET_PATH!r}, found {len(declared)}"
        )
    digest = declared[0].get("sha256")
    size = declared[0].get("size")
    if not isinstance(digest, str) or len(digest) != 64:
        raise CandidateMountError("the declared candidate asset digest is not a SHA-256")
    if isinstance(size, bool) or not isinstance(size, (int, float)) or size < 0:
        raise CandidateMountError("the declared candidate asset size is not a byte count")
    return digest, int(size)


def _declared_sample_ids(manifest: dict[str, Any]) -> set[str]:
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise CandidateMountError("the candidate input manifest declares no samples")
    declared: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise CandidateMountError("each declared sample must be an object")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise CandidateMountError("a declared sample does not name itself")
        if sample_id in declared:
            raise CandidateMountError(f"duplicate declared sample {sample_id!r}")
        declared.add(sample_id)
    return declared


def _asset_sample_ids(payload: bytes) -> set[str]:
    import io

    import numpy as np

    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
            values = np.array(loaded["sample_ids"], copy=True)
    except (OSError, ValueError, EOFError, KeyError) as error:
        raise CandidateMountError(
            "the candidate asset is not a readable inference NPZ"
        ) from error
    if values.dtype.kind != "U" or values.ndim != 1:
        raise CandidateMountError("candidate asset sample_ids must be a string array")
    return {str(value) for value in values.tolist()}


def _candidate_dataset_identity(
    manifest: dict[str, Any], document_sha256: str
) -> dict[str, str]:
    """Which data this is, stated from what ACE published rather than guessed.

    A candidate is not told the scientific dataset behind its mount — the
    profile that names it is a scorer-only asset — and it must not invent one.
    What ACE does publish is its own board dataset id, in the manifest beside
    the bytes, and that manifest is itself a complete content statement about
    this candidate input: the artifacts, their digests, and the samples.

    So: the dataset is the one ACE named, and its version is the digest of the
    manifest that published these bytes. That digest is not an invention either
    — ACE holds the same value in its materialization receipt as
    `candidate_manifest_document_digest`, so provenance written here can be
    checked against provenance ACE recorded independently.
    """

    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise CandidateMountError("the candidate input manifest names no dataset")
    return {"dataset_code": dataset_id, "dataset_version": document_sha256}


def _candidate_manifest_document(data_root: Path) -> tuple[dict[str, Any], str]:
    """Re-read the manifest `_resolve_mount` already validated.

    Read twice rather than threaded through the mount resolver, because the
    resolver's job is to answer one question — which lane is this — and widening
    its return value so a later step can skip a read would make the answer to
    that question carry a payload. The second read is bounded and re-validated
    by the same function, so a file that changed between them is refused rather
    than half-trusted.
    """

    return _candidate_manifest(
        data_root, track_id=_required_track_id()
    )


def _verified_candidate_asset(data_root: Path, *, track_id: str) -> Path:
    """Prove the mount is the one ACE said it published, before denoising it.

    Two checks, and the second is the one that matters scientifically. The digest
    catches a mount that is not the bytes ACE measured. The sample-id comparison
    catches something a digest cannot: a profile re-materialization that went
    back to carrying every split's rows, re-pinned end to end so every digest
    agrees with every other. Left alone, that mount produces a complete answer
    covering train and validation samples, and the first thing to notice is the
    scorer refusing the attempt as split leakage — a correct refusal that reads
    like the model's fault. ACE named the rows it published; this compares.
    """

    manifest, _document_digest = _candidate_manifest(data_root, track_id=track_id)
    declared_digest, declared_size = _declared_asset(manifest)
    asset = data_root / Path(CANDIDATE_ASSET_PATH)
    payload = _read_bounded(
        asset, max_bytes=MAX_CANDIDATE_ASSET_BYTES, label="the candidate asset"
    )
    if len(payload) != declared_size:
        raise CandidateMountError(
            f"the candidate asset is {len(payload)} bytes and was published as "
            f"{declared_size}"
        )
    if hashlib.sha256(payload).hexdigest() != declared_digest:
        raise CandidateMountError(
            "the candidate asset SHA-256 is not the one ACE published"
        )
    declared_ids = _declared_sample_ids(manifest)
    mounted_ids = _asset_sample_ids(payload)
    if mounted_ids != declared_ids:
        undeclared = sorted(mounted_ids - declared_ids)
        missing = sorted(declared_ids - mounted_ids)
        raise CandidateMountError(
            "the candidate asset holds rows ACE did not declare, or is missing "
            f"rows it did: undeclared={undeclared[:8]}, missing={missing[:8]}"
        )
    return asset


def _artifact_records(metrics_path: Path, paths: tuple[Path, ...]) -> list[dict[str, object]]:
    root = metrics_path.parent.resolve()
    records: list[dict[str, object]] = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact is outside the ACE output directory: {path}") from exc
        if not resolved.is_file():
            raise ValueError(f"declared artifact does not exist: {path}")
        records.append({"path": relative.as_posix(), "size": resolved.stat().st_size})
    return records


def _output_paths(out: str) -> tuple[Path, Path | None]:
    """Where the completion envelope goes, and whether an answer was asked for.

    The envelope's own location is fixed rather than taken from ``--out``: it is
    what marks the run complete, and a run that wrote its completion marker
    under whatever name happened to be requested would be a run ACE's other
    lane cannot find.
    """

    requested = Path(out)
    if requested.name == METRICS_FILENAME:
        return requested.parent / METRICS_FILENAME, None
    if requested.name == CANDIDATE_ANSWER_FILENAME:
        return requested.parent / METRICS_FILENAME, requested
    raise ValueError(
        f"--out must name {METRICS_FILENAME!r} or {CANDIDATE_ANSWER_FILENAME!r}, "
        f"got {requested.name!r}"
    )


def _resolve_mount(data_root: Path, *, track_id: str) -> tuple[str, Path]:
    """Say which lane this mount is, or refuse — never guess.

    A candidate package is recognised by ACE's reserved manifest sitting at the
    mount root, which is a document only ACE writes. A benchmark is recognised
    the way it always was. Both present is refused rather than resolved by
    precedence: a precedence rule is a silent decision about which lane ran, and
    one of these two lanes has the answers in it.
    """

    benchmarks = _benchmark_directories(data_root)
    is_candidate = (data_root / CANDIDATE_MANIFEST_FILENAME).is_file()
    if is_candidate and benchmarks:
        raise CandidateMountError(
            f"the data mount at {data_root} is both an ACE candidate package and "
            f"a HyperSpectrum benchmark directory; refusing rather than choosing"
        )
    if is_candidate:
        return "candidate", _verified_candidate_asset(data_root, track_id=track_id)
    if len(benchmarks) == 1:
        return "benchmark", benchmarks[0]
    if benchmarks:
        # Word-for-word what this said before there was a second lane. A mount
        # with two benchmarks is the same provisioning error it always was, and
        # an operator who has seen the message before should not have to work out
        # whether it now means something different.
        raise CandidateMountError(
            f"expected exactly one HyperSpectrum benchmark under {data_root}, "
            f"found {len(benchmarks)}"
        )
    raise CandidateMountError(
        f"the data mount at {data_root} is neither a HyperSpectrum benchmark "
        f"directory nor an ACE candidate package"
    )


def main(
    argv: list[str] | None = None,
    *,
    executor: Callable[..., Any] | None = None,
    inference_executor: Callable[..., Any] | None = None,
) -> int:
    args = parse_args(argv)
    started = time.monotonic()

    # The one thing ACE tells a candidate, and the only environment both lanes
    # share. Read before the mount so a container that was told nothing refuses
    # without opening the one thing it is supposed to be careful with.
    track_id = _required_track_id()
    if args.weights:
        raise ValueError("the Savitzky-Golay variant does not accept weights")

    lane, located = _resolve_mount(Path(args.data), track_id=track_id)
    metrics_path, answer_path = _output_paths(args.out)
    bundle_directory = metrics_path.parent / HYPERSPECTRUM_SUBDIR
    if lane == "candidate":
        manifest, manifest_digest = _candidate_manifest_document(Path(args.data))
        result = (inference_executor or _default_inference_executor)(
            inference_asset=located,
            output_directory=bundle_directory,
            # Not read from the environment, because on this lane ACE states
            # none of it. The image *is* the variant: these four values are
            # compiled into it, and a container checking its own constants
            # against its own constants is theatre. What ACE does state — which
            # track these bytes are — is checked against the mount instead.
            task_id=EXPECTED_TASK_ID,
            tool_id=EXPECTED_TOOL_ID,
            parameters=EXPECTED_PARAMETERS,
            **_candidate_dataset_identity(manifest, manifest_digest),
            max_samples=args.max_samples,
            weight_files=(),
        )
    else:
        if answer_path is not None:
            # `answer.py` projects a `hyperspectrum-prediction/v2` bundle, which
            # is what the inference-only facade writes. The self-scored facade
            # writes `/v3` — it plans as a v3 run, with three data identity
            # classes a candidate does not have. So this combination used to run
            # a full denoise and then fail inside the projection with
            # "unsupported schema": a message about a schema, for what is really
            # a lane with no answer contract. Nothing wires it; this is so
            # nothing can.
            raise ValueError(
                f"the self-scored lane has no {CANDIDATE_ANSWER_FILENAME!r} to "
                "write: its prediction bundle is not the contract the scoring "
                "container reads"
            )
        _require_identity("ACEBENCH_RESEARCH_VARIANT_ID", EXPECTED_VARIANT_ID, "variant")
        task_id = _require_identity(
            "ACEBENCH_HYPERSPECTRUM_TASK_ID", EXPECTED_TASK_ID, "task"
        )
        tool_id = _require_identity(
            "ACEBENCH_HYPERSPECTRUM_TOOL_ID", EXPECTED_TOOL_ID, "tool"
        )
        result = (executor or _default_executor)(
            benchmark_directory=located,
            output_directory=bundle_directory,
            task_id=task_id,
            tool_id=tool_id,
            parameters=_parameters(),
            # The dose is a property of the track, and on this lane the
            # benchmark manifest is there to be checked against it. The
            # candidate lane has no such document — the profile is scorer-only —
            # so its track identity is checked against ACE's manifest instead.
            expected_dose_fraction=_dose_fraction(track_id),
            max_samples=args.max_samples,
            weight_files=(),
        )
    artifacts = _artifact_records(metrics_path, tuple(result.artifact_paths))

    # Before the completion envelope, deliberately.  The envelope is what marks
    # the run finished; a run that marked itself complete and then failed to
    # project its answer would look, to everything downstream, like a candidate
    # that simply had nothing to say.
    if answer_path is not None:
        write_candidate_answer(bundle_directory, answer_path)

    # ACE requires a metrics envelope to mark remote completion.  The empty
    # metrics map is intentional: on the self-scored path the ingester scores
    # the canonical prediction bundle, keeping all execution backends on one
    # evaluator implementation; on the dual-asset path the number comes from the
    # scoring container, and this envelope carries no metrics for the same
    # reason — the candidate does not score itself.
    write_metrics(
        metrics_path,
        metrics={},
        n_samples=result.n_samples,
        duration_s=time.monotonic() - started,
        artifacts=artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
