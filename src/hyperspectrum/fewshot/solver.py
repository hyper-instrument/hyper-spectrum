"""Fixed ``prepare`` / ``train`` / ``predict`` contract for the few-shot kit.

The kind-agnostic kit drives this module as a subprocess::

    python -m hyperspectrum.fewshot.solver prepare --release-dir R --work-dir W
    python -m hyperspectrum.fewshot.solver train   --work-dir W --model-dir M
    python -m hyperspectrum.fewshot.solver predict --release-dir R --work-dir W \\
        --model-dir M --out-dir O

Each step writes its JSON summary (``W/prepare.json``, ``M/train.json``,
``O/predict.json``) and prints it as the last stdout line. ``--opt key=value``
may repeat; unknown keys exit 2. Every failure exits non-zero with one line
on stderr and never a traceback: expected ones through :class:`SolverError`,
anything else through the backstop in :func:`main`. The same three steps are
exposed as plain functions for in-process use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .delivery import Usage, write_delivery
from .knn import (
    DISTANCE_FLOOR,
    EXACT_MATCH_TOLERANCE,
    GRID_POINTS,
    NEIGHBOR_COUNT,
    PAIRED_POOL_KEYS,
    QUERY_REQUIRED_KEYS,
    PairIndex,
    direction_keys,
    predict_many,
    reverse_direction,
)

DEFAULT_SEED = 42
POOLS_SUBDIR = "pools"
PREPARE_OPTS = frozenset({"validation_pool"})
TRAIN_OPTS = frozenset({"seed"})
PREDICT_OPTS: frozenset[str] = frozenset()
METHOD: dict[str, Any] = {
    "name": "knn-baseline",
    "neighbors": NEIGHBOR_COUNT,
    "distance": "mean_abs",
    "weight": "inverse_square",
    "candidate_rule": "same_Z_and_edge > same_edge > all",
}


class SolverError(SystemExit):
    """An expected failure: one stderr line and a non-zero exit, no traceback.

    Subclasses ``SystemExit`` so an uncaught instance still ends the process
    with the message on stderr; :func:`main` honours ``exit_status`` (2 for
    usage errors such as an unknown ``--opt`` key, 1 otherwise).
    """

    def __init__(self, message: str, *, exit_status: int = 1) -> None:
        super().__init__(message)
        self.exit_status = exit_status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_opts(opts: Mapping[str, str], allowed: frozenset[str], command: str) -> None:
    for key in opts:
        if key not in allowed:
            known = ", ".join(sorted(allowed)) or "none"
            raise SolverError(
                f"{command}: unknown --opt key {key!r} (known keys: {known})",
                exit_status=2,
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, command: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SolverError(f"{command}: cannot read {path}: {error}") from error
    except ValueError as error:
        raise SolverError(f"{command}: {path} is not valid JSON: {error}") from error
    if not isinstance(loaded, dict):
        raise SolverError(f"{command}: {path} must contain a JSON object")
    return loaded


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_npz(path: Path, *, command: str) -> dict[str, NDArray[Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise SolverError(f"{command}: cannot load {path}: {error}") from error


def _safe_name(name: str, *, command: str) -> str:
    parts = Path(name).parts
    if not name or Path(name).is_absolute() or ".." in parts:
        raise SolverError(f"{command}: refusing manifest file name {name!r}")
    return name


# --- release manifest -------------------------------------------------------


@dataclass(frozen=True)
class _Manifest:
    """The parts of ``data_manifest.json`` the solver relies on, validated."""

    release_id: str
    files: dict[str, dict[str, Any]]
    grid: dict[str, Any]
    axis: NDArray[np.float64]
    queries_sha256: str | None


def _validate_manifest(manifest: Mapping[str, Any], *, command: str) -> _Manifest:
    release_id = manifest.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise SolverError(f"{command}: data_manifest.json lacks a release_id")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict):
        raise SolverError(f"{command}: data_manifest.json lacks a 'files' object")
    files: dict[str, dict[str, Any]] = {}
    for raw_name, entry in raw_files.items():
        name = _safe_name(str(raw_name), command=command)
        if not isinstance(entry, dict):
            raise SolverError(f"{command}: manifest entry for {name} must be an object")
        files[name] = entry
    grid = manifest.get("grid")
    if not isinstance(grid, dict) or grid.get("points") != GRID_POINTS:
        raise SolverError(
            f"{command}: unsupported grid {grid!r}; this solver needs "
            f"{GRID_POINTS} points"
        )
    start, step = grid.get("start_ev"), grid.get("step_ev")
    if (
        not isinstance(start, (int, float))
        or not isinstance(step, (int, float))
        or isinstance(start, bool)
        or isinstance(step, bool)
        or step == 0
    ):
        raise SolverError(
            f"{command}: grid needs numeric start_ev and non-zero step_ev, got {grid!r}"
        )
    axis = float(start) + float(step) * np.arange(GRID_POINTS, dtype=np.float64)
    queries_sha256 = manifest.get("queries_sha256")
    if queries_sha256 is not None and (
        not isinstance(queries_sha256, str) or not queries_sha256
    ):
        raise SolverError(
            f"{command}: queries_sha256 must be a non-empty string when present"
        )
    return _Manifest(release_id, files, dict(grid), axis, queries_sha256)


def _check_energy(
    energy: Any, axis: NDArray[np.float64], *, what: str, command: str
) -> NDArray[np.float64]:
    """Require ``energy`` to be the manifest's ``start_ev + step_ev * k`` axis."""
    values = np.asarray(energy)
    if values.shape != (GRID_POINTS,) or values.dtype.kind not in "fiu":
        raise SolverError(
            f"{command}: {what} energy must be a numeric ({GRID_POINTS},) axis, "
            f"got shape {values.shape} and dtype {values.dtype}"
        )
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or not np.allclose(values, axis):
        raise SolverError(
            f"{command}: {what} energy axis does not match the manifest grid "
            f"({axis[0]:g} eV in {axis[1] - axis[0]:g} eV steps)"
        )
    return values


# --- prepare ----------------------------------------------------------------


def _missing_paired_keys(keys: Sequence[str]) -> list[str]:
    return [key for key in PAIRED_POOL_KEYS if key not in keys]


def _skip_reason(source: Path, name: str) -> str | None:
    """Why an optional manifest file is not staged; None when it is a paired pool."""
    if not name.endswith(".npz"):
        return "not an npz archive"
    try:
        with np.load(source, allow_pickle=False) as archive:
            keys = list(archive.files)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return "not an npz archive"
    missing = _missing_paired_keys(keys)
    if missing:
        return f"not a paired pool (missing {', '.join(missing)})"
    return None


@dataclass(frozen=True)
class _StagedPool:
    record: dict[str, Any]
    sample_id: NDArray[np.str_]
    energy: NDArray[np.float64]


def _stage_pool(
    source: Path,
    staged: Path,
    name: str,
    entry: Mapping[str, Any],
    *,
    axis: NDArray[np.float64],
) -> _StagedPool:
    """Verify one paired pool against the manifest and copy it into the work dir."""
    expected = entry.get("sha256")
    actual = _sha256(source)
    if actual != expected:
        raise SolverError(
            f"prepare: sha256 mismatch for {name}: manifest says {expected}, "
            f"file has {actual}"
        )
    pool = _load_npz(source, command="prepare")
    missing = _missing_paired_keys(list(pool))
    if missing:
        raise SolverError(
            f"prepare: {name} is not a paired pool; missing keys: {', '.join(missing)}"
        )
    energy = _check_energy(pool["energy"], axis, what=name, command="prepare")
    try:
        index = PairIndex.from_pools([pool])
    except ValueError as error:
        raise SolverError(f"prepare: {name}: {error}") from error
    records = len(index)
    listed = entry.get("records")
    if listed is not None and (not _is_int(listed) or listed != records):
        raise SolverError(
            f"prepare: {name} holds {records} records but the manifest lists {listed!r}"
        )
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, staged)
    record = {
        "name": name,
        "records": records,
        "sha256": actual,
        "required": entry.get("required") is True,
        "role": "index",
        "diagnostic": False,
    }
    return _StagedPool(record, index.sample_id, energy)


def _duplicate_ids(pools: Sequence[_StagedPool]) -> list[str]:
    ids = np.concatenate([pool.sample_id for pool in pools])
    unique, counts = np.unique(ids, return_counts=True)
    return [str(value) for value in unique[counts > 1]]


def _select_validation_pool(
    names: Sequence[str], opts: Mapping[str, str]
) -> str | None:
    """Pick the diagnostic pool: the ``validation_pool`` opt, else by name.

    The diagnostic pool is scored leak-free against an index built without
    it and then re-admitted to the final index, so any staged pool, required
    or not, may be chosen. An empty opt skips the diagnostic. By default a
    pool whose name contains ``validation`` is picked, provided another pool
    exists to score it against.
    """
    if "validation_pool" in opts:
        chosen = opts["validation_pool"].strip()
        if not chosen:
            return None
        if chosen not in names:
            raise SolverError(
                f"prepare: --opt validation_pool={chosen!r} is not a staged paired "
                f"pool (staged pools: {', '.join(names)})"
            )
        if len(names) == 1:
            raise SolverError(
                f"prepare: {chosen} is the only paired pool; nothing to score it "
                "against (pass --opt validation_pool= to skip the diagnostic)"
            )
        return chosen
    if len(names) < 2:
        return None
    for name in names:
        if "validation" in name:
            return name
    return None


def prepare(
    release_dir: Path, work_dir: Path, opts: Mapping[str, str]
) -> dict[str, Any]:
    """Verify the release's paired pools and stage them into ``work_dir``.

    Every ``required: true`` file must be a paired pool whose bytes match the
    manifest's sha256. Optional files are staged only when they are paired
    pools too (single-domain pools and sidecars are skipped and listed under
    ``skipped``). Every staged pool must sit on the manifest's energy grid
    and no sample id may appear twice across pools.
    """
    _check_opts(opts, PREPARE_OPTS, "prepare")
    manifest = _validate_manifest(
        _load_json_object(release_dir / "data_manifest.json", command="prepare"),
        command="prepare",
    )

    pools_dir = work_dir / POOLS_SUBDIR
    pools_dir.mkdir(parents=True, exist_ok=True)
    staged: list[_StagedPool] = []
    skipped: list[dict[str, str]] = []
    for name, entry in manifest.files.items():
        required = entry.get("required") is True
        source = release_dir / name
        if not source.is_file():
            if required:
                raise SolverError(
                    f"prepare: required pool {name} is missing from {release_dir}"
                )
            skipped.append({"name": name, "reason": "file not present"})
            continue
        if not required:
            reason = _skip_reason(source, name)
            if reason is not None:
                skipped.append({"name": name, "reason": reason})
                continue
        pool = _stage_pool(source, pools_dir / name, name, entry, axis=manifest.axis)
        if staged and not np.allclose(pool.energy, staged[0].energy):
            raise SolverError(
                f"prepare: {name} energy axis differs from {staged[0].record['name']}"
            )
        staged.append(pool)
    if not staged:
        raise SolverError("prepare: data_manifest.json lists no paired pools")
    duplicates = _duplicate_ids(staged)
    if duplicates:
        shown = ", ".join(duplicates[:5]) + (" ..." if len(duplicates) > 5 else "")
        raise SolverError(f"prepare: duplicate sample ids across paired pools: {shown}")

    # Every staged pool joins the final index (role "index"); the diagnostic
    # pool is additionally scored leak-free before being re-admitted.
    pools = [entry.record for entry in staged]
    validation_pool = _select_validation_pool(
        [record["name"] for record in pools], opts
    )
    for record in pools:
        record["diagnostic"] = record["name"] == validation_pool

    result: dict[str, Any] = {
        "release_id": manifest.release_id,
        "grid": manifest.grid,
        "pools": pools,
        "validation_pool": validation_pool,
        "skipped": skipped,
        "created_at": _now(),
    }
    _write_json(work_dir / "prepare.json", result)
    return result


# --- train ------------------------------------------------------------------


def _prepared_pools(
    prepared: Mapping[str, Any], *, command: str
) -> list[dict[str, Any]]:
    """The staged-pool records of ``prepare.json``, shape-checked."""
    pools = prepared.get("pools")
    if not isinstance(pools, list):
        raise SolverError(
            f"{command}: prepare.json lacks a 'pools' list; rerun prepare"
        )
    checked: list[dict[str, Any]] = []
    for position, pool in enumerate(pools):
        well_formed = (
            isinstance(pool, dict)
            and isinstance(pool.get("name"), str)
            and bool(pool.get("name"))
            and _is_int(pool.get("records"))
            and pool["records"] >= 0
            and isinstance(pool.get("sha256"), str)
            and bool(pool.get("sha256"))
        )
        if not well_formed:
            raise SolverError(
                f"{command}: prepare.json pool entry {position} is malformed "
                "(needs name, records, sha256); rerun prepare"
            )
        checked.append(dict(pool))
    return checked


def _staged_pool(
    work_dir: Path, pool: Mapping[str, Any], *, command: str
) -> dict[str, NDArray[Any]]:
    name = _safe_name(str(pool["name"]), command=command)
    path = work_dir / POOLS_SUBDIR / name
    if not path.is_file():
        raise SolverError(f"{command}: staged pool {name} is missing; rerun prepare")
    actual = _sha256(path)
    if actual != pool["sha256"]:
        raise SolverError(f"{command}: staged pool {name} changed since prepare")
    return _load_npz(path, command=command)


def _direction_mae(
    index: PairIndex, pool: Mapping[str, NDArray[Any]], direction: str
) -> float:
    input_key, output_key = direction_keys(direction)
    count = len(pool["sample_id"])
    queries = {
        "direction": np.full(count, direction),
        "spectrum": pool[input_key],
        "absorber_atomic_number": pool["absorber_atomic_number"],
        "edge_code": pool["edge_code"],
    }
    predicted = np.asarray(predict_many(queries, index), dtype=np.float64)
    target = np.asarray(pool[output_key], dtype=np.float64)
    return float(np.mean(np.abs(predicted - target)))


def train(work_dir: Path, model_dir: Path, opts: Mapping[str, str]) -> dict[str, Any]:
    """Score the diagnostic pool leak-free, then index every staged pool.

    The diagnostic pool (``prepare.json["validation_pool"]``) is predicted
    from the index with its sample ids removed, so its MAE is not a
    self-match. The final ``M/index.npz`` keeps every staged pool's rows, as
    the evaluator's data-usage rule demands of required pools.
    """
    _check_opts(opts, TRAIN_OPTS, "train")
    seed_text = opts.get("seed", str(DEFAULT_SEED))
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise SolverError(
            f"train: --opt seed must be an integer, got {seed_text!r}"
        ) from error
    prepared = _load_json_object(work_dir / "prepare.json", command="train")
    pools = _prepared_pools(prepared, command="train")
    if not pools:
        raise SolverError("train: prepare.json names no paired pools")
    names = [str(pool["name"]) for pool in pools]
    staged = {
        name: _staged_pool(work_dir, pool, command="train")
        for name, pool in zip(names, pools, strict=True)
    }
    try:
        index = PairIndex.from_pools([staged[name] for name in names])
    except ValueError as error:
        raise SolverError(f"train: {error}") from error

    validation: dict[str, Any] | None = None
    diagnostic_name = prepared.get("validation_pool")
    if diagnostic_name is not None:
        diagnostic_name = str(diagnostic_name)
        if diagnostic_name not in staged:
            raise SolverError(
                f"train: diagnostic pool {diagnostic_name} is not staged; rerun prepare"
            )
        held_out = staged[diagnostic_name]
        try:
            # Drop the pool's *ids*, not merely its file, so no row scores itself.
            diagnostic_index = index.excluding(held_out["sample_id"])
            validation = {
                "name": diagnostic_name,
                "records": len(held_out["sample_id"]),
                "mae_sim2exp": _direction_mae(diagnostic_index, held_out, "sim2exp"),
                "mae_exp2sim": _direction_mae(diagnostic_index, held_out, "exp2sim"),
                "diagnostic": True,
            }
        except ValueError as error:
            raise SolverError(
                f"train: diagnostic on {diagnostic_name}: {error}"
            ) from error

    model_dir.mkdir(parents=True, exist_ok=True)
    index.save(model_dir / "index.npz")
    result: dict[str, Any] = {
        "seed": seed,
        "release_id": prepared.get("release_id"),
        "pools": names,
        "records": len(index),
        "validation": validation,
        "created_at": _now(),
    }
    _write_json(model_dir / "train.json", result)
    return result


# --- predict ----------------------------------------------------------------


def _check_trained(trained: Mapping[str, Any], *, command: str) -> None:
    pools = trained.get("pools")
    validation = trained.get("validation")
    well_formed = (
        _is_int(trained.get("seed"))
        and _is_int(trained.get("records"))
        and isinstance(pools, list)
        and all(isinstance(name, str) for name in pools)
        and (validation is None or isinstance(validation, dict))
    )
    if not well_formed:
        raise SolverError(
            f"{command}: train.json is malformed (needs integer seed and records, "
            "a pools list and validation object or null); rerun train"
        )


@dataclass(frozen=True)
class _PredictInputs:
    prepared: dict[str, Any]
    pools: list[dict[str, Any]]
    trained: dict[str, Any]
    manifest: dict[str, Any]
    queries: dict[str, NDArray[Any]]
    index: PairIndex


def _load_predict_inputs(
    release_dir: Path, work_dir: Path, model_dir: Path
) -> _PredictInputs:
    """Load and cross-check everything ``predict`` needs before predicting."""
    prepared = _load_json_object(work_dir / "prepare.json", command="predict")
    trained = _load_json_object(model_dir / "train.json", command="predict")
    raw_manifest = _load_json_object(
        release_dir / "data_manifest.json", command="predict"
    )
    manifest = _validate_manifest(raw_manifest, command="predict")
    release_ids = {
        "data_manifest.json": manifest.release_id,
        "prepare.json": prepared.get("release_id"),
        "train.json": trained.get("release_id"),
    }
    if len(set(release_ids.values())) != 1:
        raise SolverError(f"predict: release_id mismatch across {release_ids}")
    pools = _prepared_pools(prepared, command="predict")
    _check_trained(trained, command="predict")

    queries_path = release_dir / "queries.npz"
    if not queries_path.is_file():
        raise SolverError(f"predict: {queries_path} is missing")
    if manifest.queries_sha256 is not None:
        actual = _sha256(queries_path)
        if actual != manifest.queries_sha256:
            raise SolverError(
                f"predict: sha256 mismatch for queries.npz: manifest says "
                f"{manifest.queries_sha256}, file has {actual}"
            )
    queries = _load_npz(queries_path, command="predict")
    missing = [key for key in QUERY_REQUIRED_KEYS if key not in queries]
    if missing:
        raise SolverError(f"predict: queries.npz is missing keys: {', '.join(missing)}")
    if "energy" in queries:
        _check_energy(
            queries["energy"], manifest.axis, what="queries.npz", command="predict"
        )

    try:
        index = PairIndex.load(model_dir / "index.npz")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SolverError(f"predict: cannot load index.npz: {error}") from error
    if len(index) != trained["records"]:
        raise SolverError(
            f"predict: index.npz holds {len(index)} records but train.json says "
            f"{trained['records']}; rerun train"
        )
    return _PredictInputs(prepared, pools, trained, raw_manifest, queries, index)


def _report(
    prepared: Mapping[str, Any],
    trained: Mapping[str, Any],
    counts: Mapping[str, int],
) -> str:
    grid = prepared.get("grid") or {}
    summary = (
        f"Release `{prepared.get('release_id')}`, grid of {grid.get('points')} "
        f"points from {grid.get('start_ev')} eV in {grid.get('step_ev')} eV steps. "
        f"Queries answered: {counts.get('sim2exp', 0)} sim2exp, "
        f"{counts.get('exp2sim', 0)} exp2sim."
    )
    method = (
        "For every query the solver collects the training pairs with the same "
        "absorber atomic number and edge code, falling back to the same edge "
        "code and then to every pair. Candidates are ranked by the mean "
        "absolute difference between the query and the candidate spectrum in "
        f"the query's own domain (ties broken by sample id); the {NEIGHBOR_COUNT} "
        "nearest are averaged with inverse-square distance weights, the distance "
        f"floored at {DISTANCE_FLOOR:g}. An exact match (distance <= "
        f"{EXACT_MATCH_TOLERANCE:g}) is returned on its own. Arithmetic is "
        "float64; predictions are stored as float32."
    )
    cycle = (
        "Cycle predictions apply the same lookup in the reverse direction to "
        "the solver's own prediction, so they are a genuine round trip rather "
        "than the query echoed back."
    )
    table = [
        "| pool | records | required | diagnostic | rows admitted to the index |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for pool in prepared.get("pools") or []:
        table.append(
            f"| {pool.get('name')} | {pool.get('records')} | "
            f"{'yes' if pool.get('required') else 'no'} | "
            f"{'yes' if pool.get('diagnostic') else 'no'} | {pool.get('records')} |"
        )
    unused = (
        "Single-domain public pools (simulation-only or experiment-only) are "
        "not read: the baseline needs paired records. `data_usage.json` reports "
        "each staged pool's rows admitted to the index as `influencing`."
    )
    validation = trained.get("validation")
    if isinstance(validation, dict):
        others = [
            name for name in trained.get("pools") or [] if name != validation["name"]
        ]
        validation_text = (
            f"Diagnostic pool `{validation.get('name')}` ({validation.get('records')} "
            "pairs) was scored leak-free from a train-only index built without its "
            f"sample ids (pools: {', '.join(others)}): MAE sim2exp = "
            f"{validation.get('mae_sim2exp'):.6f}, MAE exp2sim = "
            f"{validation.get('mae_exp2sim'):.6f}. It was then re-admitted to the "
            "final index, so every paired pool's rows are admitted to the index "
            "that answers the queries."
        )
    else:
        validation_text = (
            "No diagnostic pool was scored; every paired pool is in the final index."
        )
    limitations = [
        (
            "- Nearest-neighbour oracle: nothing is learned, so predictions cannot "
            "leave the convex hull of the training targets."
        ),
        (
            "- Queries whose absorber or edge code is unknown (for example cycle "
            "rows carrying code 0) fall back to every training pair."
        ),
        (
            "- A query that exactly matches a training spectrum returns that pair's "
            "partner, so its cycle prediction reproduces the query verbatim."
        ),
        (
            f"- The seed ({trained.get('seed')}) is recorded for the contract only; "
            "the method is deterministic."
        ),
    ]
    sections = [
        "# k-NN baseline (hyperspectrum.fewshot)",
        summary,
        "## Method",
        method,
        cycle,
        "## Data usage",
        "\n".join(table),
        unused,
        "## Validation",
        validation_text,
        "## Limitations",
        "\n".join(limitations),
    ]
    return "\n\n".join(sections) + "\n"


def predict(
    release_dir: Path,
    work_dir: Path,
    model_dir: Path,
    out_dir: Path,
    opts: Mapping[str, str],
) -> dict[str, Any]:
    """Answer every query in both directions and write the seven-file delivery."""
    _check_opts(opts, PREDICT_OPTS, "predict")
    inputs = _load_predict_inputs(release_dir, work_dir, model_dir)
    prepared, trained, queries, index = (
        inputs.prepared,
        inputs.trained,
        inputs.queries,
        inputs.index,
    )

    started = _now()
    directions = np.asarray(queries["direction"]).astype(str)
    try:
        predictions = predict_many(queries, index)
        reversed_queries = {
            "direction": np.asarray(
                [reverse_direction(value) for value in directions.tolist()]
            ),
            "spectrum": predictions,
            "absorber_atomic_number": queries["absorber_atomic_number"],
            "edge_code": queries["edge_code"],
        }
        cycle_predictions = predict_many(reversed_queries, index)
    except ValueError as error:
        raise SolverError(f"predict: {error}") from error
    cycled = _now()

    samples = int(directions.shape[0])
    counts = {
        "sim2exp": int(np.count_nonzero(directions == "sim2exp")),
        "exp2sim": int(np.count_nonzero(directions == "exp2sim")),
    }
    usage: dict[str, Usage] = {}
    for pool in inputs.pools:
        # Every staged pool's rows are in the final index, the diagnostic one too.
        records = int(pool["records"])
        usage[str(pool["name"])] = Usage(records, records, "index")
    log: list[dict[str, Any]] = [
        {
            "step": "prepare",
            "at": prepared.get("created_at"),
            "release_id": prepared.get("release_id"),
            "pools": [pool["name"] for pool in inputs.pools],
            "validation_pool": prepared.get("validation_pool"),
        },
        {
            "step": "index",
            "at": trained.get("created_at"),
            "pools": trained.get("pools"),
            "records": trained.get("records"),
            "validation": trained.get("validation"),
        },
        {
            "step": "predict",
            "at": started,
            "samples": samples,
            "directions": counts,
            "neighbors": NEIGHBOR_COUNT,
        },
        {
            "step": "cycle",
            "at": cycled,
            "samples": samples,
            "directions": {"sim2exp": counts["exp2sim"], "exp2sim": counts["sim2exp"]},
        },
    ]
    try:
        sizes = write_delivery(
            out_dir,
            manifest=inputs.manifest,
            usage=usage,
            sample_ids=queries["sample_id"],
            predictions=predictions,
            cycle_predictions=cycle_predictions,
            method=METHOD,
            report_markdown=_report(prepared, trained, counts),
            seed=int(trained["seed"]),
            log=log,
        )
    except ValueError as error:
        raise SolverError(f"predict: {error}") from error

    result: dict[str, Any] = {"samples": samples, "directions": counts, "files": sizes}
    _write_json(out_dir / "predict.json", result)
    return result


# --- command line -----------------------------------------------------------


def _parse_opts(raw: Sequence[str]) -> dict[str, str]:
    opts: dict[str, str] = {}
    for item in raw:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise SolverError(f"--opt expects key=value, got {item!r}", exit_status=2)
        opts[key] = value
    return opts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hyperspectrum.fewshot.solver",
        description="HyperData few-shot XAS solver (k-NN baseline).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_opt(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--opt",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="solver option; may repeat",
        )

    prepare_parser = commands.add_parser("prepare", help="verify and stage the release")
    prepare_parser.add_argument("--release-dir", type=Path, required=True)
    prepare_parser.add_argument("--work-dir", type=Path, required=True)
    add_opt(prepare_parser)

    train_parser = commands.add_parser("train", help="build the neighbour index")
    train_parser.add_argument("--work-dir", type=Path, required=True)
    train_parser.add_argument("--model-dir", type=Path, required=True)
    add_opt(train_parser)

    predict_parser = commands.add_parser("predict", help="write the delivery")
    predict_parser.add_argument("--release-dir", type=Path, required=True)
    predict_parser.add_argument("--work-dir", type=Path, required=True)
    predict_parser.add_argument("--model-dir", type=Path, required=True)
    predict_parser.add_argument("--out-dir", type=Path, required=True)
    add_opt(predict_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step; print its JSON summary last on stdout.

    Failures never show a traceback: a :class:`SolverError` is printed as is
    and any other exception as ``<command>: <ExceptionName>: <message>``,
    both on a single stderr line with exit status 1 (2 for usage errors).
    """
    args = _build_parser().parse_args(argv)
    command = str(args.command)
    try:
        opts = _parse_opts(list(args.opt))
        if command == "prepare":
            result = prepare(args.release_dir, args.work_dir, opts)
        elif command == "train":
            result = train(args.work_dir, args.model_dir, opts)
        else:
            result = predict(
                args.release_dir, args.work_dir, args.model_dir, args.out_dir, opts
            )
    except SolverError as error:
        print(_one_line(str(error)), file=sys.stderr)
        return error.exit_status
    except Exception as error:  # noqa: BLE001 -- the contract forbids tracebacks
        print(_one_line(f"{command}: {type(error).__name__}: {error}"), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
