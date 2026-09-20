"""Fixed ``prepare`` / ``train`` / ``predict`` contract for the few-shot kit.

The kind-agnostic kit drives this module as a subprocess::

    python -m hyperspectrum.fewshot.solver prepare --release-dir R --work-dir W
    python -m hyperspectrum.fewshot.solver train   --work-dir W --model-dir M
    python -m hyperspectrum.fewshot.solver predict --release-dir R --work-dir W \\
        --model-dir M --out-dir O

Each step writes its JSON summary (``W/prepare.json``, ``M/train.json``,
``O/predict.json``) and prints it as the last stdout line. ``--opt key=value``
may repeat; unknown keys exit 2. Expected failures exit non-zero with one
line on stderr and never a traceback. The same three steps are exposed as
plain functions for in-process use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections.abc import Mapping, Sequence
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
    QUERY_KEYS,
    PairIndex,
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


def _missing_paired_keys(keys: Sequence[str]) -> list[str]:
    return [key for key in PAIRED_POOL_KEYS if key not in keys]


def _check_paired_pool(pool: Mapping[str, NDArray[Any]], name: str) -> int:
    """Validate the full paired-pool layout and return its record count."""
    missing = _missing_paired_keys(list(pool))
    if missing:
        raise SolverError(
            f"prepare: {name} is not a paired pool; missing keys: {', '.join(missing)}"
        )
    energy = np.asarray(pool["energy"])
    if energy.shape != (GRID_POINTS,):
        raise SolverError(
            f"prepare: {name} energy must have shape ({GRID_POINTS},), got "
            f"{energy.shape}"
        )
    try:
        index = PairIndex.from_pools([pool])
    except ValueError as error:
        raise SolverError(f"prepare: {name}: {error}") from error
    return len(index)


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


def _paired_pool_keys(path: Path) -> list[str] | None:
    """Member names of an ``.npz`` without reading its arrays, or None."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            return list(archive.files)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return None


def prepare(
    release_dir: Path, work_dir: Path, opts: Mapping[str, str]
) -> dict[str, Any]:
    """Verify the release's paired pools and stage them into ``work_dir``.

    Every ``required: true`` file must be a paired pool whose bytes match the
    manifest's sha256. Optional files are staged only when they are paired
    pools too (single-domain pools and sidecars are skipped and listed under
    ``skipped``).
    """
    _check_opts(opts, PREPARE_OPTS, "prepare")
    manifest = _load_json_object(release_dir / "data_manifest.json", command="prepare")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SolverError("prepare: data_manifest.json lacks a 'files' object")
    grid = manifest.get("grid")
    if not isinstance(grid, dict) or grid.get("points") != GRID_POINTS:
        raise SolverError(
            f"prepare: unsupported grid {grid!r}; this solver needs "
            f"{GRID_POINTS} points"
        )
    release_id = manifest.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise SolverError("prepare: data_manifest.json lacks a release_id")

    pools_dir = work_dir / POOLS_SUBDIR
    pools_dir.mkdir(parents=True, exist_ok=True)
    pools: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for raw_name, entry in files.items():
        name = _safe_name(str(raw_name), command="prepare")
        if not isinstance(entry, dict):
            raise SolverError(f"prepare: manifest entry for {name} must be an object")
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
            keys = _paired_pool_keys(source) if name.endswith(".npz") else None
            if keys is None:
                skipped.append({"name": name, "reason": "not an npz archive"})
                continue
            missing = _missing_paired_keys(keys)
            if missing:
                skipped.append(
                    {
                        "name": name,
                        "reason": f"not a paired pool (missing {', '.join(missing)})",
                    }
                )
                continue
        expected = entry.get("sha256")
        actual = _sha256(source)
        if actual != expected:
            raise SolverError(
                f"prepare: sha256 mismatch for {name}: manifest says {expected}, "
                f"file has {actual}"
            )
        records = _check_paired_pool(_load_npz(source, command="prepare"), name)
        listed = entry.get("records")
        if listed is not None and int(listed) != records:
            raise SolverError(
                f"prepare: {name} holds {records} records but the manifest lists "
                f"{listed}"
            )
        staged = pools_dir / name
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, staged)
        pools.append(
            {
                "name": name,
                "records": records,
                "sha256": actual,
                "required": required,
                "role": "index",
            }
        )
    if not pools:
        raise SolverError("prepare: data_manifest.json lists no paired pools")

    # Every staged pool joins the final index (role "index"); the diagnostic
    # pool is additionally scored leak-free before being re-admitted.
    validation_pool = _select_validation_pool(
        [str(pool["name"]) for pool in pools], opts
    )
    for pool in pools:
        pool["diagnostic"] = pool["name"] == validation_pool

    result: dict[str, Any] = {
        "release_id": release_id,
        "grid": dict(grid),
        "pools": pools,
        "validation_pool": validation_pool,
        "skipped": skipped,
        "created_at": _now(),
    }
    _write_json(work_dir / "prepare.json", result)
    return result


def _staged_pool(
    work_dir: Path, pool: Mapping[str, Any], *, command: str
) -> dict[str, NDArray[Any]]:
    name = _safe_name(str(pool["name"]), command=command)
    path = work_dir / POOLS_SUBDIR / name
    if not path.is_file():
        raise SolverError(f"{command}: staged pool {name} is missing; rerun prepare")
    actual = _sha256(path)
    if actual != pool.get("sha256"):
        raise SolverError(f"{command}: staged pool {name} changed since prepare")
    return _load_npz(path, command=command)


def _prepared_pools(
    prepared: Mapping[str, Any], *, command: str
) -> list[dict[str, Any]]:
    pools = prepared.get("pools")
    if not isinstance(pools, list) or not all(isinstance(pool, dict) for pool in pools):
        raise SolverError(
            f"{command}: prepare.json lacks a 'pools' list; rerun prepare"
        )
    return [dict(pool) for pool in pools]


def _direction_mae(
    index: PairIndex, pool: Mapping[str, NDArray[Any]], direction: str
) -> float:
    input_key, output_key = (
        ("simulation", "experiment")
        if direction == "sim2exp"
        else ("experiment", "simulation")
    )
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
    from an index built from the other pools only, so its MAE is not a
    self-match. The final ``M/index.npz`` then re-admits it: every staged
    pool influences the delivered predictions, as the evaluator's data-usage
    rule demands of required pools.
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

    validation: dict[str, Any] | None = None
    diagnostic_name = prepared.get("validation_pool")
    if diagnostic_name is not None:
        diagnostic_name = str(diagnostic_name)
        if diagnostic_name not in staged:
            raise SolverError(
                f"train: diagnostic pool {diagnostic_name} is not staged; rerun prepare"
            )
        train_only = [staged[name] for name in names if name != diagnostic_name]
        if not train_only:
            raise SolverError(
                f"train: {diagnostic_name} is the only paired pool; nothing to "
                "score it against"
            )
        held_out = staged[diagnostic_name]
        try:
            diagnostic_index = PairIndex.from_pools(train_only)
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

    try:
        index = PairIndex.from_pools([staged[name] for name in names])
    except ValueError as error:
        raise SolverError(f"train: {error}") from error
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
        "| pool | records | required | diagnostic | final index |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for pool in prepared.get("pools") or []:
        table.append(
            f"| {pool.get('name')} | {pool.get('records')} | "
            f"{'yes' if pool.get('required') else 'no'} | "
            f"{'yes' if pool.get('diagnostic') else 'no'} | yes |"
        )
    unused = (
        "Single-domain public pools (simulation-only or experiment-only) are "
        "not read: the baseline needs paired records."
    )
    validation = trained.get("validation")
    if isinstance(validation, dict):
        others = [
            name for name in trained.get("pools") or [] if name != validation["name"]
        ]
        validation_text = (
            f"Diagnostic pool `{validation.get('name')}` ({validation.get('records')} "
            "pairs) was scored leak-free from a train-only index built without it "
            f"(pools: {', '.join(others)}): MAE sim2exp = "
            f"{validation.get('mae_sim2exp'):.6f}, MAE exp2sim = "
            f"{validation.get('mae_exp2sim'):.6f}. It was then re-admitted to the "
            "final index, so every paired pool influences the delivered "
            "predictions."
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
    prepared = _load_json_object(work_dir / "prepare.json", command="predict")
    trained = _load_json_object(model_dir / "train.json", command="predict")
    manifest = _load_json_object(release_dir / "data_manifest.json", command="predict")
    release_ids = {
        "data_manifest.json": manifest.get("release_id"),
        "prepare.json": prepared.get("release_id"),
        "train.json": trained.get("release_id"),
    }
    if len(set(release_ids.values())) != 1:
        raise SolverError(f"predict: release_id mismatch across {release_ids}")

    queries = _load_npz(release_dir / "queries.npz", command="predict")
    missing = [key for key in QUERY_KEYS if key not in queries]
    if missing:
        raise SolverError(f"predict: queries.npz is missing keys: {', '.join(missing)}")
    try:
        index = PairIndex.load(model_dir / "index.npz")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SolverError(f"predict: cannot load index.npz: {error}") from error

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
    pools = _prepared_pools(prepared, command="predict")
    usage: dict[str, Usage] = {}
    for pool in pools:
        # Every staged pool is in the final index, the diagnostic one included.
        records = int(pool["records"])
        usage[str(pool["name"])] = Usage(records, records, "index")
    log: list[dict[str, Any]] = [
        {
            "step": "prepare",
            "at": prepared.get("created_at"),
            "release_id": prepared.get("release_id"),
            "pools": [pool["name"] for pool in pools],
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
            manifest=manifest,
            usage=usage,
            sample_ids=queries["sample_id"],
            predictions=predictions,
            cycle_predictions=cycle_predictions,
            method=METHOD,
            report_markdown=_report(prepared, trained, counts),
            seed=int(trained.get("seed", DEFAULT_SEED)),
            log=log,
        )
    except ValueError as error:
        raise SolverError(f"predict: {error}") from error

    result: dict[str, Any] = {"samples": samples, "directions": counts, "files": sizes}
    _write_json(out_dir / "predict.json", result)
    return result


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
    """Run one step; print its JSON summary last on stdout."""
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
        print(str(error), file=sys.stderr)
        return error.exit_status
    except (OSError, ValueError) as error:
        print(f"{command}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
