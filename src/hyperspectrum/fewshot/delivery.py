"""Write the seven-file delivery the organiser's ``xas_evaluator.py`` consumes.

Everything the evaluator checks structurally is checked here first, so a
run that finishes has a delivery that at least parses: prediction ids in
query order, finite bounded values, data-usage rules against the release
manifest, at least four iteration-log lines, and the per-file byte limits.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .knn import GRID_POINTS

KIB = 1024
MIB = 1024 * KIB

#: Byte limit per delivered file, in the order the delivery is written.
LIMITS: dict[str, int] = {
    "predictions.npz": 64 * MIB,
    "cycle_predictions.npz": 64 * MIB,
    "method_report.md": 256 * KIB,
    "method_summary.json": 128 * KIB,
    "data_usage.json": 128 * KIB,
    "iteration_log.jsonl": 1 * MIB,
    "run_manifest.json": 128 * KIB,
}
DELIVERY_FILES: tuple[str, ...] = tuple(LIMITS)
MIN_LOG_LINES = 4
PREDICTION_MAGNITUDE_LIMIT = 1e6
SOLVER_ID = "hyperspectrum.fewshot/knn-baseline/1"


@dataclass(frozen=True)
class Usage:
    """How one release file was used: rows read, rows that shaped predictions."""

    scanned: int
    influencing: int
    stage: str

    def __post_init__(self) -> None:
        for field_name in ("scanned", "influencing"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer")
        if not isinstance(self.stage, str):
            raise TypeError("stage must be a string")
        if self.scanned < 0:
            raise ValueError("scanned must be >= 0")
        if self.influencing < 0:
            raise ValueError("influencing must be >= 0")
        if self.influencing > self.scanned:
            raise ValueError("influencing must not exceed scanned")


def _prediction_block(values: Any, *, name: str, count: int) -> NDArray[np.float32]:
    block = np.asarray(values)
    if block.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric, got dtype {block.dtype}")
    if block.shape != (count, GRID_POINTS):
        raise ValueError(
            f"{name} must have shape ({count}, {GRID_POINTS}), got {block.shape}"
        )
    if not np.isfinite(block).all():
        raise ValueError(f"{name} must be finite")
    if np.abs(block).max(initial=0.0) > PREDICTION_MAGNITUDE_LIMIT:
        raise ValueError(
            f"{name} must stay within |value| <= {PREDICTION_MAGNITUDE_LIMIT:g}"
        )
    return np.asarray(block, dtype=np.float32)


def _data_usage(
    manifest: Mapping[str, Any], usage: Mapping[str, Usage]
) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("data_manifest.json must carry a 'files' object")
    entries: dict[str, dict[str, Any]] = {}
    for name, used in usage.items():
        if not isinstance(used, Usage):
            raise TypeError(f"usage for {name} must be a Usage")
        entry = files.get(name)
        if entry is None:
            raise ValueError(f"data_usage lists {name}, which is not in the manifest")
        if not isinstance(entry, Mapping):
            raise TypeError(f"manifest entry for {name} must be an object")
        records = entry.get("records")
        if records is not None and used.scanned > int(records):
            raise ValueError(
                f"data_usage for {name} scanned {used.scanned} rows but the "
                f"manifest records {records}"
            )
        sha256 = entry.get("sha256")
        if not isinstance(sha256, str) or not sha256:
            raise ValueError(f"manifest entry for {name} lacks a sha256")
        entries[name] = {
            "scanned": used.scanned,
            "influencing": used.influencing,
            "stage": used.stage,
            "sha256": sha256,
        }
    for name, entry in files.items():
        if not isinstance(entry, Mapping) or not entry.get("required"):
            continue
        listed = entries.get(name)
        if listed is None:
            raise ValueError(f"required pool {name} is missing from data_usage")
        if int(listed["influencing"]) <= 0:
            raise ValueError(f"required pool {name} must have influencing > 0")
    return {"files": entries}


def _run_manifest(seed: int) -> dict[str, Any]:
    seed_text = str(seed)
    if not seed_text:
        raise ValueError("seed must be non-empty")
    return {
        "seed": seed_text,
        "dependencies": [
            f"numpy=={np.__version__}",
            f"hyperspectrum=={importlib.metadata.version('hyperspectrum')}",
        ],
        "python": platform.python_version(),
        "solver": SOLVER_ID,
    }


def _log_lines(log: Sequence[Mapping[str, Any]]) -> str:
    if len(log) < MIN_LOG_LINES:
        raise ValueError(
            f"iteration_log.jsonl needs at least {MIN_LOG_LINES} entries, got "
            f"{len(log)}"
        )
    lines = []
    for position, entry in enumerate(log):
        if not isinstance(entry, Mapping):
            raise TypeError(f"iteration_log entry {position} must be an object")
        lines.append(json.dumps(dict(entry), ensure_ascii=True))
    return "\n".join(lines) + "\n"


def write_delivery(
    out_dir: Path,
    *,
    manifest: Mapping[str, Any],
    usage: Mapping[str, Usage],
    sample_ids: NDArray[np.str_],
    predictions: NDArray[np.float32],
    cycle_predictions: NDArray[np.float32],
    method: Mapping[str, Any],
    report_markdown: str,
    seed: int,
    log: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Write all seven files into ``out_dir`` and return their byte sizes.

    Raises ``ValueError`` before touching the disk when the content would
    not satisfy the evaluator, and after writing when a file exceeds its
    byte limit (the message names the file).
    """
    ids = np.asarray(sample_ids).astype(str)
    if ids.ndim != 1:
        raise ValueError("sample_ids must be one-dimensional")
    count = int(ids.shape[0])
    prediction_block = _prediction_block(predictions, name="predictions", count=count)
    cycle_block = _prediction_block(
        cycle_predictions, name="cycle_predictions", count=count
    )
    if not isinstance(method, Mapping):
        raise TypeError("method summary must be an object")
    data_usage = _data_usage(manifest, usage)
    log_text = _log_lines(log)
    run_manifest = _run_manifest(seed)
    method_summary = json.dumps(dict(method), indent=2, ensure_ascii=False) + "\n"

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz", sample_id=ids, prediction=prediction_block
    )
    np.savez_compressed(
        out_dir / "cycle_predictions.npz", sample_id=ids, prediction=cycle_block
    )
    (out_dir / "method_report.md").write_text(report_markdown, encoding="utf-8")
    (out_dir / "method_summary.json").write_text(method_summary, encoding="utf-8")
    (out_dir / "data_usage.json").write_text(
        json.dumps(data_usage, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "iteration_log.jsonl").write_text(log_text, encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )

    sizes: dict[str, int] = {}
    for name, limit in LIMITS.items():
        size = (out_dir / name).stat().st_size
        if size > limit:
            raise ValueError(f"{name} is {size} bytes, over the {limit}-byte limit")
        sizes[name] = size
    return sizes
