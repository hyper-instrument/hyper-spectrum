"""The CLI contract every research image honours, and the helpers to satisfy it.

    <image> [--model-entrypoint <expr>] --data /data \
            --weights <path>[!<inner-path>] [--max-samples N] --out /out/metrics.json

``--model-entrypoint`` is **accepted and ignored**, and optional. No image reads
it: each of these containers embeds exactly one model's inference code, so there
is nothing for an entrypoint argument to select. It was ``required``, which made
every image demand an argument none of them used.

It is not simply deleted, because images already built against the old contract
still refuse to start without it — dropping the flag would strand every prebuilt
image and force a rebuild to no purpose. So the platform keeps sending it, this
parser keeps tolerating it, and images built from here on do not need it. When no
prebuilt image predates this note, the sending side can stop.

The one genuinely tricky flag is ``--weights``. It carries both materializations
in a single string because the platform stores which one applies and the image
must not guess:

- ``/weights/bfo.tar``  — the file is handed over whole. AtomAI's ``bfo.tar`` and
  StarDist's ``.zip`` are archives their own loaders open; extracting them first
  gives the loader something it does not expect.
- ``/archive/MPGAN.tar.gz!trained_models/mp_g/G_best_epoch.pt`` — the weights live
  *inside* a repo tarball. Here extraction is required, and the inner path says
  what to reach for.

Deriving this from the file extension gets both wrong, and only inside the
container, twenty minutes into a run.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import tarfile
import time
import zipfile
from pathlib import Path

#: Extraction cache. A tarball is unpacked once per image lifetime rather than per
#: run; on the jet datasets that is minutes of I/O each time.
CACHE_ROOT = Path(os.environ.get("ACEBENCH_UNPACK_CACHE", "/tmp/acebench-unpack"))

ARCHIVE_SEPARATOR = "!"


class WeightsError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    # Accepted for compatibility with images built against the older contract,
    # where it was required. Nothing reads `args.model_entrypoint`.
    parser.add_argument("--model-entrypoint", default="")
    parser.add_argument("--data", default="/data")
    parser.add_argument("--weights", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", default="/out/metrics.json")
    return parser.parse_args(argv)


def resolve_weights(spec: str, *, cache_root: Path | None = None) -> Path:
    """Turn one ``--weights`` value into a path the model can load.

    Without ``!`` the path is returned untouched — that is the whole point of
    ``mount-file``. With ``!`` the archive is extracted (once, cached by name and
    mtime) and the inner path resolved inside it.
    """
    if ARCHIVE_SEPARATOR not in spec:
        path = Path(spec)
        if not path.exists():
            raise WeightsError(f"weights not found: {path}")
        return path

    archive_part, _, inner = spec.partition(ARCHIVE_SEPARATOR)
    archive = Path(archive_part)
    if not inner:
        raise WeightsError(f"{spec!r} has an archive separator but no inner path")
    if not archive.is_file():
        raise WeightsError(f"archive not found: {archive}")

    target = _extract(archive, cache_root or CACHE_ROOT)
    resolved = target / inner
    if not resolved.exists():
        # Listing what IS there turns "file not found" into something actionable:
        # inner paths are transcribed by hand and usually wrong by one directory.
        available = sorted(p.name for p in target.iterdir())[:12]
        raise WeightsError(
            f"{inner!r} not found inside {archive.name}; top level holds: {available}"
        )
    return resolved


def _extract(archive: Path, cache_root: Path) -> Path:
    stat = archive.stat()
    target = cache_root / f"{archive.name}-{int(stat.st_mtime)}-{stat.st_size}"
    marker = target / ".complete"
    if marker.exists():
        return target

    # Extract to a sibling and rename, so a crashed extraction never leaves a
    # half-populated directory that the next run would treat as a cache hit.
    staging = target.with_suffix(".partial")
    if staging.exists():
        _rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            _safe_extract_zip(zf, staging)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            _safe_extract_tar(tf, staging)
    else:
        raise WeightsError(f"{archive.name} is neither a zip nor a tar archive")

    # A repo tarball almost always wraps everything in one top-level directory
    # (`MPGAN-8e2fd43/`), which no hand-written inner path includes. Descending
    # through a lone directory is what makes those paths work as written.
    entries = list(staging.iterdir())
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging

    target.parent.mkdir(parents=True, exist_ok=True)
    root.rename(target) if root is not staging else staging.rename(target)
    if root is not staging:
        _rmtree(staging)
    (target / ".complete").write_text(str(time.time()))
    return target


def _within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    # These archives come from the internet. A member named ../../etc/passwd
    # extracts exactly where it says unless someone checks.
    for member in tf.getmembers():
        if member.issym() or member.islnk():
            continue
        if not _within(dest, dest / member.name):
            raise WeightsError(f"refusing to extract {member.name!r} outside the target")
    # `filter="data"` is the stdlib's own guard (3.12+) and becomes the default in
    # 3.14; passing it explicitly keeps behaviour identical across both and drops
    # the deprecation warning. The check above stays: it produces a message naming
    # the member, where the filter raises a generic one.
    tf.extractall(dest, filter="data")  # noqa: S202 - members validated above


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    # zipfile.extractall can create symlinks (Python 3.11+), and a member whose
    # *target* points outside the extraction tree is just as dangerous as a member
    # whose own name does. Skip symlinks entirely, mirroring the tar branch.
    for info in zf.infolist():
        if stat.S_ISLNK(info.external_attr >> 16):
            continue
        if not _within(dest, dest / info.filename):
            raise WeightsError(
                f"refusing to extract {info.filename!r} outside the target"
            )
        zf.extract(info, dest)


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def write_metrics(
    out: str | Path,
    *,
    metrics: dict[str, float],
    n_samples: int,
    duration_s: float,
    artifacts: list[dict] | None = None,
) -> None:
    """Emit the file the platform reads back. camelCase to match the wire model."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "nSamples": n_samples,
                "durationS": round(duration_s, 3),
                "artifacts": artifacts or [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
