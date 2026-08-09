"""Cu-CHA operando XAS dataset profile."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from hyperspectrum.denoising.evaluation import DenoisingPair
from hyperspectrum.denoising.sample import SpectrumAxis, SpectrumSample
from hyperspectrum.denoising.split import SplitEntry, SplitManifest, SplitName

EXPECTED_COLUMNS = (
    "musst_enestep",
    "mu_trans",
    "mu_ref",
    "I0",
    "I1",
    "I2",
)
_EXPERIMENT = re.compile(
    r"^(?P<loading>High-Cu|Low-Cu)_(?P<protocol>exposure|cycles)_.+$"
)
_SCAN_NAME = re.compile(
    r"^(?P<scan>[0-9]+)_at_(?P<temperature>-?[0-9]+)C_(?P<segment>.+)\.dat$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_CORRUPTION_CONTRACT = "hyperspectrum-cu-cha-poisson/v1"
PROFILE_SCHEMA_VERSION = "hyperspectrum-xas-denoising-profile/v2"
SOURCE_DECLARATION_SCHEMA_VERSION = "hyperspectrum-cu-cha-source-declaration/v1"
DATASET_CODE = "zenodo-10159154"
DOSE_FRACTIONS = (0.10, 0.25, 0.50)
_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd


class SpectrumRejected(ValueError):
    """One source spectrum failed a stable scientific admission rule."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ParsedCuChaSpectrum:
    """One strictly admitted Cu-CHA operando transmission scan."""

    energy_ev: npt.NDArray[np.float64]
    mu_trans: npt.NDArray[np.float64]
    mu_ref: npt.NDArray[np.float64]
    i0: npt.NDArray[np.float64]
    i1: npt.NDArray[np.float64]
    i2: npt.NDArray[np.float64]
    columns: tuple[str, ...] = EXPECTED_COLUMNS


@dataclass(frozen=True)
class CuChaSpectrumIdentity:
    """Leakage-safe semantic identity parsed from one normalized source path."""

    sample_id: str
    experiment: str
    condition_segment: str
    group_id: str
    cu_loading: str
    protocol_family: str
    scan_index: int
    temperature_c: int


@dataclass(frozen=True)
class PoissonThinnedTransmission:
    """A deterministic reduced-dose transmission observation."""

    noisy_mu_trans: npt.NDArray[np.float64]
    noisy_i0: npt.NDArray[np.float64]
    noisy_i1: npt.NDArray[np.float64]
    dose_fraction: float
    seed: int


@dataclass(frozen=True)
class CuChaTrackResult:
    """Immutable paths and digests for one reduced-dose track."""

    dose_fraction: float
    directory: Path
    benchmark_path: Path
    inference_path: Path
    manifest_path: Path
    benchmark_sha256: str
    inference_sha256: str


@dataclass(frozen=True)
class CuChaMaterializationResult:
    """Completed Cu-CHA v2 profile bundle."""

    output_directory: Path
    manifest_path: Path
    split_path: Path
    tracks: tuple[CuChaTrackResult, ...]
    source_declaration_digest: str
    source_content_manifest_digest: str
    split_manifest_digest: str
    profile_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-safe result envelope used by agents and the CLI."""

        return {
            "schema_version": "hyperspectrum-cu-cha-materialization-result/v2",
            "output_directory": str(self.output_directory),
            "manifest_path": str(self.manifest_path),
            "split_path": str(self.split_path),
            "source_declaration_digest": self.source_declaration_digest,
            "source_content_manifest_digest": self.source_content_manifest_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "profile_sha256": self.profile_sha256,
            "track_count": len(self.tracks),
            "tracks": [
                {
                    "dose_fraction": track.dose_fraction,
                    "directory": str(track.directory),
                    "benchmark_path": str(track.benchmark_path),
                    "inference_path": str(track.inference_path),
                    "manifest_path": str(track.manifest_path),
                    "benchmark_sha256": track.benchmark_sha256,
                    "inference_sha256": track.inference_sha256,
                }
                for track in self.tracks
            ],
        }


@dataclass(frozen=True)
class _SourceFile:
    path: str
    size: int
    sha256: str
    content: bytes


@dataclass(frozen=True)
class _NodeSnapshot:
    device: int
    inode: int
    size: int
    mode_type: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _AdmittedSpectrum:
    source: _SourceFile
    identity: CuChaSpectrumIdentity
    spectrum: ParsedCuChaSpectrum


def derive_cu_cha_seed(
    *,
    dataset_version: str,
    source_sha256: str,
    source_path: str,
    dose_fraction: float,
    contract_version: str = DEFAULT_CORRUPTION_CONTRACT,
) -> int:
    """Derive a stable 64-bit RNG seed from all corruption identity inputs."""

    if not dataset_version.strip() or not contract_version.strip():
        raise ValueError("dataset and corruption contract versions must be non-blank")
    if _SHA256.fullmatch(source_sha256) is None:
        raise ValueError("source_sha256 must be a lowercase SHA-256")
    parsed = PurePosixPath(source_path)
    if (
        not source_path
        or source_path.startswith("/")
        or "\\" in source_path
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != source_path
    ):
        raise ValueError("source_path must be normalized and relative")
    if not np.isfinite(dose_fraction) or not 0.0 < dose_fraction <= 1.0:
        raise ValueError("dose_fraction must be finite and in (0, 1]")
    payload = {
        "schema_version": "hyperspectrum-cu-cha-seed/v1",
        "contract_version": contract_version,
        "dataset_version": dataset_version,
        "dose_fraction": dose_fraction,
        "source_path": source_path,
        "source_sha256": source_sha256,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def poisson_thin_transmission(
    spectrum: ParsedCuChaSpectrum,
    *,
    dose_fraction: float,
    seed: int,
) -> PoissonThinnedTransmission:
    """Poisson-thin I0/I1 independently and recompute the transmission signal."""

    if not np.isfinite(dose_fraction) or not 0.0 < dose_fraction <= 1.0:
        raise ValueError("dose_fraction must be finite and in (0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    generator = np.random.Generator(np.random.PCG64(seed))
    noisy_i0 = generator.poisson(dose_fraction * spectrum.i0).astype(np.float64)
    noisy_i1 = generator.poisson(dose_fraction * spectrum.i1).astype(np.float64)
    noisy_i0 /= dose_fraction
    noisy_i1 /= dose_fraction
    if not np.all(noisy_i0 > 0) or not np.all(noisy_i1 > 0):
        raise SpectrumRejected(
            "zero_thinned_count",
            "Poisson thinning produced a zero count; no epsilon correction is allowed",
        )
    return PoissonThinnedTransmission(
        noisy_mu_trans=_readonly(np.log(noisy_i0 / noisy_i1)),
        noisy_i0=_readonly(noisy_i0),
        noisy_i1=_readonly(noisy_i1),
        dose_fraction=dose_fraction,
        seed=seed,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_npz(arrays: Mapping[str, npt.NDArray[np.generic]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(
                array_buffer,
                np.asanyarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_buffer.getvalue())
    return output.getvalue()


def _normalized_source_path(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("source integrity declaration path must be a string")
    parsed = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != value
    ):
        raise ValueError(
            "source integrity declaration path must be normalized and relative"
        )
    return value


def _snapshot(node_stat: os.stat_result) -> _NodeSnapshot:
    return _NodeSnapshot(
        device=node_stat.st_dev,
        inode=node_stat.st_ino,
        size=node_stat.st_size,
        mode_type=stat.S_IFMT(node_stat.st_mode),
        mtime_ns=node_stat.st_mtime_ns,
        ctime_ns=node_stat.st_ctime_ns,
    )


def _same_opened_node(
    opened: os.stat_result,
    expected: _NodeSnapshot,
    *,
    include_content_state: bool,
) -> bool:
    identity_matches = (
        opened.st_dev == expected.device
        and opened.st_ino == expected.inode
        and stat.S_IFMT(opened.st_mode) == expected.mode_type
    )
    if not include_content_state:
        return identity_matches
    return (
        identity_matches
        and opened.st_size == expected.size
        and opened.st_mtime_ns == expected.mtime_ns
        and opened.st_ctime_ns == expected.ctime_ns
    )


def _secure_read_source_file(
    *,
    source_root: Path,
    relative_path: str,
    root_snapshot: _NodeSnapshot,
    node_snapshots: Mapping[str, _NodeSnapshot],
) -> bytes:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    if (
        any(not hasattr(os, name) for name in required_flags)
        or not _OPEN_SUPPORTS_DIR_FD
    ):
        raise ValueError(
            "source integrity secure no-follow primitives are unavailable on this platform"
        )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptors: list[int] = []
    opened_directories: list[tuple[int, _NodeSnapshot]] = []
    try:
        root_fd = os.open(source_root, directory_flags)
        descriptors.append(root_fd)
        if not _same_opened_node(
            os.fstat(root_fd), root_snapshot, include_content_state=True
        ):
            raise ValueError("source integrity secure no-follow root identity changed")
        opened_directories.append((root_fd, root_snapshot))

        current_fd = root_fd
        parts = PurePosixPath(relative_path).parts
        traversed: list[str] = []
        for part in parts[:-1]:
            traversed.append(part)
            directory_fd = os.open(part, directory_flags, dir_fd=current_fd)
            descriptors.append(directory_fd)
            expected = node_snapshots.get("/".join(traversed))
            opened = os.fstat(directory_fd)
            if (
                expected is None
                or expected.mode_type != stat.S_IFDIR
                or not stat.S_ISDIR(opened.st_mode)
                or not _same_opened_node(opened, expected, include_content_state=True)
            ):
                raise ValueError(
                    "source integrity secure no-follow directory identity changed"
                )
            opened_directories.append((directory_fd, expected))
            current_fd = directory_fd

        file_fd = os.open(parts[-1], file_flags, dir_fd=current_fd)
        descriptors.append(file_fd)
        expected_file = node_snapshots.get(relative_path)
        before = os.fstat(file_fd)
        if (
            expected_file is None
            or expected_file.mode_type != stat.S_IFREG
            or not stat.S_ISREG(before.st_mode)
            or not _same_opened_node(before, expected_file, include_content_state=True)
        ):
            raise ValueError("source integrity secure no-follow file identity changed")

        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        if not _same_opened_node(
            after, expected_file, include_content_state=True
        ) or _snapshot(after) != _snapshot(before):
            raise ValueError(
                "source integrity secure no-follow file changed while reading"
            )
        if any(
            not _same_opened_node(
                os.fstat(directory_fd),
                expected_directory,
                include_content_state=True,
            )
            for directory_fd, expected_directory in opened_directories
        ):
            raise ValueError(
                "source integrity secure no-follow directory changed while reading"
            )
        return b"".join(chunks)
    except OSError as error:
        raise ValueError("source integrity secure no-follow read failed") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _load_and_verify_source(
    source_root: Path,
    source_declaration_file: Path,
) -> tuple[dict[str, str], tuple[_SourceFile, ...], str, str]:
    try:
        raw = json.loads(source_declaration_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("source integrity declaration is not readable JSON") from error
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "dataset", "files"}:
        raise ValueError("source integrity declaration has invalid fields")
    if raw["schema_version"] != SOURCE_DECLARATION_SCHEMA_VERSION:
        raise ValueError("source integrity declaration schema is unsupported")
    dataset = raw["dataset"]
    if not isinstance(dataset, dict) or set(dataset) != {
        "code",
        "version",
        "version_id",
    }:
        raise ValueError("source integrity dataset identity is invalid")
    if dataset.get("code") != DATASET_CODE:
        raise ValueError(f"source integrity dataset code must be {DATASET_CODE}")
    if any(
        not isinstance(dataset.get(name), str) or not cast(str, dataset[name]).strip()
        for name in ("version", "version_id")
    ):
        raise ValueError("source integrity dataset version fields must be non-blank")

    raw_files = raw["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("source integrity declaration files must be a non-empty array")
    declarations: list[tuple[str, int, str]] = []
    for item in raw_files:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ValueError("source integrity file declaration has invalid fields")
        path = _normalized_source_path(item["path"])
        size = item["size"]
        sha256 = item["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                "source integrity declared size must be a non-negative integer"
            )
        if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
            raise ValueError(
                "source integrity declared digest must be a lowercase SHA-256"
            )
        declarations.append((path, size, sha256))
    declarations.sort()
    paths = tuple(path for path, _, _ in declarations)
    if len(paths) != len(set(paths)):
        raise ValueError("source integrity declaration paths must be unique")

    try:
        root_stat = source_root.lstat()
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError("source integrity tree must be a real directory")
        observed_entries = tuple(source_root.rglob("*"))
        node_snapshots: dict[str, _NodeSnapshot] = {}
        observed_files: list[str] = []
        for observed_path in observed_entries:
            relative = observed_path.relative_to(source_root).as_posix()
            node_stat = observed_path.lstat()
            if stat.S_ISLNK(node_stat.st_mode) or not (
                stat.S_ISDIR(node_stat.st_mode) or stat.S_ISREG(node_stat.st_mode)
            ):
                raise ValueError(
                    "source integrity tree may contain only real directories and regular files"
                )
            node_snapshots[relative] = _snapshot(node_stat)
            if stat.S_ISREG(node_stat.st_mode):
                observed_files.append(relative)
        observed_paths = tuple(sorted(observed_files))
    except OSError as error:
        raise ValueError("source integrity tree is not readable") from error
    if observed_paths != paths:
        raise ValueError("source integrity tree does not exactly match the declaration")

    verified: list[_SourceFile] = []
    for relative_path, declared_size, declared_sha256 in declarations:
        content = _secure_read_source_file(
            source_root=source_root,
            relative_path=relative_path,
            root_snapshot=_snapshot(root_stat),
            node_snapshots=node_snapshots,
        )
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if len(content) != declared_size or actual_sha256 != declared_sha256:
            raise ValueError(f"source integrity mismatch: {relative_path}")
        verified.append(
            _SourceFile(
                path=relative_path,
                size=declared_size,
                sha256=actual_sha256,
                content=content,
            )
        )
    canonical_declaration = {
        "schema_version": SOURCE_DECLARATION_SCHEMA_VERSION,
        "dataset": dataset,
        "files": [
            {"path": path, "size": size, "sha256": sha256}
            for path, size, sha256 in declarations
        ],
    }
    content_manifest = {
        "schema_version": "hyperspectrum-source-content-manifest/v1",
        "files": [
            {"path": item.path, "size": item.size, "sha256": item.sha256}
            for item in verified
        ],
    }
    dataset_identity = {name: cast(str, dataset[name]) for name in dataset}
    return (
        dataset_identity,
        tuple(verified),
        _canonical_digest(canonical_declaration),
        _canonical_digest(content_manifest),
    )


def _parse_verified_spectra(
    files: tuple[_SourceFile, ...],
) -> tuple[_AdmittedSpectrum, ...]:
    admitted: list[_AdmittedSpectrum] = []
    for source in files:
        identity = parse_cu_cha_identity(source.path)
        spectrum = _parse_cu_cha_dat_bytes(source.content)
        admitted.append(
            _AdmittedSpectrum(source=source, identity=identity, spectrum=spectrum)
        )
    admitted.sort(key=lambda item: item.identity.sample_id)
    point_counts = {len(item.spectrum.energy_ev) for item in admitted}
    if len(point_counts) != 1:
        raise ValueError("source integrity profile requires one common point count")
    return tuple(admitted)


def _split_payload(split_manifest: SplitManifest) -> dict[str, object]:
    return {
        "schema_version": split_manifest.schema_version,
        "digest": split_manifest.digest,
        "entries": [
            {
                "sample_id": entry.sample_id,
                "group_id": entry.group_id,
                "split": entry.split,
            }
            for entry in split_manifest.entries
        ],
    }


def _split_stratum_counts(
    identities: tuple[CuChaSpectrumIdentity, ...],
    split_manifest: SplitManifest,
) -> list[dict[str, object]]:
    assignment_by_sample = {
        entry.sample_id: entry.split for entry in split_manifest.entries
    }
    summaries: list[dict[str, object]] = []
    strata = sorted(
        {(identity.cu_loading, identity.protocol_family) for identity in identities}
    )
    for cu_loading, protocol_family in strata:
        selected = [
            identity
            for identity in identities
            if identity.cu_loading == cu_loading
            and identity.protocol_family == protocol_family
        ]
        split_group_ids: dict[SplitName, set[str]] = {
            "train": set(),
            "val": set(),
            "test": set(),
        }
        split_sample_counts: dict[SplitName, int] = {
            "train": 0,
            "val": 0,
            "test": 0,
        }
        for identity in selected:
            split = assignment_by_sample[identity.sample_id]
            split_group_ids[split].add(identity.group_id)
            split_sample_counts[split] += 1
        summaries.append(
            {
                "cu_loading": cu_loading,
                "protocol_family": protocol_family,
                "group_count": len({identity.group_id for identity in selected}),
                "sample_count": len(selected),
                "split_group_counts": {
                    name: len(split_group_ids[name])
                    for name in ("train", "val", "test")
                },
                "split_sample_counts": {
                    name: split_sample_counts[name] for name in ("train", "val", "test")
                },
            }
        )
    return summaries


def materialize_cu_cha(
    *,
    source_root: Path,
    source_declaration_file: Path,
    output_directory: Path,
) -> CuChaMaterializationResult:
    """Materialize the Cu-CHA full-count proxy into three deterministic dose tracks."""

    if output_directory.exists():
        raise FileExistsError(f"output directory already exists: {output_directory}")
    (
        dataset,
        verified_files,
        declaration_digest,
        content_manifest_digest,
    ) = _load_and_verify_source(source_root, source_declaration_file)
    admitted = _parse_verified_spectra(verified_files)
    split_manifest = build_grouped_balanced_split(
        tuple(item.identity for item in admitted)
    )
    split_by_sample = {entry.sample_id: entry.split for entry in split_manifest.entries}

    energy = np.stack([item.spectrum.energy_ev for item in admitted])
    target = np.stack([item.spectrum.mu_trans for item in admitted])
    i0 = np.stack([item.spectrum.i0 for item in admitted])
    i1 = np.stack([item.spectrum.i1 for item in admitted])
    i2 = np.stack([item.spectrum.i2 for item in admitted])
    sample_ids = np.asarray([item.identity.sample_id for item in admitted])
    group_ids = np.asarray([item.identity.group_id for item in admitted])
    experiments = np.asarray([item.identity.experiment for item in admitted])
    segments = np.asarray([item.identity.condition_segment for item in admitted])
    cu_loadings = np.asarray([item.identity.cu_loading for item in admitted])
    protocol_families = np.asarray([item.identity.protocol_family for item in admitted])
    source_paths = np.asarray([item.source.path for item in admitted])
    source_sha256 = np.asarray([item.source.sha256 for item in admitted])
    splits = np.asarray([split_by_sample[item.identity.sample_id] for item in admitted])

    track_payloads: list[dict[str, object]] = []
    track_file_bytes: list[tuple[str, bytes, bytes, bytes]] = []
    for dose_fraction in DOSE_FRACTIONS:
        thinned: list[PoissonThinnedTransmission] = []
        seeds: list[int] = []
        for item in admitted:
            seed = derive_cu_cha_seed(
                dataset_version=dataset["version_id"],
                source_sha256=item.source.sha256,
                source_path=item.source.path,
                dose_fraction=dose_fraction,
            )
            seeds.append(seed)
            thinned.append(
                poisson_thin_transmission(
                    item.spectrum,
                    dose_fraction=dose_fraction,
                    seed=seed,
                )
            )
        noisy = np.stack([item.noisy_mu_trans for item in thinned])
        noisy_i0 = np.stack([item.noisy_i0 for item in thinned])
        noisy_i1 = np.stack([item.noisy_i1 for item in thinned])
        benchmark_bytes = _canonical_npz(
            {
                "condition_segments": segments,
                "cu_loadings": cu_loadings,
                "energy": energy,
                "energy_unit": np.asarray("eV"),
                "experiments": experiments,
                "group_ids": group_ids,
                "i0_full_count": i0,
                "i1_full_count": i1,
                "i2_full_count": i2,
                "noisy": noisy,
                "noisy_i0_counts": noisy_i0,
                "noisy_i1_counts": noisy_i1,
                "proxy_full_count": target,
                "protocol_families": protocol_families,
                "sample_ids": sample_ids,
                "sample_seeds": np.asarray(seeds, dtype=np.uint64),
                "source_paths": source_paths,
                "source_sha256": source_sha256,
                "splits": splits,
            }
        )
        inference_bytes = _canonical_npz(
            {
                "energy": energy,
                "energy_unit": np.asarray("eV"),
                "group_ids": group_ids,
                "noisy": noisy,
                "sample_ids": sample_ids,
            }
        )
        benchmark_digest = hashlib.sha256(benchmark_bytes).hexdigest()
        inference_digest = hashlib.sha256(inference_bytes).hexdigest()
        directory_name = f"dose-{dose_fraction:.2f}"
        track_manifest = {
            "schema_version": "hyperspectrum-xas-denoising-track/v2",
            "profile_schema_version": PROFILE_SCHEMA_VERSION,
            "dataset_code": DATASET_CODE,
            "dose_fraction": dose_fraction,
            "corruption": {
                "algorithm": "Poisson(dose_fraction * count) / dose_fraction",
                "channels": ["I0", "I1"],
                "contract_version": DEFAULT_CORRUPTION_CONTRACT,
                "zero_count_policy": "reject_without_epsilon_correction",
            },
            "target": {
                "ground_truth_kind": "proxy_full_count",
                "is_physical_noiseless_ground_truth": False,
                "signal": "mu_trans_full_count",
            },
            "split_manifest_digest": split_manifest.digest,
            "benchmark_asset": {
                "path": "benchmark.npz",
                "sha256": benchmark_digest,
            },
            "inference_asset": {
                "path": "inference.npz",
                "sha256": inference_digest,
                "contract": "hyperspectrum-local-inference-npz/v1",
            },
        }
        track_manifest_bytes = _canonical_json_bytes(track_manifest) + b"\n"
        track_file_bytes.append(
            (directory_name, benchmark_bytes, inference_bytes, track_manifest_bytes)
        )
        track_payloads.append(
            {
                "dose_fraction": dose_fraction,
                "directory": f"tracks/{directory_name}",
                "benchmark_asset_sha256": benchmark_digest,
                "inference_asset_sha256": inference_digest,
            }
        )

    root_manifest = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_code": "cu-cha-operando-full-count-proxy",
        "dataset": dataset,
        "source": {
            "declaration_digest": declaration_digest,
            "content_manifest_digest": content_manifest_digest,
            "file_count": len(verified_files),
        },
        "signal": {
            "modality": "xas",
            "representation": "dense",
            "axis_name": "energy",
            "axis_unit": "eV",
            "input": "poisson_thinned_log_ratio",
        },
        "target": {
            "ground_truth_kind": "proxy_full_count",
            "is_physical_noiseless_ground_truth": False,
            "signal": "mu_trans_full_count",
        },
        "split": {
            "manifest_path": "split.json",
            "manifest_digest": split_manifest.digest,
            "group_key": "experiment/condition_segment",
            "policy_version": "hyperspectrum-cu-cha-grouped-balanced-split/v1",
            "strata": ["cu_loading", "protocol_family"],
            "stratum_counts": _split_stratum_counts(
                tuple(item.identity for item in admitted), split_manifest
            ),
        },
        "tracks": track_payloads,
        "compatibility": {
            "hyperspectrum_core": "DenoisingPair",
            "local_execution": "tracks/*/inference.npz",
            "ace_v3": "requires_profile_v2_adapter",
            "v1_xanes_contract_reused": False,
        },
        "execution_boundary": {
            "candidate_container_asset": "inference_only",
            "leaderboard_scored_split": "test",
            "scorer_only_asset": "benchmark_with_proxy_target",
        },
    }
    root_manifest_bytes = _canonical_json_bytes(root_manifest) + b"\n"
    split_bytes = _canonical_json_bytes(_split_payload(split_manifest)) + b"\n"

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.tmp-",
            dir=output_directory.parent,
        )
    )
    try:
        (temporary_directory / "manifest.json").write_bytes(root_manifest_bytes)
        (temporary_directory / "split.json").write_bytes(split_bytes)
        for (
            directory_name,
            benchmark_bytes,
            inference_bytes,
            track_manifest_bytes,
        ) in track_file_bytes:
            track_directory = temporary_directory / "tracks" / directory_name
            track_directory.mkdir(parents=True)
            (track_directory / "benchmark.npz").write_bytes(benchmark_bytes)
            (track_directory / "inference.npz").write_bytes(inference_bytes)
            (track_directory / "manifest.json").write_bytes(track_manifest_bytes)
        if output_directory.exists():
            raise FileExistsError(
                f"output directory already exists: {output_directory}"
            )
        temporary_directory.rename(output_directory)
    except BaseException:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        raise

    track_results = tuple(
        CuChaTrackResult(
            dose_fraction=float(directory_name.removeprefix("dose-")),
            directory=output_directory / "tracks" / directory_name,
            benchmark_path=output_directory
            / "tracks"
            / directory_name
            / "benchmark.npz",
            inference_path=output_directory
            / "tracks"
            / directory_name
            / "inference.npz",
            manifest_path=output_directory
            / "tracks"
            / directory_name
            / "manifest.json",
            benchmark_sha256=hashlib.sha256(benchmark_bytes).hexdigest(),
            inference_sha256=hashlib.sha256(inference_bytes).hexdigest(),
        )
        for directory_name, benchmark_bytes, inference_bytes, _ in track_file_bytes
    )
    return CuChaMaterializationResult(
        output_directory=output_directory,
        manifest_path=output_directory / "manifest.json",
        split_path=output_directory / "split.json",
        tracks=track_results,
        source_declaration_digest=declaration_digest,
        source_content_manifest_digest=content_manifest_digest,
        split_manifest_digest=split_manifest.digest,
        profile_sha256=hashlib.sha256(root_manifest_bytes).hexdigest(),
    )


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object")
    return cast(dict[str, Any], value)


def _validate_profile_manifest(
    root: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    expected_target = {
        "ground_truth_kind": "proxy_full_count",
        "is_physical_noiseless_ground_truth": False,
        "signal": "mu_trans_full_count",
    }
    expected_signal = {
        "modality": "xas",
        "representation": "dense",
        "axis_name": "energy",
        "axis_unit": "eV",
        "input": "poisson_thinned_log_ratio",
    }
    expected_compatibility = {
        "hyperspectrum_core": "DenoisingPair",
        "local_execution": "tracks/*/inference.npz",
        "ace_v3": "requires_profile_v2_adapter",
        "v1_xanes_contract_reused": False,
    }
    expected_execution_boundary = {
        "candidate_container_asset": "inference_only",
        "leaderboard_scored_split": "test",
        "scorer_only_asset": "benchmark_with_proxy_target",
    }
    if (
        set(root)
        != {
            "schema_version",
            "profile_code",
            "dataset",
            "source",
            "signal",
            "target",
            "split",
            "tracks",
            "compatibility",
            "execution_boundary",
        }
        or root.get("schema_version") != PROFILE_SCHEMA_VERSION
        or root.get("profile_code") != "cu-cha-operando-full-count-proxy"
        or root.get("signal") != expected_signal
        or root.get("target") != expected_target
        or root.get("compatibility") != expected_compatibility
        or root.get("execution_boundary") != expected_execution_boundary
    ):
        raise ValueError("profile manifest has an invalid v2 root contract")

    dataset = root["dataset"]
    if (
        not isinstance(dataset, dict)
        or set(dataset) != {"code", "version", "version_id"}
        or dataset.get("code") != DATASET_CODE
        or any(
            not isinstance(dataset.get(field), str)
            or not cast(str, dataset[field]).strip()
            for field in ("version", "version_id")
        )
    ):
        raise ValueError("profile manifest dataset identity is invalid")

    source = root["source"]
    if (
        not isinstance(source, dict)
        or set(source)
        != {"declaration_digest", "content_manifest_digest", "file_count"}
        or any(
            not isinstance(source.get(field), str)
            or _SHA256.fullmatch(cast(str, source[field])) is None
            for field in ("declaration_digest", "content_manifest_digest")
        )
        or isinstance(source.get("file_count"), bool)
        or not isinstance(source.get("file_count"), int)
        or cast(int, source["file_count"]) < 1
    ):
        raise ValueError("profile manifest source identity is invalid")

    split = root["split"]
    if (
        not isinstance(split, dict)
        or set(split)
        != {
            "manifest_path",
            "manifest_digest",
            "group_key",
            "policy_version",
            "strata",
            "stratum_counts",
        }
        or split.get("manifest_path") != "split.json"
        or not isinstance(split.get("manifest_digest"), str)
        or _SHA256.fullmatch(cast(str, split["manifest_digest"])) is None
        or split.get("group_key") != "experiment/condition_segment"
        or split.get("policy_version")
        != "hyperspectrum-cu-cha-grouped-balanced-split/v1"
        or split.get("strata") != ["cu_loading", "protocol_family"]
    ):
        raise ValueError("profile manifest split identity is invalid")
    raw_stratum_counts = split["stratum_counts"]
    if not isinstance(raw_stratum_counts, list) or not raw_stratum_counts:
        raise ValueError("profile manifest split stratum counts are invalid")
    previous_stratum: tuple[str, str] | None = None
    for summary in raw_stratum_counts:
        if not isinstance(summary, dict) or set(summary) != {
            "cu_loading",
            "protocol_family",
            "group_count",
            "sample_count",
            "split_group_counts",
            "split_sample_counts",
        }:
            raise ValueError("profile manifest split stratum counts are invalid")
        stratum = (summary.get("cu_loading"), summary.get("protocol_family"))
        if (
            stratum[0] not in {"High-Cu", "Low-Cu"}
            or stratum[1] not in {"exposure", "cycles"}
            or previous_stratum is not None
            and cast(tuple[str, str], stratum) <= previous_stratum
        ):
            raise ValueError("profile manifest split strata are invalid or unsorted")
        previous_stratum = cast(tuple[str, str], stratum)
        group_count = summary.get("group_count")
        sample_count = summary.get("sample_count")
        group_counts = summary.get("split_group_counts")
        sample_counts = summary.get("split_sample_counts")
        if (
            isinstance(group_count, bool)
            or not isinstance(group_count, int)
            or group_count < 3
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < group_count
            or not isinstance(group_counts, dict)
            or set(group_counts) != {"train", "val", "test"}
            or not isinstance(sample_counts, dict)
            or set(sample_counts) != {"train", "val", "test"}
            or any(
                isinstance(group_counts[name], bool)
                or not isinstance(group_counts[name], int)
                or group_counts[name] < 1
                for name in ("train", "val", "test")
            )
            or any(
                isinstance(sample_counts[name], bool)
                or not isinstance(sample_counts[name], int)
                or sample_counts[name] < 1
                for name in ("train", "val", "test")
            )
            or sum(cast(int, group_counts[name]) for name in group_counts)
            != group_count
            or sum(cast(int, sample_counts[name]) for name in sample_counts)
            != sample_count
        ):
            raise ValueError("profile manifest split stratum counts are inconsistent")

    raw_tracks = root["tracks"]
    if not isinstance(raw_tracks, list) or len(raw_tracks) != len(DOSE_FRACTIONS):
        raise ValueError("profile manifest must contain the three fixed dose tracks")
    tracks: list[dict[str, Any]] = []
    for expected_dose, item in zip(DOSE_FRACTIONS, raw_tracks, strict=True):
        expected_directory = f"tracks/dose-{expected_dose:.2f}"
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "dose_fraction",
                "directory",
                "benchmark_asset_sha256",
                "inference_asset_sha256",
            }
            or isinstance(item.get("dose_fraction"), bool)
            or item.get("dose_fraction") != expected_dose
            or item.get("directory") != expected_directory
            or any(
                not isinstance(item.get(field), str)
                or _SHA256.fullmatch(cast(str, item[field])) is None
                for field in ("benchmark_asset_sha256", "inference_asset_sha256")
            )
        ):
            raise ValueError("profile manifest track identity is invalid")
        tracks.append(cast(dict[str, Any], item))
    return (
        cast(dict[str, Any], dataset),
        cast(dict[str, Any], source),
        cast(dict[str, Any], split),
        tracks,
    )


def load_cu_cha_denoising_pairs(
    profile_directory: Path,
    *,
    dose_fraction: float,
    expected_profile_sha256: str,
) -> tuple[DenoisingPair, ...]:
    """Load and verify one Cu-CHA v2 track as model-independent denoising pairs."""

    if _SHA256.fullmatch(expected_profile_sha256) is None:
        raise ValueError("expected_profile_sha256 must be a lowercase SHA-256")
    try:
        profile_bytes = (profile_directory / "manifest.json").read_bytes()
    except OSError as error:
        raise ValueError("profile manifest is not readable") from error
    if hashlib.sha256(profile_bytes).hexdigest() != expected_profile_sha256:
        raise ValueError(
            "profile manifest SHA-256 does not match the external trust root"
        )
    try:
        root_value = json.loads(profile_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("profile manifest is not readable JSON") from error
    if not isinstance(root_value, dict):
        raise TypeError("profile manifest must be a JSON object")
    root = cast(dict[str, Any], root_value)
    dataset_identity, source_identity, split_identity, raw_tracks = (
        _validate_profile_manifest(root)
    )
    matching = [
        item
        for item in raw_tracks
        if isinstance(item, dict) and item.get("dose_fraction") == dose_fraction
    ]
    if len(matching) != 1:
        raise ValueError("profile must contain exactly one requested dose track")
    track = matching[0]
    directory_value = track.get("directory")
    if not isinstance(directory_value, str):
        raise TypeError("track directory is invalid")
    track_directory = profile_directory / _normalized_source_path(directory_value)
    track_manifest = _read_json_object(
        track_directory / "manifest.json", description="track manifest"
    )
    expected_corruption = {
        "algorithm": "Poisson(dose_fraction * count) / dose_fraction",
        "channels": ["I0", "I1"],
        "contract_version": DEFAULT_CORRUPTION_CONTRACT,
        "zero_count_policy": "reject_without_epsilon_correction",
    }
    expected_target = {
        "ground_truth_kind": "proxy_full_count",
        "is_physical_noiseless_ground_truth": False,
        "signal": "mu_trans_full_count",
    }
    expected_benchmark_asset = {
        "path": "benchmark.npz",
        "sha256": track.get("benchmark_asset_sha256"),
    }
    expected_inference_asset = {
        "path": "inference.npz",
        "sha256": track.get("inference_asset_sha256"),
        "contract": "hyperspectrum-local-inference-npz/v1",
    }
    if (
        set(track_manifest)
        != {
            "schema_version",
            "profile_schema_version",
            "dataset_code",
            "dose_fraction",
            "corruption",
            "target",
            "split_manifest_digest",
            "benchmark_asset",
            "inference_asset",
        }
        or track_manifest.get("schema_version")
        != "hyperspectrum-xas-denoising-track/v2"
        or track_manifest.get("profile_schema_version") != PROFILE_SCHEMA_VERSION
        or track_manifest.get("dataset_code") != DATASET_CODE
        or track_manifest.get("dose_fraction") != dose_fraction
        or track_manifest.get("corruption") != expected_corruption
        or track_manifest.get("target") != expected_target
        or track_manifest.get("split_manifest_digest")
        != split_identity["manifest_digest"]
        or track_manifest.get("benchmark_asset") != expected_benchmark_asset
        or track_manifest.get("inference_asset") != expected_inference_asset
    ):
        raise ValueError("track manifest is invalid or inconsistent with the profile")

    asset_path = track_directory / "benchmark.npz"
    asset_bytes = asset_path.read_bytes()
    if hashlib.sha256(asset_bytes).hexdigest() != track.get("benchmark_asset_sha256"):
        raise ValueError("benchmark asset SHA-256 does not match its manifest")
    inference_bytes = (track_directory / "inference.npz").read_bytes()
    if hashlib.sha256(inference_bytes).hexdigest() != track.get(
        "inference_asset_sha256"
    ):
        raise ValueError("inference asset SHA-256 does not match its manifest")
    try:
        with np.load(io.BytesIO(inference_bytes), allow_pickle=False) as loaded:
            inference_arrays = {
                name: np.array(loaded[name], copy=True) for name in loaded.files
            }
    except (OSError, ValueError) as error:
        raise ValueError("inference asset is not a readable NPZ") from error
    if set(inference_arrays) != {
        "energy",
        "energy_unit",
        "group_ids",
        "noisy",
        "sample_ids",
    }:
        raise ValueError("inference asset keys do not match the exact input contract")
    inference_energy = np.asarray(inference_arrays["energy"], dtype=np.float64)
    inference_noisy = np.asarray(inference_arrays["noisy"], dtype=np.float64)
    inference_unit = np.asarray(inference_arrays["energy_unit"])
    inference_sample_ids = np.asarray(inference_arrays["sample_ids"])
    inference_group_ids = np.asarray(inference_arrays["group_ids"])
    if (
        inference_energy.ndim != 2
        or inference_noisy.shape != inference_energy.shape
        or inference_energy.shape[0] == 0
        or inference_energy.shape[1] < 2
        or not np.isfinite(inference_energy).all()
        or not np.isfinite(inference_noisy).all()
        or not np.all(np.diff(inference_energy, axis=1) > 0)
        or inference_unit.shape != ()
        or inference_unit.dtype.kind != "U"
        or str(inference_unit) != "eV"
        or inference_sample_ids.ndim != 1
        or inference_group_ids.ndim != 1
        or inference_sample_ids.dtype.kind != "U"
        or inference_group_ids.dtype.kind != "U"
        or len(inference_sample_ids) != inference_energy.shape[0]
        or len(inference_group_ids) != inference_energy.shape[0]
        or any(not str(value).strip() for value in inference_sample_ids)
        or any(not str(value).strip() for value in inference_group_ids)
        or len(set(inference_sample_ids.tolist())) != len(inference_sample_ids)
    ):
        raise ValueError("inference asset arrays violate the strict input contract")

    split_payload = _read_json_object(
        profile_directory / "split.json", description="split manifest"
    )
    raw_entries = split_payload.get("entries")
    if not isinstance(raw_entries, list):
        raise TypeError("split manifest entries must be an array")
    try:
        split_manifest = SplitManifest(
            schema_version="hyperspectrum-split-manifest/v1",
            entries=tuple(
                SplitEntry(
                    sample_id=cast(str, entry["sample_id"]),
                    group_id=cast(str, entry["group_id"]),
                    split=cast(Any, entry["split"]),
                )
                for entry in raw_entries
                if isinstance(entry, dict)
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("split manifest is invalid") from error
    if (
        len(split_manifest.entries) != len(raw_entries)
        or split_payload.get("digest") != split_manifest.digest
        or split_identity["manifest_digest"] != split_manifest.digest
    ):
        raise ValueError("split manifest digest is inconsistent")

    with np.load(io.BytesIO(asset_bytes), allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    expected_keys = {
        "condition_segments",
        "cu_loadings",
        "energy",
        "energy_unit",
        "experiments",
        "group_ids",
        "i0_full_count",
        "i1_full_count",
        "i2_full_count",
        "noisy",
        "noisy_i0_counts",
        "noisy_i1_counts",
        "proxy_full_count",
        "protocol_families",
        "sample_ids",
        "sample_seeds",
        "source_paths",
        "source_sha256",
        "splits",
    }
    if set(arrays) != expected_keys:
        raise ValueError("benchmark asset keys do not match the v2 track contract")
    energy = np.asarray(arrays["energy"], dtype=np.float64)
    noisy = np.asarray(arrays["noisy"], dtype=np.float64)
    proxy = np.asarray(arrays["proxy_full_count"], dtype=np.float64)
    energy_unit = np.asarray(arrays["energy_unit"])
    if (
        energy.ndim != 2
        or noisy.shape != energy.shape
        or proxy.shape != energy.shape
        or energy.shape[0] == 0
        or energy.shape[1] < 2
        or not np.isfinite(energy).all()
        or not np.isfinite(noisy).all()
        or not np.isfinite(proxy).all()
        or not np.all(np.diff(energy, axis=1) > 0)
        or energy_unit.shape != ()
        or energy_unit.dtype.kind not in {"U", "S"}
        or str(energy_unit) != "eV"
    ):
        raise ValueError("benchmark signal arrays are invalid")
    count = energy.shape[0]
    i0_full = np.asarray(arrays["i0_full_count"], dtype=np.float64)
    i1_full = np.asarray(arrays["i1_full_count"], dtype=np.float64)
    i2_full = np.asarray(arrays["i2_full_count"], dtype=np.float64)
    noisy_i0 = np.asarray(arrays["noisy_i0_counts"], dtype=np.float64)
    noisy_i1 = np.asarray(arrays["noisy_i1_counts"], dtype=np.float64)
    count_arrays = (i0_full, i1_full, i2_full, noisy_i0, noisy_i1)
    seeds = np.asarray(arrays["sample_seeds"])
    if (
        any(values.shape != energy.shape for values in count_arrays)
        or any(not np.isfinite(values).all() for values in count_arrays)
        or any(not np.all(values > 0) for values in count_arrays)
        or seeds.shape != (count,)
        or seeds.dtype != np.dtype(np.uint64)
        or source_identity["file_count"] != count
    ):
        raise ValueError("benchmark count channels or source count are invalid")
    if not np.allclose(
        proxy,
        np.log(i0_full / i1_full),
        rtol=1e-6,
        atol=5e-6,
    ):
        raise ValueError("benchmark full-count proxy does not agree with log(I0/I1)")
    if not np.allclose(
        noisy,
        np.log(noisy_i0 / noisy_i1),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("benchmark noisy signal does not agree with thinned counts")

    def strings(name: str) -> list[str]:
        values = np.asarray(arrays[name])
        if (
            values.ndim != 1
            or len(values) != count
            or values.dtype.kind not in {"U", "S"}
            or any(not str(item).strip() for item in values)
        ):
            raise ValueError(f"benchmark {name} must be a one-dimensional string array")
        return [str(item) for item in values.tolist()]

    sample_ids = strings("sample_ids")
    group_ids = strings("group_ids")
    experiments = strings("experiments")
    segments = strings("condition_segments")
    cu_loadings = strings("cu_loadings")
    protocol_families = strings("protocol_families")
    source_paths = strings("source_paths")
    source_sha256 = strings("source_sha256")
    splits = strings("splits")
    if (
        len(split_manifest.entries) != count
        or len(sample_ids) != len(set(sample_ids))
        or any(_SHA256.fullmatch(value) is None for value in source_sha256)
    ):
        raise ValueError("benchmark sample identities are inconsistent")
    if (
        not np.array_equal(inference_energy, energy)
        or not np.array_equal(inference_noisy, noisy)
        or inference_sample_ids.tolist() != sample_ids
        or inference_group_ids.tolist() != group_ids
    ):
        raise ValueError("inference arrays do not exactly match the benchmark inputs")

    expected_seeds = np.asarray(
        [
            derive_cu_cha_seed(
                dataset_version=cast(str, dataset_identity["version_id"]),
                source_sha256=source_sha256[index],
                source_path=source_paths[index],
                dose_fraction=dose_fraction,
                contract_version=DEFAULT_CORRUPTION_CONTRACT,
            )
            for index in range(count)
        ],
        dtype=np.uint64,
    )
    if not np.array_equal(seeds, expected_seeds):
        raise ValueError(
            "benchmark sample seeds do not match the corruption identity chain"
        )
    expected_noisy_i0_rows: list[npt.NDArray[np.float64]] = []
    expected_noisy_i1_rows: list[npt.NDArray[np.float64]] = []
    for index, seed in enumerate(expected_seeds.tolist()):
        generator = np.random.Generator(np.random.PCG64(seed))
        expected_noisy_i0_rows.append(
            generator.poisson(dose_fraction * i0_full[index]).astype(np.float64)
            / dose_fraction
        )
        expected_noisy_i1_rows.append(
            generator.poisson(dose_fraction * i1_full[index]).astype(np.float64)
            / dose_fraction
        )
    if not np.array_equal(
        noisy_i0, np.stack(expected_noisy_i0_rows)
    ) or not np.array_equal(noisy_i1, np.stack(expected_noisy_i1_rows)):
        raise ValueError(
            "benchmark count channels do not match the deterministic Poisson draws"
        )

    parsed_identities: list[CuChaSpectrumIdentity] = []
    for index, sample_id in enumerate(sample_ids):
        parsed_identity = parse_cu_cha_identity(source_paths[index])
        if (
            parsed_identity.sample_id != sample_id
            or parsed_identity.group_id != group_ids[index]
            or parsed_identity.experiment != experiments[index]
            or parsed_identity.condition_segment != segments[index]
            or parsed_identity.cu_loading != cu_loadings[index]
            or parsed_identity.protocol_family != protocol_families[index]
        ):
            raise ValueError("benchmark path-derived identities are inconsistent")
        parsed_identities.append(parsed_identity)
    if split_identity["stratum_counts"] != _split_stratum_counts(
        tuple(parsed_identities), split_manifest
    ):
        raise ValueError(
            "profile split stratum counts do not match benchmark identities"
        )

    pairs: list[DenoisingPair] = []
    for index, sample_id in enumerate(sample_ids):
        assignment = split_manifest.assignment_for(sample_id, group_ids[index])
        if assignment.split != splits[index]:
            raise ValueError("benchmark split array and manifest disagree")
        axis = SpectrumAxis(
            name="energy",
            unit="eV",
            direction="increasing",
            values=energy[index],
        )
        valid_mask = np.ones(energy.shape[1], dtype=np.bool_)
        shared_metadata = {
            "dataset_code": DATASET_CODE,
            "experiment": experiments[index],
            "condition_segment": segments[index],
            "cu_loading": cu_loadings[index],
            "protocol_family": protocol_families[index],
            "source_path": source_paths[index],
            "split": assignment.split,
        }
        shared_provenance = {
            "source_sha256": source_sha256[index],
            "source_declaration_digest": source_identity["declaration_digest"],
            "source_content_manifest_digest": source_identity[
                "content_manifest_digest"
            ],
            "benchmark_asset_digest": track["benchmark_asset_sha256"],
        }
        noisy_sample = SpectrumSample(
            sample_id=sample_id,
            group_id=group_ids[index],
            modality="xas",
            representation="dense",
            axes=(axis,),
            signal=noisy[index],
            valid_mask=valid_mask,
            signal_unit="absorbance log(I0/I1)",
            metadata=shared_metadata,
            provenance={
                **shared_provenance,
                "corruption_contract_version": DEFAULT_CORRUPTION_CONTRACT,
                "dose_fraction": dose_fraction,
            },
        )
        target_sample = SpectrumSample(
            sample_id=sample_id,
            group_id=group_ids[index],
            modality="xas",
            representation="dense",
            axes=(axis,),
            signal=proxy[index],
            valid_mask=valid_mask,
            signal_unit="absorbance log(I0/I1)",
            metadata=shared_metadata,
            provenance={
                **shared_provenance,
                "ground_truth_kind": "proxy_full_count",
                "is_physical_noiseless_ground_truth": False,
            },
        )
        pairs.append(
            DenoisingPair(
                noisy=noisy_sample,
                clean=target_sample,
                assignment=assignment,
            )
        )
    return tuple(pairs)


def build_grouped_balanced_split(
    identities: tuple[CuChaSpectrumIdentity, ...],
) -> SplitManifest:
    """Assign complete condition segments to deterministic balanced splits."""

    if not identities:
        raise ValueError("split identities must be non-empty")
    sample_ids = tuple(item.sample_id for item in identities)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("split sample IDs must be unique")

    groups: dict[str, list[CuChaSpectrumIdentity]] = {}
    group_strata: dict[str, tuple[str, str]] = {}
    for identity in identities:
        groups.setdefault(identity.group_id, []).append(identity)
        stratum = (identity.cu_loading, identity.protocol_family)
        previous = group_strata.setdefault(identity.group_id, stratum)
        if previous != stratum:
            raise ValueError("one condition group cannot cross split strata")

    strata_groups: dict[tuple[str, str], list[str]] = {}
    for group_id, stratum in group_strata.items():
        strata_groups.setdefault(stratum, []).append(group_id)
    undercovered = [
        stratum for stratum, group_ids in strata_groups.items() if len(group_ids) < 3
    ]
    if undercovered:
        raise ValueError(
            f"split stratum {min(undercovered)} requires at least three condition groups"
        )

    policy_version = "hyperspectrum-cu-cha-grouped-balanced-split/v1"
    target_fraction: dict[SplitName, float] = {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15,
    }
    split_order: tuple[SplitName, ...] = ("train", "val", "test")
    group_assignments: dict[str, SplitName] = {}
    for stratum in sorted(strata_groups):
        ordered_groups = sorted(
            strata_groups[stratum],
            key=lambda group_id: (
                -len(groups[group_id]),
                hashlib.sha256(
                    f"{policy_version}\0{stratum[0]}\0{stratum[1]}\0{group_id}".encode()
                ).hexdigest(),
                group_id,
            ),
        )
        assigned_counts: dict[SplitName, int] = {name: 0 for name in split_order}
        for group_id in ordered_groups:
            split = min(
                split_order,
                key=lambda name: assigned_counts[name] / target_fraction[name],
            )
            group_assignments[group_id] = split
            assigned_counts[split] += len(groups[group_id])

    return SplitManifest(
        schema_version="hyperspectrum-split-manifest/v1",
        entries=tuple(
            SplitEntry(
                sample_id=identity.sample_id,
                group_id=identity.group_id,
                split=group_assignments[identity.group_id],
            )
            for identity in sorted(identities, key=lambda item: item.sample_id)
        ),
    )


def parse_cu_cha_identity(relative_path: str) -> CuChaSpectrumIdentity:
    """Parse a Cu-CHA scan path into its stable sample and group identities."""

    parsed = PurePosixPath(relative_path)
    if (
        not relative_path
        or relative_path.startswith("/")
        or "\\" in relative_path
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != relative_path
        or len(parsed.parts) < 3
    ):
        raise ValueError("source path must be normalized, relative, and nested")

    experiment = parsed.parent.name
    experiment_match = _EXPERIMENT.fullmatch(experiment)
    scan_match = _SCAN_NAME.fullmatch(parsed.name)
    if experiment_match is None or scan_match is None:
        raise ValueError("source path does not match the Cu-CHA naming contract")

    condition_segment = scan_match.group("segment")
    if not condition_segment.strip():
        raise ValueError("condition segment must be non-blank")
    return CuChaSpectrumIdentity(
        sample_id=parsed.with_suffix("").as_posix(),
        experiment=experiment,
        condition_segment=condition_segment,
        group_id=f"{experiment}/{condition_segment}",
        cu_loading=experiment_match.group("loading"),
        protocol_family=experiment_match.group("protocol"),
        scan_index=int(scan_match.group("scan")),
        temperature_c=int(scan_match.group("temperature")),
    )


def _readonly(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def parse_cu_cha_dat(path: Path) -> ParsedCuChaSpectrum:
    """Parse one Cu-CHA operando DAT scan."""

    try:
        content = path.read_bytes()
    except OSError as error:
        raise SpectrumRejected("unreadable", "DAT source is not readable") from error
    return _parse_cu_cha_dat_bytes(content)


def _parse_cu_cha_dat_bytes(content: bytes) -> ParsedCuChaSpectrum:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SpectrumRejected("encoding", "DAT source is not strict UTF-8") from error

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header_indexes = [
        index
        for index, line in enumerate(lines)
        if not line.startswith("#")
        and tuple(part.strip() for part in line.split(",")) == EXPECTED_COLUMNS
    ]
    if len(header_indexes) != 1:
        raise SpectrumRejected(
            "columns", "DAT source must contain the exact six-column header"
        )

    data_lines = [
        line for line in lines[header_indexes[0] + 1 :] if not line.startswith("#")
    ]
    if len(data_lines) < 5:
        raise SpectrumRejected(
            "rows", "DAT source must contain at least five data rows"
        )

    rows: list[list[float]] = []
    for line in data_lines:
        fields = line.split()
        if len(fields) != len(EXPECTED_COLUMNS):
            raise SpectrumRejected(
                "columns", "Each DAT data row must contain six values"
            )
        try:
            rows.append([float(field) for field in fields])
        except ValueError as error:
            raise SpectrumRejected(
                "numeric", "DAT data values must be numeric"
            ) from error

    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        raise SpectrumRejected("nonfinite", "DAT data values must all be finite")

    energy_ev = values[:, 0] * 1_000.0
    if not np.all(np.diff(energy_ev) > 0):
        raise SpectrumRejected(
            "energy_not_increasing", "Energy must be strictly increasing"
        )

    counts = values[:, 3:6]
    if not np.all(counts > 0):
        raise SpectrumRejected("nonpositive_counts", "I0, I1, and I2 must be positive")

    mu_trans = values[:, 1]
    if not np.allclose(
        mu_trans, np.log(values[:, 3] / values[:, 4]), rtol=1e-6, atol=5e-6
    ):
        raise SpectrumRejected(
            "mu_trans_mismatch", "mu_trans must agree with log(I0 / I1)"
        )

    return ParsedCuChaSpectrum(
        energy_ev=_readonly(energy_ev),
        mu_trans=_readonly(mu_trans),
        mu_ref=_readonly(values[:, 2]),
        i0=_readonly(values[:, 3]),
        i1=_readonly(values[:, 4]),
        i2=_readonly(values[:, 5]),
    )
