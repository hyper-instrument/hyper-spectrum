"""Strict, deterministic materialization for the Zenodo 10606662 XANES data."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from hyperspectrum.denoising.evaluation import DenoisingPair
from hyperspectrum.denoising.sample import SpectrumAxis, SpectrumSample
from hyperspectrum.denoising.split import SplitEntry, SplitManifest

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPERIMENT_PATH = re.compile(
    r"^Exp(?P<number>[2-9])_(?P<composition>(?:La|Gd)[0-9]{2})_.+\.dat$"
)
_EXPERIMENT_ONE_PATH = re.compile(r"^exp1_.+\.dat$")
_POWDER_PATH = re.compile(
    r"^Powder_(?P<composition>(?:La|Gd)[0-9]{2})_.+\.dat$"
)
_EXPECTED_EXPERIMENT_COMPOSITION = {
    "Exp2": "La05",
    "Exp3": "La05",
    "Exp4": "La08",
    "Exp5": "La02",
    "Exp6": "La00",
    "Exp7": "Gd00",
    "Exp8": "Gd05",
    "Exp9": "La08",
}
_TARGET_SEMANTICS = "pseudo-clean frozen measurement"
_ENERGY_START = 5693.0
_ENERGY_END = 5801.4
_ENERGY_ENDPOINT_TOLERANCE_EV = 0.05
_REFERENCE_GRID_JITTER_TOLERANCE_EV = 0.05


class SourceIntegrityError(ValueError):
    """The source tree cannot be bound to its declared immutable inventory."""


class SpectrumRejected(ValueError):
    """One spectrum failed a stable, fail-closed admission rule."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


class SourceDatasetIdentity(BaseModel):
    """The explicit semantic identity declared for one upstream dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    code: str
    version: str
    upstream_revision: str | None
    declared_manifest_digest: str

    @field_validator("id", "code", "version")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dataset identity fields must be non-blank")
        return value

    @field_validator("upstream_revision")
    @classmethod
    def require_explicit_revision(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("upstream_revision must be null or non-blank")
        return value

    @field_validator("declared_manifest_digest")
    @classmethod
    def validate_declared_manifest_digest(cls, value: str) -> str:
        return _require_sha256(value, name="declared_manifest_digest")

    def digest_payload(self) -> dict[str, object]:
        """Return versioned semantic identity, preserving a null revision."""

        return {
            "schema_version": "hyperspectrum-source-dataset-identity/v1",
            **self.model_dump(mode="json"),
        }


class SourceFileDeclaration(BaseModel):
    """One source-relative immutable file declaration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    size: int = Field(ge=0)
    sha256: str

    @field_validator("path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            not value
            or value.startswith("/")
            or "\\" in value
            or parsed.is_absolute()
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or parsed.as_posix() != value
        ):
            raise ValueError("source file path must be normalized and relative")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _require_sha256(value, name="source file sha256")


class SourceDeclaration(BaseModel):
    """Complete declared inventory accepted by the materializer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["hyperspectrum-xanes-source-declaration/v1"]
    dataset: SourceDatasetIdentity
    expected_file_count: int = Field(ge=1)
    files: tuple[SourceFileDeclaration, ...]

    @model_validator(mode="after")
    def require_complete_unique_inventory(self) -> Self:
        if len(self.files) != self.expected_file_count:
            raise ValueError("declared file count does not match expected_file_count")
        paths = tuple(item.path for item in self.files)
        if len(paths) != len(set(paths)):
            raise ValueError("source declaration paths must be unique")
        return self


class SourceContentFile(BaseModel):
    """One observed source file in the verified content manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    actual_size: int = Field(ge=0)
    sha256: str

    @field_validator("path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        return SourceFileDeclaration.require_safe_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _require_sha256(value, name="source content sha256")


class SourceContentManifest(BaseModel):
    """Canonical observed-byte inventory embedded in a benchmark manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["hyperspectrum-source-content-manifest/v1"]
    files: tuple[SourceContentFile, ...]

    @model_validator(mode="after")
    def require_sorted_unique_inventory(self) -> Self:
        paths = tuple(item.path for item in self.files)
        if not paths or paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("source content manifest paths must be sorted and unique")
        return self


class _ReferenceGridManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_path: str
    source_sha256: str
    point_count: Literal[135]
    energy_unit: Literal["eV"]

    @field_validator("source_path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        return SourceFileDeclaration.require_safe_relative_path(value)

    @field_validator("source_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _require_sha256(value, name="reference source sha256")


class _PreprocessingManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["hyperspectrum-xanes-preprocessing/v1"]
    signal: Literal["ketek/i0"]
    interpolation: Literal["linear"]
    endpoint_behavior: Literal[
        "nearest_measured_value_for_reference_endpoint_jitter"
    ]
    smoothing: Literal["none"]
    normalization: Literal["none"]


class _NoiseManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["hyperspectrum-xanes-binomial-thinning/v1"]
    algorithm: Literal["Binomial(round(count), dose_fraction) / dose_fraction"]
    dose_fraction: float = Field(gt=0.0, le=1.0)
    global_seed: int = Field(strict=True)
    channels: tuple[Literal["i0", "ketek"], ...]

    @model_validator(mode="after")
    def require_exact_channels_and_finite_dose(self) -> Self:
        if self.channels != ("i0", "ketek"):
            raise ValueError("noise channels must be exactly i0 and ketek")
        if not np.isfinite(self.dose_fraction):
            raise ValueError("noise dose_fraction must be finite")
        return self


class _TargetManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    semantics: Literal["pseudo-clean frozen measurement"]
    is_physical_noiseless_ground_truth: Literal[False]


class _SplitSummaryManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["hyperspectrum-split-manifest/v1"]
    digest: str
    group_key: Literal["composition"]
    policy_version: Literal["xanes-zenodo-10606662-fixed/v1"]

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _require_sha256(value, name="split manifest digest")


class _BenchmarkCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_file_count: int = Field(ge=1)
    spectrum_candidate_count: int = Field(ge=1)
    admitted_spectrum_count: int = Field(ge=1)
    rejected_spectrum_count: int = Field(ge=0)
    excluded_file_count: int = Field(ge=0)
    sidecar_file_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_consistent_counts(self) -> Self:
        if self.spectrum_candidate_count != (
            self.admitted_spectrum_count + self.rejected_spectrum_count
        ):
            raise ValueError("benchmark spectrum counts are inconsistent")
        if self.source_file_count != (
            self.spectrum_candidate_count
            + self.excluded_file_count
            + self.sidecar_file_count
        ):
            raise ValueError("benchmark source file counts are inconsistent")
        return self


class BenchmarkManifest(BaseModel):
    """Complete self-validating materializer manifest used by v3 planning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["hyperspectrum-xanes-benchmark/v1"]
    dataset: SourceDatasetIdentity
    source_dataset_digest: str
    source_content_manifest_digest: str
    benchmark_asset_digest: str
    source_content_manifest: SourceContentManifest
    reference_grid: _ReferenceGridManifest
    preprocessing: _PreprocessingManifest
    noise: _NoiseManifest
    target: _TargetManifest
    split: _SplitSummaryManifest
    counts: _BenchmarkCounts

    @field_validator(
        "source_dataset_digest",
        "source_content_manifest_digest",
        "benchmark_asset_digest",
    )
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _require_sha256(value, name="benchmark manifest digest")

    @model_validator(mode="after")
    def recompute_identity_chain(self) -> Self:
        if self.source_dataset_digest != _canonical_digest(
            self.dataset.digest_payload()
        ):
            raise ValueError("source dataset digest does not match dataset identity")
        content_payload = self.source_content_manifest.model_dump(mode="json")
        if self.source_content_manifest_digest != _canonical_digest(content_payload):
            raise ValueError(
                "source content manifest digest does not match verified inventory"
            )
        if self.counts.source_file_count != len(self.source_content_manifest.files):
            raise ValueError("source content manifest and benchmark counts differ")
        by_path = {item.path: item for item in self.source_content_manifest.files}
        reference = by_path.get(self.reference_grid.source_path)
        if reference is None or reference.sha256 != self.reference_grid.source_sha256:
            raise ValueError("reference grid source is not bound to verified content")
        if len(
            {
                self.source_dataset_digest,
                self.source_content_manifest_digest,
                self.benchmark_asset_digest,
            }
        ) != 3:
            raise ValueError("benchmark identity classes must be distinct")
        return self


@dataclass(frozen=True, slots=True)
class ParsedXanesSpectrum:
    """One admitted SPEC scan with only the count channels used downstream."""

    columns: tuple[str, ...]
    energy: NDArray[np.float64]
    i0: NDArray[np.float64]
    ketek: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in ("energy", "i0", "ketek"):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class _SpectrumIdentity:
    sample_id: str
    group_id: str
    experiment: str
    source_kind: Literal["experiment", "powder"]


@dataclass(frozen=True, slots=True)
class _VerifiedFile:
    path: str
    declared_size: int
    actual_size: int
    sha256: str
    content: bytes

    @property
    def size_matches(self) -> bool:
        return self.declared_size == self.actual_size


@dataclass(frozen=True, slots=True)
class _AdmittedSpectrum:
    source: _VerifiedFile
    identity: _SpectrumIdentity
    spectrum: ParsedXanesSpectrum
    split: Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class BenchmarkAssetIdentity:
    """The three independent identities required by v3 execution."""

    dataset_code: str
    dataset_version: str
    declared_manifest_digest: str
    source_dataset_digest: str
    source_content_manifest_digest: str
    benchmark_asset_digest: str

    def __post_init__(self) -> None:
        if not self.dataset_code.strip() or not self.dataset_version.strip():
            raise ValueError("benchmark dataset identity must be non-blank")
        for name in (
            "declared_manifest_digest",
            "source_dataset_digest",
            "source_content_manifest_digest",
            "benchmark_asset_digest",
        ):
            _require_sha256(cast(str, getattr(self, name)), name=name)


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """Paths and immutable identities of one completed benchmark bundle."""

    output_directory: Path
    asset_path: Path
    manifest_path: Path
    split_path: Path
    qc_path: Path
    source_dataset_digest: str
    source_content_manifest_digest: str
    benchmark_asset_digest: str
    split_manifest_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "hyperspectrum-xanes-materialization-result/v1",
            "output_directory": str(self.output_directory),
            "asset_path": str(self.asset_path),
            "manifest_path": str(self.manifest_path),
            "split_path": str(self.split_path),
            "qc_path": str(self.qc_path),
            "source_dataset_digest": self.source_dataset_digest,
            "source_content_manifest_digest": self.source_content_manifest_digest,
            "benchmark_asset_digest": self.benchmark_asset_digest,
            "split_manifest_digest": self.split_manifest_digest,
        }


def _parse_xanes_spec_bytes(content: bytes) -> ParsedXanesSpectrum:
    try:
        text = content.decode("utf-8")
    except UnicodeError as error:
        raise SpectrumRejected("unreadable", "SPEC file is not readable UTF-8") from error
    lines = text.splitlines()
    scan_indices = [index for index, line in enumerate(lines) if line.startswith("#S")]
    if len(scan_indices) != 1:
        raise SpectrumRejected("scan_count", "SPEC file must contain exactly one #S")
    label_indices = [index for index, line in enumerate(lines) if line.startswith("#L")]
    if len(label_indices) != 1 or label_indices[0] <= scan_indices[0]:
        raise SpectrumRejected("label_count", "SPEC scan must contain exactly one #L")
    label_index = label_indices[0]
    columns = tuple(lines[label_index][2:].split())
    lowered = tuple(column.casefold() for column in columns)
    if len(columns) != 16:
        raise SpectrumRejected("column_count", "SPEC #L must declare 16 columns")
    if len(set(lowered)) != 16:
        raise SpectrumRejected("duplicate_columns", "SPEC #L columns must be unique")
    for required in ("energy", "i0", "ketek"):
        if lowered.count(required) != 1:
            raise SpectrumRejected(
                "required_column", f"SPEC #L must uniquely declare {required}"
            )
    declared_counts = []
    for line in lines[scan_indices[0] + 1 : label_index]:
        if line.startswith("#N"):
            fields = line[2:].split()
            if len(fields) != 1:
                raise SpectrumRejected("column_count", "SPEC #N is malformed")
            try:
                declared_counts.append(int(fields[0]))
            except ValueError as error:
                raise SpectrumRejected("column_count", "SPEC #N is malformed") from error
    if declared_counts and declared_counts != [16]:
        raise SpectrumRejected("column_count", "SPEC #N must declare 16 columns")

    rows: list[list[float]] = []
    for line in lines[label_index + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 16:
            raise SpectrumRejected("column_count", "SPEC data row must have 16 columns")
        try:
            rows.append([float(item) for item in fields])
        except ValueError as error:
            raise SpectrumRejected("numeric_data", "SPEC data must be numeric") from error
    if len(rows) != 135:
        raise SpectrumRejected("row_count", "SPEC scan must contain exactly 135 rows")
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (135, 16) or not np.isfinite(values).all():
        raise SpectrumRejected("nonfinite", "SPEC scan must be a finite 135x16 array")
    energy = values[:, lowered.index("energy")]
    i0 = values[:, lowered.index("i0")]
    ketek = values[:, lowered.index("ketek")]
    if not np.all(np.diff(energy) > 0.0):
        raise SpectrumRejected(
            "energy_not_increasing", "SPEC energy must be strictly increasing"
        )
    if (
        abs(float(energy[0]) - _ENERGY_START) > _ENERGY_ENDPOINT_TOLERANCE_EV
        or abs(float(energy[-1]) - _ENERGY_END)
        > _ENERGY_ENDPOINT_TOLERANCE_EV
    ):
        raise SpectrumRejected(
            "energy_coverage", "SPEC energy must span approximately 5693.0--5801.4 eV"
        )
    if not np.all(i0 > 0.0):
        raise SpectrumRejected("i0_nonpositive", "SPEC i0 must be positive")
    if not np.all(ketek >= 0.0):
        raise SpectrumRejected("ketek_negative", "SPEC ketek must be non-negative")
    return ParsedXanesSpectrum(
        columns=columns,
        energy=energy,
        i0=i0,
        ketek=ketek,
    )


def parse_xanes_spec(path: Path) -> ParsedXanesSpectrum:
    """Parse exactly one strict 135x16 XANES SPEC scan."""

    try:
        content = path.read_bytes()
    except OSError as error:
        raise SpectrumRejected("unreadable", "SPEC file is not readable") from error
    return _parse_xanes_spec_bytes(content)


def _load_source_declaration(path: Path) -> SourceDeclaration:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return SourceDeclaration.model_validate(loaded)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise SourceIntegrityError("source declaration is invalid") from error


def _verify_source_tree(
    root: Path, declaration: SourceDeclaration
) -> tuple[_VerifiedFile, ...]:
    if not root.is_dir():
        raise SourceIntegrityError("source root must be an existing directory")
    actual_paths: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SourceIntegrityError("source tree must not contain symbolic links")
        if path.is_file():
            actual_paths.append(path.relative_to(root).as_posix())
    declared_by_path = {item.path: item for item in declaration.files}
    if set(actual_paths) != set(declared_by_path):
        raise SourceIntegrityError(
            "source inventory differs from the complete declared file inventory"
        )
    verified: list[_VerifiedFile] = []
    for relative_path in sorted(actual_paths):
        declared = declared_by_path[relative_path]
        path = root / PurePosixPath(relative_path)
        try:
            content = path.read_bytes()
        except OSError as error:
            raise SourceIntegrityError(
                f"source file is not readable: {relative_path}"
            ) from error
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if actual_sha256 != declared.sha256:
            raise SourceIntegrityError(
                f"source file SHA-256 mismatch: {relative_path}"
            )
        verified.append(
            _VerifiedFile(
                path=relative_path,
                declared_size=declared.size,
                actual_size=len(content),
                sha256=actual_sha256,
                content=content,
            )
        )
    if len(verified) != declaration.expected_file_count:
        raise SourceIntegrityError("verified file count differs from declaration")
    return tuple(verified)


def _spectrum_identity(relative_path: str) -> _SpectrumIdentity:
    filename = PurePosixPath(relative_path).name
    experiment_match = _EXPERIMENT_PATH.fullmatch(filename)
    if experiment_match is not None:
        experiment = f"Exp{experiment_match.group('number')}"
        composition = experiment_match.group("composition")
        if _EXPECTED_EXPERIMENT_COMPOSITION[experiment] != composition:
            raise SpectrumRejected(
                "path_identity", "experiment and composition path identity disagree"
            )
        return _SpectrumIdentity(
            sample_id=relative_path.removesuffix(".dat"),
            group_id=composition,
            experiment=experiment,
            source_kind="experiment",
        )
    if _EXPERIMENT_ONE_PATH.fullmatch(filename) is not None:
        return _SpectrumIdentity(
            sample_id=relative_path.removesuffix(".dat"),
            group_id="La00",
            experiment="Exp1",
            source_kind="experiment",
        )
    powder_match = _POWDER_PATH.fullmatch(filename)
    if powder_match is not None:
        return _SpectrumIdentity(
            sample_id=relative_path.removesuffix(".dat"),
            group_id=powder_match.group("composition"),
            experiment=f"Powder_{powder_match.group('composition')}",
            source_kind="powder",
        )
    raise SpectrumRejected(
        "path_identity", "spectrum path does not encode a supported sample identity"
    )


def _fixed_split(identity: _SpectrumIdentity) -> Literal["train", "val", "test"]:
    if identity.group_id == "La08":
        if identity.experiment not in {"Exp4", "Exp9", "Powder_La08"}:
            raise SpectrumRejected("split_identity", "La08 path is outside fixed test groups")
        return "test"
    if identity.group_id == "Gd05":
        if identity.experiment not in {"Exp8", "Powder_Gd05"}:
            raise SpectrumRejected("split_identity", "Gd05 path is outside fixed val groups")
        return "val"
    return "train"


def _base_qc_entry(source: _VerifiedFile) -> dict[str, object]:
    return {
        "actual_size": source.actual_size,
        "declared_size": source.declared_size,
        "path": source.path,
        "sha256": source.sha256,
        "size_matches": source.size_matches,
    }


def _content_manifest(files: Sequence[_VerifiedFile]) -> dict[str, object]:
    return {
        "schema_version": "hyperspectrum-source-content-manifest/v1",
        "files": [
            {
                "path": item.path,
                "actual_size": item.actual_size,
                "sha256": item.sha256,
            }
            for item in files
        ],
    }


def _sample_seed(
    *,
    global_seed: int,
    sample_id: str,
    source_sha256: str,
    dose_fraction: float,
) -> int:
    payload = {
        "schema_version": "hyperspectrum-xanes-binomial-thinning-seed/v1",
        "global_seed": global_seed,
        "sample_id": sample_id,
        "source_sha256": source_sha256,
        "dose_fraction": dose_fraction,
    }
    return int.from_bytes(hashlib.sha256(_canonical_json_bytes(payload)).digest()[:8], "big")


def _thin_counts(
    values: NDArray[np.float64], *, dose_fraction: float, generator: np.random.Generator
) -> NDArray[np.float64]:
    maximum = float(np.max(values))
    if maximum > float(np.iinfo(np.int64).max):
        raise SourceIntegrityError("source counts exceed supported integer range")
    rounded = np.rint(values).astype(np.int64)
    return generator.binomial(rounded, dose_fraction).astype(np.float64) / dose_fraction


def _canonical_npz(arrays: Mapping[str, NDArray[np.generic]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(
                array_buffer, np.asanyarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_buffer.getvalue())
    return output.getvalue()


def _json_file_bytes(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def materialize_xanes_spec(
    *,
    source_root: Path,
    source_declaration_file: Path,
    output_directory: Path,
    dose_fraction: float = 0.25,
    global_seed: int = 0,
) -> MaterializationResult:
    """Verify, parse, split, noise, and freeze one canonical benchmark bundle."""

    if not np.isfinite(dose_fraction) or not 0.0 < dose_fraction <= 1.0:
        raise ValueError("dose_fraction must be finite and in (0, 1]")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")
    declaration = _load_source_declaration(source_declaration_file)
    verified_files = _verify_source_tree(source_root, declaration)
    source_dataset_payload = declaration.dataset.digest_payload()
    source_dataset_digest = _canonical_digest(source_dataset_payload)
    content_manifest = _content_manifest(verified_files)
    source_content_manifest_digest = _canonical_digest(content_manifest)

    qc_entries: list[dict[str, object]] = []
    admitted: list[_AdmittedSpectrum] = []
    for source in verified_files:
        entry = _base_qc_entry(source)
        suffix = PurePosixPath(source.path).suffix.casefold()
        if suffix == ".txt":
            entry.update(status="sidecar", reason_code="conditions_sidecar")
            qc_entries.append(entry)
            continue
        if suffix == "":
            entry.update(status="excluded", reason_code="extensionless_motor_scan")
            qc_entries.append(entry)
            continue
        if suffix != ".dat":
            entry.update(status="excluded", reason_code="unsupported_extension")
            qc_entries.append(entry)
            continue
        try:
            identity = _spectrum_identity(source.path)
            spectrum = _parse_xanes_spec_bytes(source.content)
            split = _fixed_split(identity)
        except SpectrumRejected as error:
            entry.update(status="rejected", reason_code=error.code)
            qc_entries.append(entry)
            continue
        entry.update(status="admitted", reason_code="admitted_spectrum")
        qc_entries.append(entry)
        admitted.append(
            _AdmittedSpectrum(
                source=source,
                identity=identity,
                spectrum=spectrum,
                split=split,
            )
        )
    admitted.sort(key=lambda item: item.source.path)
    if not admitted:
        raise SourceIntegrityError("no XANES spectrum passed strict QC")
    train_samples = [item for item in admitted if item.split == "train"]
    if not train_samples:
        raise SourceIntegrityError("no admitted train spectrum can define the grid")
    reference = train_samples[0]
    reference_energy = np.array(reference.spectrum.energy, copy=True)
    compatible: list[_AdmittedSpectrum] = []
    qc_by_path = {cast(str, entry["path"]): entry for entry in qc_entries}
    for item in admitted:
        lower_overhang = max(
            0.0, float(item.spectrum.energy[0] - reference_energy[0])
        )
        upper_overhang = max(
            0.0, float(reference_energy[-1] - item.spectrum.energy[-1])
        )
        if (
            lower_overhang > _REFERENCE_GRID_JITTER_TOLERANCE_EV
            or upper_overhang > _REFERENCE_GRID_JITTER_TOLERANCE_EV
        ):
            qc_by_path[item.source.path].update(
                status="rejected", reason_code="reference_grid_coverage"
            )
            continue
        compatible.append(item)
    admitted = compatible
    sample_ids = tuple(item.identity.sample_id for item in admitted)
    if len(sample_ids) != len(set(sample_ids)):
        raise SourceIntegrityError("admitted spectrum sample IDs are not unique")

    split_manifest = SplitManifest(
        schema_version="hyperspectrum-split-manifest/v1",
        entries=tuple(
            SplitEntry(
                sample_id=item.identity.sample_id,
                group_id=item.identity.group_id,
                split=item.split,
            )
            for item in admitted
        ),
    )
    pseudo_clean_rows: list[NDArray[np.float64]] = []
    noisy_rows: list[NDArray[np.float64]] = []
    noisy_i0_rows: list[NDArray[np.float64]] = []
    noisy_ketek_rows: list[NDArray[np.float64]] = []
    sample_seeds: list[int] = []
    for item in admitted:
        seed = _sample_seed(
            global_seed=global_seed,
            sample_id=item.identity.sample_id,
            source_sha256=item.source.sha256,
            dose_fraction=dose_fraction,
        )
        generator = np.random.default_rng(seed)
        noisy_i0 = _thin_counts(
            item.spectrum.i0, dose_fraction=dose_fraction, generator=generator
        )
        noisy_ketek = _thin_counts(
            item.spectrum.ketek, dose_fraction=dose_fraction, generator=generator
        )
        if np.any(noisy_i0 <= 0.0):
            raise SourceIntegrityError(
                f"Binomial thinning produced non-positive i0: {item.source.path}"
            )
        pseudo_clean = item.spectrum.ketek / item.spectrum.i0
        noisy = noisy_ketek / noisy_i0
        pseudo_clean_rows.append(
            np.interp(reference_energy, item.spectrum.energy, pseudo_clean)
        )
        noisy_rows.append(np.interp(reference_energy, item.spectrum.energy, noisy))
        noisy_i0_rows.append(noisy_i0)
        noisy_ketek_rows.append(noisy_ketek)
        sample_seeds.append(seed)

    arrays: dict[str, NDArray[np.generic]] = {
        "energy": reference_energy,
        "energy_unit": np.asarray("eV", dtype=np.str_),
        "experiments": np.asarray(
            [item.identity.experiment for item in admitted], dtype=np.str_
        ),
        "group_ids": np.asarray(
            [item.identity.group_id for item in admitted], dtype=np.str_
        ),
        "i0_counts": np.stack([item.spectrum.i0 for item in admitted]),
        "ketek_counts": np.stack([item.spectrum.ketek for item in admitted]),
        "noisy": np.stack(noisy_rows),
        "noisy_i0_counts": np.stack(noisy_i0_rows),
        "noisy_ketek_counts": np.stack(noisy_ketek_rows),
        "pseudo_clean": np.stack(pseudo_clean_rows),
        "sample_ids": np.asarray(sample_ids, dtype=np.str_),
        "sample_seeds": np.asarray(sample_seeds, dtype=np.uint64),
        "source_energy": np.stack([item.spectrum.energy for item in admitted]),
        "source_paths": np.asarray(
            [item.source.path for item in admitted], dtype=np.str_
        ),
        "source_sha256": np.asarray(
            [item.source.sha256 for item in admitted], dtype=np.str_
        ),
        "splits": np.asarray([item.split for item in admitted], dtype=np.str_),
    }
    asset_bytes = _canonical_npz(arrays)
    benchmark_asset_digest = hashlib.sha256(asset_bytes).hexdigest()
    statuses = [cast(str, item["status"]) for item in qc_entries]
    manifest = {
        "schema_version": "hyperspectrum-xanes-benchmark/v1",
        "dataset": declaration.dataset.model_dump(mode="json"),
        "source_dataset_digest": source_dataset_digest,
        "source_content_manifest_digest": source_content_manifest_digest,
        "benchmark_asset_digest": benchmark_asset_digest,
        "source_content_manifest": content_manifest,
        "reference_grid": {
            "source_path": reference.source.path,
            "source_sha256": reference.source.sha256,
            "point_count": len(reference_energy),
            "energy_unit": "eV",
        },
        "preprocessing": {
            "schema_version": "hyperspectrum-xanes-preprocessing/v1",
            "signal": "ketek/i0",
            "interpolation": "linear",
            "endpoint_behavior": "nearest_measured_value_for_reference_endpoint_jitter",
            "smoothing": "none",
            "normalization": "none",
        },
        "noise": {
            "schema_version": "hyperspectrum-xanes-binomial-thinning/v1",
            "algorithm": "Binomial(round(count), dose_fraction) / dose_fraction",
            "dose_fraction": dose_fraction,
            "global_seed": global_seed,
            "channels": ["i0", "ketek"],
        },
        "target": {
            "semantics": _TARGET_SEMANTICS,
            "is_physical_noiseless_ground_truth": False,
        },
        "split": {
            "schema_version": split_manifest.schema_version,
            "digest": split_manifest.digest,
            "group_key": "composition",
            "policy_version": "xanes-zenodo-10606662-fixed/v1",
        },
        "counts": {
            "source_file_count": len(verified_files),
            "spectrum_candidate_count": statuses.count("admitted")
            + statuses.count("rejected"),
            "admitted_spectrum_count": statuses.count("admitted"),
            "rejected_spectrum_count": statuses.count("rejected"),
            "excluded_file_count": statuses.count("excluded"),
            "sidecar_file_count": statuses.count("sidecar"),
        },
    }
    split_payload = {
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
    qc_payload = {
        "schema_version": "hyperspectrum-xanes-qc/v1",
        "source_file_count": len(verified_files),
        "size_metadata_drift_count": sum(
            not item.size_matches for item in verified_files
        ),
        "files": qc_entries,
    }
    if output_directory.exists():
        raise FileExistsError(f"output directory already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    asset_path = output_directory / "benchmark.npz"
    manifest_path = output_directory / "manifest.json"
    split_path = output_directory / "split.json"
    qc_path = output_directory / "qc.json"
    asset_path.write_bytes(asset_bytes)
    manifest_path.write_bytes(_json_file_bytes(manifest))
    split_path.write_bytes(_json_file_bytes(split_payload))
    qc_path.write_bytes(_json_file_bytes(qc_payload))
    return MaterializationResult(
        output_directory=output_directory,
        asset_path=asset_path,
        manifest_path=manifest_path,
        split_path=split_path,
        qc_path=qc_path,
        source_dataset_digest=source_dataset_digest,
        source_content_manifest_digest=source_content_manifest_digest,
        benchmark_asset_digest=benchmark_asset_digest,
        split_manifest_digest=split_manifest.digest,
    )


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON") from error
    if not isinstance(loaded, dict):
        raise TypeError(f"{description} must be a JSON object")
    return cast(dict[str, Any], loaded)


def _validate_benchmark_manifest(raw_manifest: object) -> BenchmarkManifest:
    try:
        return BenchmarkManifest.model_validate(raw_manifest)
    except (TypeError, ValueError) as error:
        raise ValueError(f"benchmark manifest is invalid: {error}") from error


def _benchmark_asset_identity(manifest: BenchmarkManifest) -> BenchmarkAssetIdentity:
    return BenchmarkAssetIdentity(
        dataset_code=manifest.dataset.code,
        dataset_version=manifest.dataset.version,
        declared_manifest_digest=manifest.dataset.declared_manifest_digest,
        source_dataset_digest=manifest.source_dataset_digest,
        source_content_manifest_digest=manifest.source_content_manifest_digest,
        benchmark_asset_digest=manifest.benchmark_asset_digest,
    )


def load_benchmark_asset_identity(manifest_path: Path) -> BenchmarkAssetIdentity:
    """Load only the validated explicit digest identity needed by v3 planning."""

    raw_manifest = _load_json_object(manifest_path, description="benchmark manifest")
    return _benchmark_asset_identity(_validate_benchmark_manifest(raw_manifest))


def _string_array(
    bundle: Mapping[str, NDArray[np.generic]], key: str, count: int
) -> list[str]:
    values = np.asarray(bundle[key])
    if values.ndim != 1 or len(values) != count or values.dtype.kind not in {"U", "S"}:
        raise ValueError(f"benchmark {key} must be a one-dimensional string array")
    return [str(item) for item in values.tolist()]


def load_denoising_pairs(output_directory: Path) -> tuple[DenoisingPair, ...]:
    """Load a verified materialization as strict reusable denoising pairs."""

    manifest_path = output_directory / "manifest.json"
    asset_path = output_directory / "benchmark.npz"
    split_path = output_directory / "split.json"
    raw_manifest = _load_json_object(manifest_path, description="benchmark manifest")
    validated_manifest = _validate_benchmark_manifest(raw_manifest)
    identity = _benchmark_asset_identity(validated_manifest)
    manifest = validated_manifest.model_dump(mode="json")
    try:
        asset_bytes = asset_path.read_bytes()
    except OSError as error:
        raise ValueError("benchmark asset is not readable") from error
    asset_digest = hashlib.sha256(asset_bytes).hexdigest()
    if asset_digest != identity.benchmark_asset_digest:
        raise ValueError("benchmark asset SHA-256 does not match its manifest")
    split_payload = _load_json_object(split_path, description="split manifest")
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
        or cast(dict[str, Any], manifest.get("split", {})).get("digest")
        != split_manifest.digest
    ):
        raise ValueError("split manifest digest is inconsistent")

    with np.load(io.BytesIO(asset_bytes), allow_pickle=False) as loaded:
        arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    energy = np.asarray(arrays["energy"], dtype=np.float64)
    noisy = np.asarray(arrays["noisy"], dtype=np.float64)
    pseudo_clean = np.asarray(arrays["pseudo_clean"], dtype=np.float64)
    if energy.shape != (135,) or noisy.ndim != 2 or pseudo_clean.shape != noisy.shape:
        raise ValueError("benchmark signal arrays have invalid shapes")
    count = noisy.shape[0]
    if noisy.shape[1] != len(energy) or not np.isfinite(noisy).all() or not np.isfinite(
        pseudo_clean
    ).all():
        raise ValueError("benchmark signal arrays must be finite on the reference grid")
    sample_ids = _string_array(arrays, "sample_ids", count)
    group_ids = _string_array(arrays, "group_ids", count)
    experiments = _string_array(arrays, "experiments", count)
    source_paths = _string_array(arrays, "source_paths", count)
    source_sha256 = _string_array(arrays, "source_sha256", count)
    splits = _string_array(arrays, "splits", count)
    if len(split_manifest.entries) != count:
        raise ValueError("split manifest and benchmark sample counts differ")
    axis = SpectrumAxis(
        name="energy", unit="eV", direction="increasing", values=energy
    )
    valid_mask = np.ones(energy.shape, dtype=np.bool_)
    pairs: list[DenoisingPair] = []
    for index, sample_id in enumerate(sample_ids):
        assignment = split_manifest.assignment_for(sample_id, group_ids[index])
        if assignment.split != splits[index]:
            raise ValueError("benchmark split array and manifest disagree")
        shared_metadata = {
            "experiment": experiments[index],
            "composition": group_ids[index],
            "source_path": source_paths[index],
            "split": assignment.split,
        }
        shared_provenance = {
            "source_sha256": source_sha256[index],
            "source_dataset_digest": identity.source_dataset_digest,
            "source_content_manifest_digest": identity.source_content_manifest_digest,
            "benchmark_asset_digest": identity.benchmark_asset_digest,
        }
        noisy_sample = SpectrumSample(
            sample_id=sample_id,
            group_id=group_ids[index],
            modality="xanes",
            representation="dense",
            axes=(axis,),
            signal=noisy[index],
            valid_mask=valid_mask,
            signal_unit="ketek/i0 ratio",
            metadata=shared_metadata,
            provenance={
                **shared_provenance,
                "noise_schema_version": "hyperspectrum-xanes-binomial-thinning/v1",
                "dose_fraction": cast(dict[str, Any], manifest["noise"])[
                    "dose_fraction"
                ],
            },
        )
        target_sample = SpectrumSample(
            sample_id=sample_id,
            group_id=group_ids[index],
            modality="xanes",
            representation="dense",
            axes=(axis,),
            signal=pseudo_clean[index],
            valid_mask=valid_mask,
            signal_unit="ketek/i0 ratio",
            metadata=shared_metadata,
            provenance={
                **shared_provenance,
                "target_semantics": _TARGET_SEMANTICS,
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
