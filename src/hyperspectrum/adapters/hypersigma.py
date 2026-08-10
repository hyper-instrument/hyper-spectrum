"""Fail-closed adapter boundary for the official HyperSIGMA denoiser.

The upstream source and checkpoints stay external to this package.  This
module records their immutable identities and only operates on caller-mounted
files; it never downloads code, weights, or datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Literal, cast

WeightVariant = Literal["gaussian", "complex"]

HYPERSIGMA_SOURCE_COMMIT = "07e9ea24e3072fcb5c3a92a2bcb8185e43b295b9"
HYPERSIGMA_CODE_LICENSE = "Apache-2.0"
HYPERSIGMA_SOURCE_FILES: Mapping[str, str] = MappingProxyType(
    {
        "ImageDenoising/models/hypersigma/model.py": (
            "66c2165b1e04aa7df7c995f3d9ee84a332aa189572364ffb74be8a3b89711c1f"
        ),
        "ImageDenoising/models/hypersigma/Spatial.py": (
            "b9834e915333c9f7b8c48d818a0a7bf995c6c8966a1f9b1b3299e642bdb2256b"
        ),
        "ImageDenoising/models/hypersigma/Spectral.py": (
            "c2fbbdcfbf75622c7ecfd88a3ff7c42295f19684e24b6022fd65e1f04050da92"
        ),
        "ImageDenoising/models/hypersigma/Spatial_route.py": (
            "6b80337dfa94253504936584a85d62b5c81db380ba45ef52bc1a59cf0026883f"
        ),
        "ImageDenoising/models/hypersigma/Spectral_route.py": (
            "617916efbe01fff98248f7b95bcd53bc9265b14f425aed2f90d0a8ee800931b3"
        ),
    }
)


@dataclass(frozen=True, slots=True)
class WeightAssetContract:
    """Immutable identity of one externally mounted checkpoint."""

    variant: WeightVariant
    repository: str
    revision: str
    asset_id: str
    source_url: str
    filename: str
    size_bytes: int
    sha256: str
    license: str


HYPERSIGMA_WEIGHTS: Mapping[WeightVariant, WeightAssetContract] = MappingProxyType(
    {
        "gaussian": WeightAssetContract(
            variant="gaussian",
            repository="WHU-Sigma/HyperSIGMA",
            revision="e0567395fbdfddbae994695baf5fc73358a1ec3c",
            asset_id="hf-whu-sigma-hypersigma-gaussian-e0567395",
            source_url=(
                "https://huggingface.co/WHU-Sigma/HyperSIGMA/resolve/"
                "e0567395fbdfddbae994695baf5fc73358a1ec3c/"
                "Denoising_models/hypersigma_gaussian_noise_model.pth"
            ),
            filename="hypersigma_gaussian_noise_model.pth",
            size_bytes=2266408098,
            sha256=(
                "dc101cfe7d462d721eb46395b1621d82103cf4e72166d8591b5619cb2d10e806"
            ),
            license="Apache-2.0",
        ),
        "complex": WeightAssetContract(
            variant="complex",
            repository="WHU-Sigma/HyperSIGMA",
            revision="e0567395fbdfddbae994695baf5fc73358a1ec3c",
            asset_id="hf-whu-sigma-hypersigma-complex-e0567395",
            source_url=(
                "https://huggingface.co/WHU-Sigma/HyperSIGMA/resolve/"
                "e0567395fbdfddbae994695baf5fc73358a1ec3c/"
                "Denoising_models/hypersigma_complex_noise_model.pth"
            ),
            filename="hypersigma_complex_noise_model.pth",
            size_bytes=2266408098,
            sha256=(
                "8b1162aae6811af67d287b9271e74154db5448c5e2fd6df953dae5d7807bbe7e"
            ),
            license="Apache-2.0",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class VerifiedWeightIdentity:
    """Identity proven from the exact checkpoint descriptor used for loading."""

    variant: WeightVariant
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedSourceTree:
    """Expected upstream files proven byte-for-byte before dynamic import."""

    root: Path
    commit: str
    file_sha256: Mapping[str, str]


def _stream_sha256(stream: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


def verify_source_tree(
    source_root: Path,
    *,
    expected_files: Mapping[str, str] = HYPERSIGMA_SOURCE_FILES,
) -> VerifiedSourceTree:
    """Verify each pinned upstream Python file and fail on the first mismatch."""

    actual: dict[str, str] = {}
    for relative, expected_sha256 in expected_files.items():
        path = source_root / relative
        try:
            with path.open("rb") as stream:
                _, digest = _stream_sha256(stream)
        except OSError as error:
            raise ValueError(
                f"cannot read verified source {relative}: {error}"
            ) from error
        if digest != expected_sha256:
            raise ValueError(
                f"source SHA-256 mismatch for {relative}: "
                f"expected {expected_sha256}, actual {digest}"
            )
        actual[relative] = digest
    return VerifiedSourceTree(
        root=source_root.resolve(),
        commit=HYPERSIGMA_SOURCE_COMMIT,
        file_sha256=MappingProxyType(actual),
    )


def verify_weight_stream(
    stream: BinaryIO,
    contract: WeightAssetContract,
    *,
    source_path: Path,
) -> VerifiedWeightIdentity:
    """Hash one open checkpoint stream, validate it, and rewind the same stream."""

    size, digest = _stream_sha256(stream)
    if size != contract.size_bytes:
        raise ValueError(
            f"weight size mismatch: expected {contract.size_bytes}, actual {size}"
        )
    if digest != contract.sha256:
        raise ValueError(
            f"weight SHA-256 mismatch: expected {contract.sha256}, actual {digest}"
        )
    stream.seek(0)
    return VerifiedWeightIdentity(
        variant=contract.variant,
        path=source_path.resolve(),
        size_bytes=size,
        sha256=digest,
    )


def load_verified_state_dict(
    path: Path,
    contract: WeightAssetContract,
    torch_module: Any,
) -> tuple[Mapping[str, Any], VerifiedWeightIdentity]:
    """Safely load ``checkpoint['net']`` from the exact verified descriptor."""

    try:
        with path.open("rb") as stream:
            identity = verify_weight_stream(stream, contract, source_path=path)
            loaded = torch_module.load(
                stream,
                map_location="cpu",
                weights_only=True,
            )
    except OSError as error:
        raise ValueError(f"cannot read weight asset: {error}") from error
    if not isinstance(loaded, Mapping):
        raise TypeError("checkpoint must be a mapping with top-level net")
    if "net" not in loaded:
        raise ValueError("checkpoint must be a mapping with top-level net")
    state = loaded["net"]
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint net must be a state-dictionary mapping")
    return cast(Mapping[str, Any], state), identity


def _verify_weight_path(
    path: Path, contract: WeightAssetContract
) -> VerifiedWeightIdentity:
    try:
        with path.open("rb") as stream:
            return verify_weight_stream(stream, contract, source_path=path)
    except OSError as error:
        raise ValueError(f"cannot read weight asset: {error}") from error


def verification_report(
    source_root: Path | None = None,
    weight_paths: Mapping[WeightVariant, Path] | None = None,
) -> Mapping[str, object]:
    """Return static identities plus any explicitly requested local checks."""

    source: dict[str, object] = {
        "commit": HYPERSIGMA_SOURCE_COMMIT,
        "license": HYPERSIGMA_CODE_LICENSE,
        "files": dict(HYPERSIGMA_SOURCE_FILES),
        "status": "declared",
    }
    if source_root is not None:
        verified_source = verify_source_tree(source_root)
        source.update(
            {
                "root": str(verified_source.root),
                "status": "verified",
            }
        )

    supplied_weights = weight_paths or {}
    unknown = set(supplied_weights).difference(HYPERSIGMA_WEIGHTS)
    if unknown:
        raise ValueError(f"unknown HyperSIGMA weight variant: {min(unknown)}")
    weights: dict[str, object] = {}
    for variant, contract in HYPERSIGMA_WEIGHTS.items():
        details: dict[str, object] = asdict(contract)
        details["status"] = "declared"
        path = supplied_weights.get(variant)
        if path is not None:
            identity = _verify_weight_path(path, contract)
            details.update(
                {
                    "path": str(identity.path),
                    "status": "verified",
                }
            )
        weights[variant] = details

    return {
        "schema_version": "hyperspectrum-hypersigma-verification/v1",
        "source": source,
        "weights": weights,
    }


def _parse_weight_argument(value: str) -> tuple[WeightVariant, Path]:
    variant, separator, raw_path = value.partition("=")
    if separator == "" or not raw_path:
        raise argparse.ArgumentTypeError("weight must be VARIANT=/absolute/path")
    if variant not in HYPERSIGMA_WEIGHTS:
        raise argparse.ArgumentTypeError(
            f"unknown HyperSIGMA weight variant: {variant or '<empty>'}"
        )
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("weight path must be absolute")
    return cast(WeightVariant, variant), path


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify immutable HyperSIGMA source and checkpoint identities."
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        type=_parse_weight_argument,
        metavar="VARIANT=/ABSOLUTE/PATH",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run static or caller-requested local identity verification."""

    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    if not arguments.verify:
        parser.error("--verify is required")

    weight_paths: dict[WeightVariant, Path] = {}
    for variant, path in arguments.weight:
        if variant in weight_paths:
            parser.error(f"duplicate HyperSIGMA weight variant: {variant}")
        weight_paths[variant] = path
    try:
        report = verification_report(
            source_root=arguments.source_root,
            weight_paths=weight_paths,
        )
    except ValueError as error:
        parser.exit(1, f"HyperSIGMA verification failed: {error}\n")
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
