"""Fail-closed adapter boundary for the official HyperSIGMA denoiser.

The upstream source and checkpoints stay external to this package.  This
module records their immutable identities and only operates on caller-mounted
files; it never downloads code, weights, or datasets.

The pinned ``ImageDenoising/models/hypersigma/model.py`` configures a spatial
encoder with patch 2, embedding 768, depth 12, 12 heads, outputs 3/5/7/11,
interval 3, and 8 sampling points; its spectral encoder uses 100 tokens,
embedding 768, depth 12, 12 heads, and output 11.  The spatial adapter uses
patch 1, embedding 768, depth 12, and output 3.  The spectral adapter declares
100 tokens, embedding 128, depth 12, and output 3; pinned
``Spectral_route.py`` actually executes its four constructed blocks.  Final
3x3 convolutions reconstruct 382 to 191 channels and then 191 to 191.  Class
implementations are pinned in ``Spatial.py``, ``Spectral.py``,
``Spatial_route.py``, and ``Spectral_route.py`` by the hashes below.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import sys
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import CodeType, MappingProxyType, ModuleType
from typing import Any, BinaryIO, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from hyperspectrum.denoising.model import (
    CanonicalDenoisingInput,
    CanonicalDenoisingOutput,
    ModelCapabilities,
)
from hyperspectrum.plugins.hsi.arrays import resolve_hsi_layout

WeightVariant = Literal["gaussian", "complex"]
_MODEL_LOCK = threading.Lock()
_UPSTREAM_STEMS = ("Spatial", "Spectral", "Spatial_route", "Spectral_route", "model")


class _PredictRuntime(Protocol):
    def predict(self, values: NDArray[np.float64]) -> NDArray[np.float64]: ...

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


@dataclass(frozen=True, slots=True)
class HyperSIGMARuntimeIdentity:
    """Exact source, checkpoint, and device used by one runtime instance."""

    source_commit: str
    source_sha256: Mapping[str, str]
    weight: VerifiedWeightIdentity
    device: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_sha256", MappingProxyType(dict(self.source_sha256))
        )


class _VerifiedSourceLoader(importlib.machinery.SourceFileLoader):
    """Compile the verified source bytes without reading or writing ``pyc`` files."""

    def __init__(
        self,
        fullname: str,
        path: str,
        *,
        relative_path: str,
        expected_sha256: str,
    ) -> None:
        super().__init__(fullname, path)
        self._relative_path = relative_path
        self._expected_sha256 = expected_sha256

    def get_code(self, fullname: str) -> CodeType:
        source_path = self.get_filename(fullname)
        source_bytes = self.get_data(source_path)
        actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if actual_sha256 != self._expected_sha256:
            raise ValueError(
                f"source SHA-256 mismatch at execution for {self._relative_path}: "
                f"expected {self._expected_sha256}, actual {actual_sha256}"
            )
        return self.source_to_code(source_bytes, source_path)


def _load_verified_modules(
    verified: VerifiedSourceTree,
) -> tuple[dict[str, ModuleType], tuple[str, ...]]:
    model_dir = verified.root / "ImageDenoising/models/hypersigma"
    package_name = f"_hypersigma_verified_{uuid.uuid4().hex}"
    package = ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(model_dir)]
    loaded_names = [package_name]
    sys.modules[package_name] = package
    modules: dict[str, ModuleType] = {}
    try:
        for stem in _UPSTREAM_STEMS:
            qualified = f"{package_name}.{stem}"
            relative_path = f"ImageDenoising/models/hypersigma/{stem}.py"
            source_path = verified.root / relative_path
            spec = importlib.util.spec_from_file_location(
                qualified,
                source_path,
                loader=_VerifiedSourceLoader(
                    qualified,
                    str(source_path),
                    relative_path=relative_path,
                    expected_sha256=verified.file_sha256[relative_path],
                ),
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for verified {stem}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            loaded_names.append(qualified)
            spec.loader.exec_module(module)
            modules[stem] = module
    except BaseException:
        for name in reversed(loaded_names):
            sys.modules.pop(name, None)
        raise
    return modules, tuple(loaded_names)


def _unload_verified_modules(names: Sequence[str]) -> None:
    for name in reversed(tuple(names)):
        sys.modules.pop(name, None)


def _construct_official_model(modules: Mapping[str, ModuleType]) -> Any:
    spatial_class = modules["Spatial"].SpatialVisionTransformer
    spectral_class = modules["Spectral"].SpectralVisionTransformer
    spatial_init = spatial_class.init_weights
    spectral_init = spectral_class.init_weights

    def skip_private_pretrain(_self: Any, *_args: Any, **_kwargs: Any) -> None:
        return None

    spatial_class.init_weights = skip_private_pretrain
    spectral_class.init_weights = skip_private_pretrain
    try:
        return modules["model"].spat_vit_b_rvsa()
    finally:
        spatial_class.init_weights = spatial_init
        spectral_class.init_weights = spectral_init


def _build_official_model(verified: VerifiedSourceTree) -> Any:
    with _MODEL_LOCK:
        modules, loaded_names = _load_verified_modules(verified)
        try:
            return _construct_official_model(modules)
        finally:
            _unload_verified_modules(loaded_names)


class HyperSIGMADenoiseAdapter:
    """Canonical HSI adapter for one pinned official HyperSIGMA checkpoint."""

    capabilities = ModelCapabilities(
        schema_version="hyperspectrum-denoising-capabilities/v1",
        axis_ranks=(3,),
        representations=("dense",),
        channel_counts=(1,),
        required_normalization="per_spectrum_range",
        native_unit_recovery=True,
    )

    def __init__(
        self,
        source_root: Path,
        weight_path: Path,
        *,
        variant: WeightVariant,
        device: str = "cpu",
        _runtime: _PredictRuntime | None = None,
    ) -> None:
        if variant not in HYPERSIGMA_WEIGHTS:
            raise ValueError(f"unknown HyperSIGMA weight variant: {variant}")
        self._runtime = (
            _runtime
            if _runtime is not None
            else _TorchRuntime(
                source_root,
                weight_path,
                contract=HYPERSIGMA_WEIGHTS[variant],
                device=device,
            )
        )
        self.source_root = source_root
        self.weight_path = weight_path
        self.variant = variant
        self.requested_device = device

    def predict(
        self, model_input: CanonicalDenoisingInput
    ) -> CanonicalDenoisingOutput:
        """Denoise one explicitly declared, normalized 191×64×64 HSI cube."""

        if model_input.modality != "hyperspectral":
            raise ValueError("HyperSIGMA requires hyperspectral modality")
        if model_input.representation != "dense" or np.iscomplexobj(
            model_input.signal
        ):
            raise ValueError("HyperSIGMA requires real dense input")
        if model_input.channel_labels:
            raise ValueError("HyperSIGMA requires one implicit channel")
        if model_input.normalization_method != "per_spectrum_range":
            raise ValueError("HyperSIGMA requires per_spectrum_range normalization")
        if not np.all(model_input.valid_mask):
            raise ValueError("HyperSIGMA requires a fully valid cube")
        if not np.isfinite(model_input.signal).all():
            raise ValueError("HyperSIGMA requires finite input")
        layout = resolve_hsi_layout(
            model_input.axis_names,
            model_input.axis_units,
            tuple(len(values) for values in model_input.axis_values),
            expected_band_count=191,
            expected_spatial_shape=(64, 64),
        )
        model_values = np.asarray(
            layout.to_model_layout(model_input.signal), dtype=np.float64
        )
        prediction = self._runtime.predict(model_values)
        if not np.isfinite(prediction).all():
            raise ValueError("HyperSIGMA returned non-finite output")
        restored = layout.from_model_layout(prediction)
        return CanonicalDenoisingOutput(
            sample_id=model_input.sample_id,
            signal=restored,
            valid_mask=model_input.valid_mask,
            normalization_state_digest=model_input.normalization_state_digest,
        )


def denoise_cubes(
    model_inputs: Sequence[CanonicalDenoisingInput],
    *,
    source_root: Path,
    weight_path: Path,
    variant: WeightVariant,
    device: str = "cpu",
) -> tuple[CanonicalDenoisingOutput, ...]:
    """Construct one runtime and denoise all cubes in caller order."""

    adapter = HyperSIGMADenoiseAdapter(
        source_root,
        weight_path,
        variant=variant,
        device=device,
    )
    return tuple(adapter.predict(model_input) for model_input in model_inputs)


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
            numpy_multiarray = importlib.import_module("numpy._core.multiarray")
            safe_globals: list[object] = [
                (
                    numpy_multiarray.scalar,
                    "numpy.core.multiarray.scalar",
                ),
                np.dtype,
                type(np.dtype(np.float64)),
            ]
            with torch_module.serialization.safe_globals(safe_globals):
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


class _TorchRuntime:
    """Lazy PyTorch boundary for strict official-model construction and inference."""

    def __init__(
        self,
        source_root: Path,
        weight_path: Path,
        *,
        contract: WeightAssetContract,
        device: str,
    ) -> None:
        verified_source = verify_source_tree(source_root)
        try:
            torch_module = importlib.import_module("torch")
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is required at runtime for HyperSIGMA; install torch, "
                "timm, and einops in the local execution environment"
            ) from error
        state_dict, verified_weight = load_verified_state_dict(
            weight_path, contract, torch_module
        )
        model = _build_official_model(verified_source)
        incompatible = model.load_state_dict(state_dict, strict=True)
        missing_keys = tuple(getattr(incompatible, "missing_keys", ()))
        unexpected_keys = tuple(getattr(incompatible, "unexpected_keys", ()))
        if missing_keys or unexpected_keys:
            raise ValueError(
                "strict HyperSIGMA state load reported incompatible keys"
            )
        model.requires_grad_(False)
        model.eval()
        if getattr(model, "training", True):
            raise RuntimeError("HyperSIGMA model did not enter evaluation mode")
        self._torch = torch_module
        self._device = self._resolve_device(device)
        self._model = model.to(self._device)
        self._runtime_identity = HyperSIGMARuntimeIdentity(
            source_commit=verified_source.commit,
            source_sha256=verified_source.file_sha256,
            weight=verified_weight,
            device=str(self._device),
        )

    def _resolve_device(self, requested: str) -> Any:
        if requested == "auto":
            name = "cuda:0" if self._torch.cuda.is_available() else "cpu"
        elif requested == "cuda":
            name = "cuda:0"
        else:
            name = requested
        if name != "cpu" and not name.startswith("cuda:"):
            raise ValueError("HyperSIGMA device must be auto, cpu, cuda, or cuda:N")
        device = self._torch.device(name)
        if device.type == "cuda" and not self._torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device

    @property
    def runtime_identity(self) -> HyperSIGMARuntimeIdentity:
        return self._runtime_identity

    def predict(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if values.shape != (191, 64, 64) or not np.isfinite(values).all():
            raise ValueError("HyperSIGMA runtime requires finite (191, 64, 64) values")
        tensor = self._torch.from_numpy(
            np.ascontiguousarray(values, dtype=np.float32)
        ).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            predicted = self._model(tensor)
        array = predicted.detach().cpu().numpy()
        if array.shape != (1, 191, 64, 64) or not np.isfinite(array).all():
            raise ValueError("HyperSIGMA returned an invalid output cube")
        return np.asarray(array[0], dtype=np.float64)


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
