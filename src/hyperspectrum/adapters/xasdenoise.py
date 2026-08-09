"""Safe thin adapter for the official XASDenoise nonuniform checkpoint.

The architecture and step-baseline transform follow XASDenoise commit
``bda749ee956f9e02acc6995f238d759682ee2ca8`` (MIT). Checkpoint bytes remain
an external Zenodo asset and are never downloaded by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, TypeAlias, cast, overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from hyperspectrum.denoising.model import (
    CanonicalDenoisingInput,
    CanonicalDenoisingOutput,
    ModelCapabilities,
    validate_model_output,
)
from hyperspectrum.denoising.normalization import (
    NormalizedSpectrum,
    denormalize,
    normalize,
)
from hyperspectrum.denoising.sample import SpectrumAxis, SpectrumSample
from hyperspectrum.plugins.xas.arrays import XASSpectrum

XASDENOISE_SOURCE_COMMIT = "bda749ee956f9e02acc6995f238d759682ee2ca8"
XASDENOISE_CODE_LICENSE = "MIT"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_LAYERS = 4
_MODEL_KERNEL_SIZE = 9
_MODEL_EDGE_CROP = 16
XASDENOISE_REQUIRED_NORMALIZATION = "upstream_pre_edge_post_edge_normalized"


@dataclass(frozen=True, slots=True)
class WeightAssetContract:
    """Immutable identity of one externally mounted model checkpoint."""

    asset_id: str
    source_url: str
    filename: str
    size_bytes: int
    sha256: str
    license: str

    def __post_init__(self) -> None:
        for name in ("asset_id", "source_url", "filename", "license"):
            value = cast(str, getattr(self, name))
            if not value.strip():
                raise ValueError(f"weight {name} must be non-empty")
        if not self.source_url.startswith("https://"):
            raise ValueError("weight source_url must use HTTPS")
        if Path(self.filename).name != self.filename:
            raise ValueError("weight filename must be a basename")
        if self.size_bytes < 1:
            raise ValueError("weight size_bytes must be positive")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("weight sha256 must be a lowercase SHA-256")


XASDENOISE_WEIGHT = WeightAssetContract(
    asset_id="zenodo-17434349",
    source_url=(
        "https://zenodo.org/api/records/17434349/files/"
        "xas_denoiser_model_noise2noise_nonuniformly_sampled_notnormalized.pth/content"
    ),
    filename="xas_denoiser_model_noise2noise_nonuniformly_sampled_notnormalized.pth",
    size_bytes=780409,
    sha256="09620ee9ea0c96585f534d76ce42aa72edf2cf71e481f5737e43e93116e24160",
    license="CC-BY-4.0",
)


@dataclass(frozen=True, slots=True)
class VerifiedWeightAsset:
    """Immutable bytes proven to match one exact weight contract."""

    path: Path | None
    payload: bytes
    size_bytes: int
    sha256: str
    contract: WeightAssetContract


def verify_weight_bytes(
    payload: bytes,
    *,
    contract: WeightAssetContract = XASDENOISE_WEIGHT,
    source_path: Path | None = None,
) -> VerifiedWeightAsset:
    """Verify immutable checkpoint bytes without consulting a mutable pathname."""

    immutable = bytes(payload)
    size_bytes = len(immutable)
    if size_bytes != contract.size_bytes:
        raise ValueError(
            f"weight asset size mismatch: expected {contract.size_bytes}, got {size_bytes}"
        )
    digest = hashlib.sha256(immutable).hexdigest()
    if digest != contract.sha256:
        raise ValueError("weight asset SHA-256 mismatch")
    return VerifiedWeightAsset(
        path=source_path,
        payload=immutable,
        size_bytes=size_bytes,
        sha256=digest,
        contract=contract,
    )


def verify_weight_asset(
    path: Path, *, contract: WeightAssetContract = XASDENOISE_WEIGHT
) -> VerifiedWeightAsset:
    """Read once, then verify the exact bytes retained for deserialization."""

    try:
        with path.open("rb") as stream:
            payload = stream.read()
    except OSError as error:
        raise ValueError(f"cannot read weight asset: {error}") from error
    return verify_weight_bytes(
        payload,
        contract=contract,
        source_path=path,
    )


def load_state_dict_safely(
    verified: VerifiedWeightAsset, *, torch_module: Any
) -> Mapping[str, object]:
    """Load the exact verified bytes onto CPU and reject unsafe payload shapes."""

    loaded_size = len(verified.payload)
    loaded_digest = hashlib.sha256(verified.payload).hexdigest()
    if (
        loaded_size != verified.size_bytes
        or loaded_size != verified.contract.size_bytes
        or loaded_digest != verified.sha256
        or loaded_digest != verified.contract.sha256
    ):
        raise ValueError("loaded weight bytes do not match their verified identity")
    loaded = torch_module.load(
        io.BytesIO(verified.payload), map_location="cpu", weights_only=True
    )
    if not isinstance(loaded, Mapping) or not loaded:
        raise ValueError("checkpoint must contain one non-empty state dictionary")
    if any(not isinstance(key, str) or not key for key in loaded):
        raise ValueError("checkpoint state dictionary keys must be non-empty strings")
    return cast(Mapping[str, object], loaded)


def _canonical_float64_bytes(values: NDArray[np.float64]) -> bytes:
    return cast(bytes, np.ascontiguousarray(values, dtype="<f8").tobytes())


@dataclass(frozen=True, slots=True)
class XASDenoisePreprocessingState:
    """Replayable step subtraction within upstream-normalized absorption units."""

    schema_version: Literal["hyperspectrum-xasdenoise-step-baseline/v1"]
    method: Literal["symmetric_tanh_step"]
    edge_strategy: Literal["maximum_first_derivative"]
    fit_status: Literal["fitted", "initial_guess_fallback"]
    fit_parameters: tuple[float, float, float]
    baseline_sha256: str
    inverse: Literal["add_same_fitted_baseline"]
    normalization_method: Literal["upstream_pre_edge_post_edge_normalized"]
    model_normalization_method: None
    native_output_semantics: Literal[
        "model residual plus the exact fitted baseline in upstream-normalized absorption units"
    ]

    def __post_init__(self) -> None:
        parameters = tuple(float(value) for value in self.fit_parameters)
        object.__setattr__(self, "fit_parameters", parameters)
        if len(parameters) != 3 or not np.isfinite(parameters).all():
            raise ValueError("step-baseline fit parameters must be three finite values")
        if parameters[1] <= 0.0:
            raise ValueError("step-baseline width must be positive")
        if _SHA256.fullmatch(self.baseline_sha256) is None:
            raise ValueError("baseline_sha256 must be a lowercase SHA-256")

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON semantics for run provenance."""

        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "edge_strategy": self.edge_strategy,
            "fit_status": self.fit_status,
            "fit_parameters": list(self.fit_parameters),
            "baseline_sha256": self.baseline_sha256,
            "inverse": self.inverse,
            "normalization_method": self.normalization_method,
            "model_normalization_method": self.model_normalization_method,
            "native_output_semantics": self.native_output_semantics,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _validate_energy_signal(
    energy: NDArray[np.float64] | np.ndarray,
    signal: NDArray[np.float64] | np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.array(energy, dtype=np.float64, copy=True)
    y = np.array(signal, dtype=np.float64, copy=True)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("energy and signal must be equal one-dimensional arrays")
    if len(x) <= _MODEL_EDGE_CROP:
        raise ValueError("XASDenoise input is too short for reflect padding")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("energy and signal must contain only finite values")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("XASDenoise energy must be strictly increasing")
    return x, y


def _symmetric_tanh_step(
    energy: NDArray[np.float64], edge: float, width: float, unused: float
) -> NDArray[np.float64]:
    _ = unused
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return 0.5 * (1.0 + np.tanh((energy - edge) / width))


def preprocess_step_baseline(
    energy: NDArray[np.float64] | np.ndarray,
    signal: NDArray[np.float64] | np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.float64], XASDenoisePreprocessingState]:
    """Subtract the official step from an already normalized XAS spectrum."""

    x, y = _validate_energy_signal(energy, signal)
    derivatives = np.diff(y) / np.diff(x)
    edge_guess = float(x[int(np.argmax(derivatives))])
    width_guess = float((x[-1] - x[0]) * 0.05)
    initial = (edge_guess, width_guess, width_guess)
    status: Literal["fitted", "initial_guess_fallback"] = "fitted"
    try:
        fitted, _ = curve_fit(
            _symmetric_tanh_step,
            x,
            y,
            p0=initial,
            bounds=([float(x[0]), 0.0, 0.0], [float(x[-1]), np.inf, np.inf]),
            maxfev=100,
        )
        parameters = tuple(float(value) for value in fitted)
        if (
            len(parameters) != 3
            or not np.isfinite(parameters).all()
            or parameters[1] <= 0.0
        ):
            raise ValueError("non-finite fitted step parameters")
    except (RuntimeError, TypeError, ValueError, FloatingPointError):
        parameters = initial
        status = "initial_guess_fallback"
    baseline = np.asarray(_symmetric_tanh_step(x, *parameters), dtype=np.float64)
    if baseline.shape != y.shape or not np.isfinite(baseline).all():
        raise ValueError("step-baseline fitting produced an invalid baseline")
    transformed = np.asarray(y - baseline, dtype=np.float64)
    baseline.setflags(write=False)
    transformed.setflags(write=False)
    state = XASDenoisePreprocessingState(
        schema_version="hyperspectrum-xasdenoise-step-baseline/v1",
        method="symmetric_tanh_step",
        edge_strategy="maximum_first_derivative",
        fit_status=status,
        fit_parameters=parameters,
        baseline_sha256=hashlib.sha256(_canonical_float64_bytes(baseline)).hexdigest(),
        inverse="add_same_fitted_baseline",
        normalization_method="upstream_pre_edge_post_edge_normalized",
        model_normalization_method=None,
        native_output_semantics=(
            "model residual plus the exact fitted baseline in upstream-normalized absorption units"
        ),
    )
    return transformed, baseline, state


def restore_step_baseline(
    model_output: NDArray[np.float64] | np.ndarray,
    baseline: NDArray[np.float64] | np.ndarray,
    state: XASDenoisePreprocessingState,
) -> NDArray[np.float64]:
    """Apply the exact step inverse in upstream-normalized absorption units."""

    prediction = np.array(model_output, dtype=np.float64, copy=True)
    stored_baseline = np.array(baseline, dtype=np.float64, copy=True)
    if prediction.ndim != 1 or prediction.shape != stored_baseline.shape:
        raise ValueError("prediction and baseline must be equal one-dimensional arrays")
    observed_digest = hashlib.sha256(
        _canonical_float64_bytes(stored_baseline)
    ).hexdigest()
    if observed_digest != state.baseline_sha256:
        raise ValueError("baseline bytes do not match the preprocessing state")
    restored = prediction + stored_baseline
    if not np.isfinite(restored).all():
        raise ValueError("baseline restoration produced non-finite output")
    restored.setflags(write=False)
    return restored


def _build_official_model(torch_module: ModuleType) -> Any:
    """Build the exact fixed-weight architecture without importing upstream code."""

    nn = torch_module.nn

    class ConvDenoisingAutoencoder(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            channels = [16 * (2**index) for index in range(_MODEL_LAYERS)]
            encoder: list[Any] = []
            input_channels = 1
            for output_channels in channels:
                encoder.extend(
                    [
                        nn.Conv1d(
                            input_channels,
                            output_channels,
                            kernel_size=_MODEL_KERNEL_SIZE,
                            padding=_MODEL_KERNEL_SIZE // 2,
                            bias=False,
                            padding_mode="reflect",
                        ),
                        nn.ReLU(),
                    ]
                )
                input_channels = output_channels
            self.encoder = nn.Sequential(*encoder)
            decoder: list[Any] = []
            reversed_channels = list(reversed(channels))
            for index in range(len(reversed_channels) - 1):
                decoder.extend(
                    [
                        nn.ConvTranspose1d(
                            reversed_channels[index],
                            reversed_channels[index + 1],
                            kernel_size=_MODEL_KERNEL_SIZE,
                            padding=_MODEL_KERNEL_SIZE // 2,
                            bias=False,
                        ),
                        nn.ReLU(),
                    ]
                )
            decoder.append(
                nn.ConvTranspose1d(
                    reversed_channels[-1],
                    1,
                    kernel_size=_MODEL_KERNEL_SIZE,
                    padding=_MODEL_KERNEL_SIZE // 2,
                    bias=False,
                )
            )
            self.decoder = nn.Sequential(*decoder)

        def forward(self, values: Any) -> Any:
            expanded = values.unsqueeze(1)
            padded = nn.functional.pad(
                expanded, (_MODEL_EDGE_CROP, _MODEL_EDGE_CROP), mode="reflect"
            )
            decoded = self.decoder(self.encoder(padded))
            decoded = decoded[..., _MODEL_EDGE_CROP:-_MODEL_EDGE_CROP]
            return decoded.squeeze(1)

    return ConvDenoisingAutoencoder()


def _parameter_digest(model: Any) -> str:
    hasher = hashlib.sha256()
    state = model.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("model state dictionary is empty")
    for key, tensor in sorted(state.items()):
        values = np.ascontiguousarray(tensor.detach().cpu().numpy())
        hasher.update(key.encode("utf-8"))
        hasher.update(str(values.dtype).encode("ascii"))
        hasher.update(json.dumps(values.shape, separators=(",", ":")).encode("ascii"))
        hasher.update(values.tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class XASDenoiseRuntimeIdentity:
    """Batch-level runtime identity bound into every executed XAS run."""

    schema_version: Literal["hyperspectrum-xasdenoise-runtime/v1"]
    requested_device: str
    resolved_device: str
    backend: Literal["cpu", "cuda"]
    torch_version: str
    cuda_version: str | None
    cudnn_version: str | None
    loaded_weight_sha256: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", self.requested_device) is None:
            raise ValueError("requested_device is invalid")
        if re.fullmatch(r"cpu|cuda:[0-9]+", self.resolved_device) is None:
            raise ValueError("resolved_device must name an actual CPU or CUDA device")
        expected_backend = (
            "cuda" if self.resolved_device.startswith("cuda:") else "cpu"
        )
        if self.backend != expected_backend:
            raise ValueError("runtime backend does not match the resolved device")
        if not self.torch_version.strip() or self.torch_version == "unknown":
            raise ValueError("torch_version must identify the actual Torch runtime")
        if self.backend == "cuda" and self.cuda_version is None:
            raise ValueError("CUDA execution requires a CUDA runtime version")
        if _SHA256.fullmatch(self.loaded_weight_sha256) is None:
            raise ValueError("loaded_weight_sha256 must be a lowercase SHA-256")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "requested_device": self.requested_device,
            "device": self.resolved_device,
            "resolved_device": self.resolved_device,
            "backend": self.backend,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "loaded_weight_sha256": self.loaded_weight_sha256,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class _TorchRuntime:
    """Torch boundary isolated from preprocessing and canonical contracts."""

    def __init__(self, verified: VerifiedWeightAsset, *, device: str) -> None:
        try:
            torch_module = importlib.import_module("torch")
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is required at runtime for XASDenoise; install it in the OCI/local environment"
            ) from error
        self._torch = torch_module
        self.version = str(getattr(torch_module, "__version__", "unknown"))
        self.requested_device = device
        self._loaded_weight_sha256 = verified.sha256
        self.device = self._resolve_device(device)
        state = load_state_dict_safely(verified, torch_module=torch_module)
        model = _build_official_model(torch_module)
        model.load_state_dict(state, strict=True)
        self._model = model.to(self.device)
        self._model.eval()
        if self._model.training:
            raise RuntimeError("XASDenoise model did not enter evaluation mode")

    def _resolve_device(self, requested: str) -> Any:
        if requested == "auto":
            name = "cuda:0" if self._torch.cuda.is_available() else "cpu"
        elif requested == "cuda":
            name = "cuda:0"
        else:
            name = requested
        device = self._torch.device(name)
        if device.type == "cuda" and not self._torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device

    @property
    def device_name(self) -> str:
        return str(self.device)

    @property
    def runtime_identity(self) -> XASDenoiseRuntimeIdentity:
        version_namespace = getattr(self._torch, "version", None)
        cuda_version_value = getattr(version_namespace, "cuda", None)
        cuda_version = (
            str(cuda_version_value) if cuda_version_value is not None else None
        )
        cudnn = getattr(getattr(self._torch, "backends", None), "cudnn", None)
        cudnn_version_fn = getattr(cudnn, "version", None)
        cudnn_version_value = (
            cudnn_version_fn() if callable(cudnn_version_fn) else None
        )
        cudnn_version = (
            str(cudnn_version_value) if cudnn_version_value is not None else None
        )
        backend: Literal["cpu", "cuda"] = (
            "cuda" if self.device.type == "cuda" else "cpu"
        )
        return XASDenoiseRuntimeIdentity(
            schema_version="hyperspectrum-xasdenoise-runtime/v1",
            requested_device=self.requested_device,
            resolved_device=self.device_name,
            backend=backend,
            torch_version=self.version,
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            loaded_weight_sha256=self._loaded_weight_sha256,
        )

    def predict(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        before = _parameter_digest(self._model)
        tensor = self._torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))[
            None, :
        ].to(self.device)
        with self._torch.inference_mode():
            predicted = self._model(tensor)
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)
        output = np.array(predicted.detach().cpu().numpy()[0], dtype=np.float64)
        after = _parameter_digest(self._model)
        if after != before:
            raise RuntimeError("XASDenoise parameters changed during inference")
        if output.shape != values.shape:
            raise RuntimeError("XASDenoise output shape does not match input")
        if not np.isfinite(output).all():
            raise RuntimeError("XASDenoise output contains non-finite values")
        return output


class XASDenoiseAdapter:
    """Canonical denoising-model adapter for the pinned official checkpoint."""

    capabilities = ModelCapabilities(
        schema_version="hyperspectrum-denoising-capabilities/v1",
        axis_ranks=(1,),
        representations=("dense",),
        channel_counts=(1,),
        required_normalization=XASDENOISE_REQUIRED_NORMALIZATION,
        native_unit_recovery=False,
    )

    def __init__(
        self,
        weights_path: Path | None = None,
        *,
        weight_bytes: bytes | None = None,
        device: str = "auto",
    ) -> None:
        if (weights_path is None) == (weight_bytes is None):
            raise ValueError("provide exactly one of weights_path or weight_bytes")
        verified = (
            verify_weight_asset(weights_path)
            if weights_path is not None
            else verify_weight_bytes(cast(bytes, weight_bytes))
        )
        self._runtime = _TorchRuntime(verified, device=device)
        self._preprocessing_states: dict[str, XASDenoisePreprocessingState] = {}

    @staticmethod
    def validate_input(model_input: CanonicalDenoisingInput) -> None:
        """Reject unsupported physics/shape before fitting or tensor allocation."""

        if model_input.modality not in {"xas", "xanes"}:
            raise ValueError("XASDenoise supports only XAS/XANES inputs")
        if model_input.representation != "dense":
            raise ValueError("XASDenoise requires a real dense representation")
        if model_input.signal.ndim != 1 or model_input.channel_labels:
            raise ValueError("XASDenoise requires one unchannelled spectrum")
        if len(model_input.axis_values) != 1:
            raise ValueError("XASDenoise requires one explicit physical axis")
        if (
            model_input.axis_names != ("energy",)
            or model_input.axis_units != ("eV",)
            or model_input.axis_directions != ("increasing",)
        ):
            raise ValueError("XASDenoise requires a strictly increasing energy/eV axis")
        if not np.all(np.diff(model_input.axis_values[0]) > 0.0):
            raise ValueError("XASDenoise energy must be strictly increasing")
        if model_input.normalization_method != XASDENOISE_REQUIRED_NORMALIZATION:
            raise ValueError(
                "input_contract_unverified: the pinned checkpoint requires the "
                "upstream per-spectrum pre-edge/post-edge normalization state; "
                "raw ketek/i0 ratios and generic normalization are not compatible"
            )
        if not np.all(model_input.valid_mask):
            raise ValueError("XASDenoise does not support masked signal values")
        _validate_energy_signal(model_input.axis_values[0], model_input.signal)

    @property
    def device(self) -> str:
        return self._runtime.device_name

    @property
    def torch_version(self) -> str:
        return self._runtime.version

    @property
    def runtime_identity(self) -> XASDenoiseRuntimeIdentity:
        return self._runtime.runtime_identity

    def preprocessing_state(self, sample_id: str) -> XASDenoisePreprocessingState:
        try:
            return self._preprocessing_states[sample_id]
        except KeyError as error:
            raise KeyError(f"no preprocessing state for sample {sample_id}") from error

    def predict(self, model_input: CanonicalDenoisingInput) -> CanonicalDenoisingOutput:
        self.validate_input(model_input)
        transformed, baseline, state = preprocess_step_baseline(
            model_input.axis_values[0], model_input.signal
        )
        residual = self._runtime.predict(transformed)
        restored_normalized = restore_step_baseline(residual, baseline, state)
        self._preprocessing_states[model_input.sample_id] = state
        return CanonicalDenoisingOutput(
            sample_id=model_input.sample_id,
            signal=restored_normalized,
            valid_mask=model_input.valid_mask,
            normalization_state_digest=model_input.normalization_state_digest,
        )


@dataclass(frozen=True, slots=True)
class XASDenoisePrediction:
    """One successful prediction plus its normalized-unit preprocessing evidence."""

    spectrum: XASSpectrum
    method: Literal["xasdenoise"]
    optimization_kind: Literal["no_training"]
    preprocessing_state: XASDenoisePreprocessingState
    device: str
    torch_version: str | None = None


@dataclass(frozen=True, slots=True)
class XASDenoiseFailure:
    """One accounted model failure retaining identity and physical coordinates."""

    sample_id: str
    group_id: str
    energy: NDArray[np.float64]
    energy_unit: Literal["eV"]
    method: Literal["xasdenoise"]
    error_type: str
    message: str

    def __post_init__(self) -> None:
        energy = np.array(self.energy, dtype=np.float64, copy=True)
        energy.setflags(write=False)
        object.__setattr__(self, "energy", energy)


XASDenoiseResult: TypeAlias = XASDenoisePrediction | XASDenoiseFailure


@dataclass(frozen=True, slots=True)
class XASDenoiseBatchResult(Sequence[XASDenoiseResult]):
    """Ordered outcomes plus runtime identity, including all-failure batches."""

    results: tuple[XASDenoiseResult, ...]
    runtime_identity: XASDenoiseRuntimeIdentity

    def __len__(self) -> int:
        return len(self.results)

    @overload
    def __getitem__(self, index: int) -> XASDenoiseResult: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[XASDenoiseResult, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> XASDenoiseResult | tuple[XASDenoiseResult, ...]:
        return self.results[index]


def denoise_spectra(
    samples: Sequence[XASSpectrum],
    *,
    weights_path: Path | None = None,
    weight_bytes: bytes | None = None,
    device: str = "auto",
    _adapter: XASDenoiseAdapter | None = None,
) -> XASDenoiseBatchResult:
    """Run the one official adapter over ordered XAS inputs on any host backend."""

    adapter = _adapter or XASDenoiseAdapter(
        weights_path, weight_bytes=weight_bytes, device=device
    )
    results: list[XASDenoiseResult] = []
    for source in samples:
        try:
            direction: Literal["increasing", "decreasing"] = (
                "increasing" if source.energy[0] < source.energy[-1] else "decreasing"
            )
            sample = SpectrumSample(
                sample_id=source.sample_id,
                group_id=source.group_id,
                modality="xas",
                representation="dense",
                axes=(
                    SpectrumAxis(
                        name="energy",
                        unit=source.energy_unit,
                        direction=direction,
                        values=source.energy,
                    ),
                ),
                signal=source.intensity,
                valid_mask=np.ones(source.intensity.shape, dtype=bool),
                signal_unit="ketek/i0 ratio",
                metadata={},
                provenance={},
            )
            normalized = normalize(sample, "identity_raw")
            model_input = CanonicalDenoisingInput.from_normalized(normalized)
            output = adapter.predict(model_input)
            validate_model_output(model_input, output)
            normalized_prediction = NormalizedSpectrum(
                normalized.sample.with_signal(output.signal), normalized.state
            )
            native_prediction = denormalize(normalized_prediction)
            prediction = XASSpectrum(
                sample_id=source.sample_id,
                group_id=source.group_id,
                energy=source.energy,
                intensity=np.asarray(native_prediction.signal, dtype=np.float64),
                energy_unit="eV",
            )
            results.append(
                XASDenoisePrediction(
                    spectrum=prediction,
                    method="xasdenoise",
                    optimization_kind="no_training",
                    preprocessing_state=adapter.preprocessing_state(source.sample_id),
                    device=adapter.device,
                    torch_version=getattr(adapter, "torch_version", None),
                )
            )
        except (RuntimeError, TypeError, ValueError) as error:
            results.append(
                XASDenoiseFailure(
                    sample_id=source.sample_id,
                    group_id=source.group_id,
                    energy=source.energy,
                    energy_unit="eV",
                    method="xasdenoise",
                    error_type="model_failure",
                    message=str(error),
                )
            )
    return XASDenoiseBatchResult(
        results=tuple(results), runtime_identity=adapter.runtime_identity
    )


def verification_report() -> dict[str, object]:
    """Return the static, network-free identity checked by registry verification."""

    return {
        "adapter": "xasdenoise",
        "source_commit": XASDENOISE_SOURCE_COMMIT,
        "code_license": XASDENOISE_CODE_LICENSE,
        "checkpoint_normalization_method": None,
        "required_input_normalization": XASDENOISE_REQUIRED_NORMALIZATION,
        "raw_input_contract_status": "unverified",
        "availability_reason": "input_contract_unverified",
        "preprocessing_schema_version": ("hyperspectrum-xasdenoise-step-baseline/v1"),
        "weight": {
            "asset_id": XASDENOISE_WEIGHT.asset_id,
            "source_url": XASDENOISE_WEIGHT.source_url,
            "filename": XASDENOISE_WEIGHT.filename,
            "size_bytes": XASDENOISE_WEIGHT.size_bytes,
            "sha256": XASDENOISE_WEIGHT.sha256,
            "license": XASDENOISE_WEIGHT.license,
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Provide the manifest's static, network-free verification command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="emit the pinned adapter identity"
    )
    arguments = parser.parse_args(argv)
    if not arguments.verify:
        parser.error("--verify is required")
    print(json.dumps(verification_report(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
