"""Canonical model boundary for modality-independent denoising reuse."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .normalization import NormalizedSpectrum
from .sample import (
    AxisDirection,
    SpectralModality,
    SpectrumRepresentation,
    SpectrumSample,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    """One structured reason a model cannot consume an otherwise valid sample."""

    code: Literal[
        "axis_rank",
        "representation",
        "channel_count",
        "normalization",
        "native_unit_recovery",
    ]
    message: str


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Shape/representation contract deliberately independent of modality names."""

    schema_version: Literal["hyperspectrum-denoising-capabilities/v1"]
    axis_ranks: tuple[int, ...]
    representations: tuple[SpectrumRepresentation, ...]
    channel_counts: tuple[int, ...]
    required_normalization: str
    native_unit_recovery: bool

    def __post_init__(self) -> None:
        axis_ranks = tuple(self.axis_ranks)
        representations = tuple(self.representations)
        channel_counts = tuple(self.channel_counts)
        object.__setattr__(self, "axis_ranks", axis_ranks)
        object.__setattr__(self, "representations", representations)
        object.__setattr__(self, "channel_counts", channel_counts)
        if self.schema_version != "hyperspectrum-denoising-capabilities/v1":
            raise ValueError("unsupported denoising capability schema")
        if not axis_ranks or any(rank < 1 for rank in axis_ranks):
            raise ValueError("axis_ranks must contain positive integers")
        if not representations:
            raise ValueError("representations must be non-empty")
        if not channel_counts or any(count < 1 for count in channel_counts):
            raise ValueError("channel_counts must contain positive integers")
        if not self.required_normalization.strip():
            raise ValueError("required_normalization must be non-empty")
        if len(axis_ranks) != len(set(axis_ranks)):
            raise ValueError("axis_ranks must be unique")
        if len(representations) != len(set(representations)):
            raise ValueError("representations must be unique")
        if len(channel_counts) != len(set(channel_counts)):
            raise ValueError("channel_counts must be unique")

    def check(
        self,
        normalized: NormalizedSpectrum,
        *,
        require_native_unit_recovery: bool = False,
    ) -> tuple[CompatibilityIssue, ...]:
        """Return every incompatibility without coercing or executing the model."""
        return self.check_sample(
            normalized.sample,
            normalization_method=normalized.state.method,
            require_native_unit_recovery=require_native_unit_recovery,
        )

    def check_sample(
        self,
        sample: SpectrumSample,
        *,
        normalization_method: str,
        require_native_unit_recovery: bool = False,
    ) -> tuple[CompatibilityIssue, ...]:
        """Preflight raw structure before any potentially failing normalization."""
        issues: list[CompatibilityIssue] = []
        if sample.axis_rank not in self.axis_ranks:
            issues.append(
                CompatibilityIssue(
                    "axis_rank",
                    f"axis rank {sample.axis_rank} is not in {self.axis_ranks}",
                )
            )
        if sample.representation not in self.representations:
            issues.append(
                CompatibilityIssue(
                    "representation",
                    f"representation {sample.representation} is not supported",
                )
            )
        if sample.channel_count not in self.channel_counts:
            issues.append(
                CompatibilityIssue(
                    "channel_count",
                    f"channel count {sample.channel_count} is not in {self.channel_counts}",
                )
            )
        if normalization_method != self.required_normalization:
            issues.append(
                CompatibilityIssue(
                    "normalization",
                    f"requires {self.required_normalization}, got {normalization_method}",
                )
            )
        if require_native_unit_recovery and not self.native_unit_recovery:
            issues.append(
                CompatibilityIssue(
                    "native_unit_recovery",
                    "model does not declare native-unit recovery support",
                )
            )
        return tuple(issues)


def normalization_state_digest(normalized: NormalizedSpectrum) -> str:
    """Bind canonical model I/O to the exact persisted normalization state."""
    payload = json.dumps(
        normalized.state.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _readonly_numeric(values: np.ndarray, *, name: str) -> NDArray[np.generic]:
    array = np.array(values, copy=True)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(f"{name} must be numeric")
    array.setflags(write=False)
    return array


def _readonly_mask(values: np.ndarray) -> NDArray[np.bool_]:
    array = np.array(values, copy=True)
    if array.dtype != np.dtype(np.bool_):
        raise ValueError("valid_mask must be boolean")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CanonicalDenoisingInput:
    """Normalized, immutable model input with all physical axes retained."""

    sample_id: str
    modality: SpectralModality
    representation: SpectrumRepresentation
    signal: NDArray[np.generic]
    valid_mask: NDArray[np.bool_]
    axis_values: tuple[NDArray[np.float64], ...]
    axis_names: tuple[str, ...]
    axis_units: tuple[str, ...]
    axis_directions: tuple[AxisDirection, ...]
    channel_labels: tuple[str, ...]
    normalization_method: str
    normalization_state_digest: str

    def __post_init__(self) -> None:
        axis_names = tuple(self.axis_names)
        axis_units = tuple(self.axis_units)
        axis_directions = tuple(self.axis_directions)
        channel_labels = tuple(self.channel_labels)
        object.__setattr__(self, "axis_names", axis_names)
        object.__setattr__(self, "axis_units", axis_units)
        object.__setattr__(self, "axis_directions", axis_directions)
        object.__setattr__(self, "channel_labels", channel_labels)
        signal = _readonly_numeric(self.signal, name="signal")
        mask = _readonly_mask(self.valid_mask)
        axes: list[NDArray[np.float64]] = []
        for values in self.axis_values:
            axis = np.array(values, dtype=np.float64, copy=True)
            if axis.ndim != 1 or not np.isfinite(axis).all():
                raise ValueError("canonical axis values must be finite vectors")
            axis.setflags(write=False)
            axes.append(axis)
        if mask.shape != signal.shape:
            raise ValueError("canonical valid_mask shape must match signal shape")
        if not np.isfinite(signal[mask]).all():
            raise ValueError("canonical valid signal values must be finite")
        if not (
            len(axes) == len(axis_names) == len(axis_units) == len(axis_directions)
            and len(axes) >= 1
        ):
            raise ValueError("canonical axes, names, units, and directions must align")
        if any(
            direction not in {"increasing", "decreasing", "unordered"}
            for direction in axis_directions
        ):
            raise ValueError("canonical axis direction is invalid")
        if len(axis_names) != len(set(axis_names)):
            raise ValueError("canonical axis names must be unique")
        if _SHA256.fullmatch(self.normalization_state_digest) is None:
            raise ValueError("normalization_state_digest must be a lowercase SHA-256")
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "axis_values", tuple(axes))

    @classmethod
    def from_normalized(cls, normalized: NormalizedSpectrum) -> CanonicalDenoisingInput:
        sample = normalized.sample
        return cls(
            sample_id=sample.sample_id,
            modality=sample.modality,
            representation=sample.representation,
            signal=sample.signal,
            valid_mask=sample.valid_mask,
            axis_values=tuple(axis.values for axis in sample.axes),
            axis_names=tuple(axis.name for axis in sample.axes),
            axis_units=tuple(axis.unit for axis in sample.axes),
            axis_directions=tuple(axis.direction for axis in sample.axes),
            channel_labels=sample.channel_labels,
            normalization_method=normalized.state.method,
            normalization_state_digest=normalization_state_digest(normalized),
        )


@dataclass(frozen=True, slots=True)
class CanonicalDenoisingOutput:
    """Normalized model output; validation against its input is mandatory."""

    sample_id: str
    signal: NDArray[np.generic]
    valid_mask: NDArray[np.bool_]
    normalization_state_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "signal", _readonly_numeric(self.signal, name="signal")
        )
        object.__setattr__(self, "valid_mask", _readonly_mask(self.valid_mask))
        if _SHA256.fullmatch(self.normalization_state_digest) is None:
            raise ValueError("normalization_state_digest must be a lowercase SHA-256")


def validate_model_output(
    model_input: CanonicalDenoisingInput, output: CanonicalDenoisingOutput
) -> None:
    """Reject output drift; never align, reshape, fill, or reinterpret it."""
    if output.sample_id != model_input.sample_id:
        raise ValueError("model output sample_id does not match input")
    if output.signal.shape != model_input.signal.shape:
        raise ValueError("model output signal shape does not match input")
    if output.valid_mask.shape != model_input.valid_mask.shape or not np.array_equal(
        output.valid_mask, model_input.valid_mask
    ):
        raise ValueError("model output valid_mask does not match input")
    if output.normalization_state_digest != model_input.normalization_state_digest:
        raise ValueError("model output normalization state does not match input")
    if not np.isfinite(output.signal[output.valid_mask]).all():
        raise ValueError("model output valid values must be finite")


class DenoisingModel(Protocol):
    """Small adapter protocol implemented by classic or learned denoisers."""

    capabilities: ModelCapabilities

    def predict(self, model_input: CanonicalDenoisingInput) -> CanonicalDenoisingOutput:
        """Return one normalized prediction bound to the exact input state."""
        ...
