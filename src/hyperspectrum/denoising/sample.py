"""Immutable in-memory samples for model-independent spectral denoising."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hyperspectrum.contracts.json import FrozenJsonMapping, freeze_json_mapping

SpectralModality: TypeAlias = Literal[
    "xas",
    "xanes",
    "exafs",
    "raman",
    "ir",
    "nmr",
    "mass_spectrometry",
    "eels",
    "hyperspectral",
]
SpectrumRepresentation: TypeAlias = Literal["dense", "complex", "sparse_peaks"]
AxisDirection: TypeAlias = Literal["increasing", "decreasing", "unordered"]

SUPPORTED_MODALITIES = frozenset(
    {
        "xas",
        "xanes",
        "exafs",
        "raman",
        "ir",
        "nmr",
        "mass_spectrometry",
        "eels",
        "hyperspectral",
    }
)


def _nonblank(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _readonly_array(values: ArrayLike, *, name: str) -> NDArray[np.generic]:
    array = np.array(values, copy=True)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(f"{name} must be a numeric array")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class SpectrumAxis:
    """One explicitly named physical coordinate axis."""

    name: str
    unit: str
    direction: AxisDirection
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        _nonblank(self.name, name="axis name")
        _nonblank(self.unit, name="axis unit")
        if self.direction not in {"increasing", "decreasing", "unordered"}:
            raise ValueError(
                "axis direction must be increasing, decreasing, or unordered"
            )
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.ndim != 1 or len(values) < 1:
            raise ValueError("axis values must be a non-empty one-dimensional array")
        if not np.isfinite(values).all():
            raise ValueError("axis values must be finite")
        differences = np.diff(values)
        if self.direction == "increasing" and not np.all(differences > 0.0):
            raise ValueError("increasing axis values must be strictly increasing")
        if self.direction == "decreasing" and not np.all(differences < 0.0):
            raise ValueError("decreasing axis values must be strictly decreasing")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class SpectrumSample:
    """A strict spectral array with explicit axes, mask, units, and representation."""

    sample_id: str
    group_id: str
    modality: SpectralModality
    representation: SpectrumRepresentation
    axes: tuple[SpectrumAxis, ...]
    signal: NDArray[np.generic]
    valid_mask: NDArray[np.bool_]
    signal_unit: str
    channel_labels: tuple[str, ...] = ()
    metadata: FrozenJsonMapping | Mapping[str, object] = field(
        default_factory=lambda: freeze_json_mapping({})
    )
    provenance: FrozenJsonMapping | Mapping[str, object] = field(
        default_factory=lambda: freeze_json_mapping({})
    )

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        object.__setattr__(self, "axes", axes)
        _nonblank(self.sample_id, name="sample_id")
        _nonblank(self.group_id, name="group_id")
        _nonblank(self.signal_unit, name="signal_unit")
        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(f"unsupported modality: {self.modality}")
        if self.representation not in {"dense", "complex", "sparse_peaks"}:
            raise ValueError(f"unsupported representation: {self.representation}")
        if not axes:
            raise ValueError("at least one explicit axis is required")
        if any(not isinstance(axis, SpectrumAxis) for axis in axes):
            raise TypeError("axes must contain SpectrumAxis values")
        axis_names = tuple(axis.name for axis in axes)
        if len(axis_names) != len(set(axis_names)):
            raise ValueError("axis names must be unique")

        signal = _readonly_array(self.signal, name="signal")
        mask = np.array(self.valid_mask, copy=True)
        if mask.dtype != np.dtype(np.bool_):
            raise ValueError("valid_mask must be a boolean array")
        if mask.shape != signal.shape:
            raise ValueError("valid_mask shape must match signal shape")
        if not np.any(mask):
            raise ValueError("valid_mask must select at least one valid signal value")
        mask.setflags(write=False)

        labels = tuple(self.channel_labels)
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("channel labels must be non-empty strings")
        if len(labels) != len(set(labels)):
            raise ValueError("channel labels must be unique")
        if self.representation == "sparse_peaks" and (
            signal.ndim != 1 or len(axes) != 1 or labels
        ):
            raise ValueError("sparse_peaks must be one-dimensional without channels")
        axis_offset = 1 if labels else 0
        if signal.ndim != len(axes) + axis_offset:
            raise ValueError(
                "dense signal must have exactly one dimension per axis plus an explicit channel dimension"
            )
        if labels and signal.shape[0] != len(labels):
            raise ValueError("channel labels must match the leading signal dimension")
        for index, axis in enumerate(axes):
            if signal.shape[index + axis_offset] != len(axis.values):
                raise ValueError(f"signal dimension does not match axis {axis.name}")

        if self.representation == "complex":
            if not np.iscomplexobj(signal):
                raise ValueError(
                    "complex representation requires a complex-valued signal"
                )
        elif np.iscomplexobj(signal):
            raise ValueError("complex signals require the complex representation")
        if not np.isfinite(signal[mask]).all():
            raise ValueError("valid signal values must be finite")

        try:
            metadata = freeze_json_mapping(self.metadata)
            provenance = freeze_json_mapping(self.provenance)
        except TypeError as error:
            raise ValueError(str(error)) from error
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "channel_labels", labels)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "provenance", provenance)

    @property
    def axis_rank(self) -> int:
        """Return the number of declared physical axes, excluding channels."""
        return len(self.axes)

    @property
    def channel_count(self) -> int:
        """Return the explicit channel count, or one for an unchannelled signal."""
        return len(self.channel_labels) if self.channel_labels else 1

    def with_signal(
        self,
        signal: ArrayLike,
        *,
        signal_unit: str | None = None,
        provenance: FrozenJsonMapping | Mapping[str, object] | None = None,
    ) -> SpectrumSample:
        """Create a revalidated sample with a replacement signal."""
        return SpectrumSample(
            sample_id=self.sample_id,
            group_id=self.group_id,
            modality=self.modality,
            representation=self.representation,
            axes=self.axes,
            signal=np.asarray(signal),
            valid_mask=self.valid_mask,
            signal_unit=self.signal_unit if signal_unit is None else signal_unit,
            channel_labels=self.channel_labels,
            metadata=self.metadata,
            provenance=self.provenance if provenance is None else provenance,
        )


def _update_array_digest(hasher: object, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    canonical_dtype = contiguous.dtype.newbyteorder("<")
    canonical = contiguous.astype(canonical_dtype, copy=False)
    hasher.update(str(canonical.dtype).encode("ascii"))  # type: ignore[attr-defined]
    hasher.update(json.dumps(canonical.shape, separators=(",", ":")).encode("ascii"))  # type: ignore[attr-defined]
    hasher.update(canonical.tobytes(order="C"))  # type: ignore[attr-defined]


def samples_digest(samples: tuple[SpectrumSample, ...]) -> str:
    """Digest sample identity, physical declarations, coordinates, signal, and mask."""
    if not samples:
        raise ValueError("at least one sample is required for a digest")
    hasher = hashlib.sha256()
    for sample in samples:
        header = {
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
            "modality": sample.modality,
            "representation": sample.representation,
            "signal_unit": sample.signal_unit,
            "channels": sample.channel_labels,
            "axes": [
                {"name": axis.name, "unit": axis.unit, "direction": axis.direction}
                for axis in sample.axes
            ],
        }
        hasher.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        for axis in sample.axes:
            _update_array_digest(hasher, axis.values)
        _update_array_digest(hasher, sample.signal)
        _update_array_digest(hasher, sample.valid_mask)
    return hasher.hexdigest()


def spectrum_digest(sample: SpectrumSample) -> str:
    """Return the canonical digest for one strict spectrum sample."""
    return samples_digest((sample,))
