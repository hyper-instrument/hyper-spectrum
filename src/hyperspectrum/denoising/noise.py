"""Deterministic synthetic corruption in native spectral signal space."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import numpy as np

from hyperspectrum.contracts.json import (
    FrozenJsonMapping,
    freeze_json_mapping,
    thaw_json_mapping,
)

from .sample import SpectrumRepresentation, SpectrumSample, spectrum_digest

NoiseKind: TypeAlias = Literal["gaussian_signal", "poisson_counts"]


@dataclass(frozen=True, slots=True)
class NoiseSpec:
    """A fully seeded native-space corruption request."""

    kind: NoiseKind
    seed: int
    parameters: FrozenJsonMapping | Mapping[str, object]

    def __post_init__(self) -> None:
        if self.kind not in {"gaussian_signal", "poisson_counts"}:
            raise ValueError(f"unsupported noise kind: {self.kind}")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        parameters = freeze_json_mapping(self.parameters)
        expected = "sigma" if self.kind == "gaussian_signal" else "count_scale"
        if set(parameters) != {expected}:
            raise ValueError(f"{self.kind} requires exactly the {expected} parameter")
        value = parameters[expected]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{expected} must be a finite number")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{expected} must be finite")
        if self.kind == "gaussian_signal" and numeric < 0.0:
            raise ValueError("sigma must be non-negative")
        if self.kind == "poisson_counts" and numeric <= 0.0:
            raise ValueError("count_scale must be greater than zero")
        object.__setattr__(self, "parameters", parameters)


@dataclass(frozen=True, slots=True)
class NoiseProvenance:
    """Replayable description of one synthetic native-space corruption."""

    schema_version: Literal["hyperspectrum-noise/v1"]
    kind: NoiseKind
    seed: int
    parameters: FrozenJsonMapping | Mapping[str, object]
    clean_sample_digest: str
    native_signal_unit: str
    representation: SpectrumRepresentation

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", freeze_json_mapping(self.parameters))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "seed": self.seed,
            "parameters": thaw_json_mapping(cast(FrozenJsonMapping, self.parameters)),
            "clean_sample_digest": self.clean_sample_digest,
            "native_signal_unit": self.native_signal_unit,
            "representation": self.representation,
        }


@dataclass(frozen=True, slots=True)
class CorruptedSpectrum:
    """A noisy sample plus immutable corruption provenance."""

    sample: SpectrumSample
    provenance: NoiseProvenance


def inject_noise(sample: SpectrumSample, spec: NoiseSpec) -> CorruptedSpectrum:
    """Inject noise without normalization, type coercion, or dimensional changes."""
    if np.iscomplexobj(sample.signal) or sample.representation == "complex":
        raise ValueError(
            "built-in synthetic noise currently requires a real-valued signal"
        )
    if spec.kind == "gaussian_signal" and sample.representation != "dense":
        raise ValueError("Gaussian signal noise requires the dense representation")
    if spec.kind == "poisson_counts" and sample.signal_unit not in {"count", "counts"}:
        raise ValueError("Poisson count noise requires an explicit count signal unit")
    valid_values = sample.signal[sample.valid_mask]
    rng = np.random.default_rng(spec.seed)
    corrupted = np.array(sample.signal, dtype=np.float64, copy=True)
    if spec.kind == "gaussian_signal":
        sigma_value = spec.parameters["sigma"]
        if not isinstance(sigma_value, (int, float)) or isinstance(sigma_value, bool):
            raise TypeError("sigma must be numeric")
        sigma = float(sigma_value)
        noise = rng.normal(0.0, sigma, size=valid_values.shape)
        corrupted[sample.valid_mask] = valid_values + noise
    elif spec.kind == "poisson_counts":
        if np.any(valid_values < 0.0):
            raise ValueError("Poisson count noise requires non-negative valid values")
        count_scale_value = spec.parameters["count_scale"]
        if not isinstance(count_scale_value, (int, float)) or isinstance(
            count_scale_value, bool
        ):
            raise TypeError("count_scale must be numeric")
        count_scale = float(count_scale_value)
        corrupted[sample.valid_mask] = (
            rng.poisson(valid_values * count_scale) / count_scale
        )
    else:  # pragma: no cover - NoiseSpec closes this branch.
        raise ValueError(f"unsupported noise kind: {spec.kind}")
    noisy_sample = sample.with_signal(corrupted)
    provenance = NoiseProvenance(
        schema_version="hyperspectrum-noise/v1",
        kind=spec.kind,
        seed=spec.seed,
        parameters=spec.parameters,
        clean_sample_digest=spectrum_digest(sample),
        native_signal_unit=sample.signal_unit,
        representation=sample.representation,
    )
    return CorruptedSpectrum(noisy_sample, provenance)
