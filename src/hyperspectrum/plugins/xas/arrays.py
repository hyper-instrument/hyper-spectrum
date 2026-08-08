"""Canonical in-memory XAS spectrum arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

EnergyUnit = Literal["eV"]


def _finite_vector(values: ArrayLike, *, name: str, minimum_length: int = 2) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if len(array) < minimum_length:
        raise ValueError(f"{name} must contain at least {minimum_length} values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _require_strict_monotonic(energy: NDArray[np.float64]) -> None:
    differences = np.diff(energy)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError("energy must be strictly monotonic")


@dataclass(frozen=True, slots=True)
class XASSpectrum:
    """One XAS spectrum with explicit identity, grouping, coordinates, and units."""

    sample_id: str
    group_id: str
    energy: NDArray[np.float64]
    intensity: NDArray[np.float64]
    energy_unit: EnergyUnit

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("group_id must be a non-empty string")
        if self.energy_unit != "eV":
            raise ValueError("energy_unit must be exactly 'eV'")

        energy = _finite_vector(self.energy, name="energy")
        intensity = _finite_vector(self.intensity, name="intensity")
        if energy.shape != intensity.shape:
            raise ValueError("energy and intensity must have the same length")
        _require_strict_monotonic(energy)
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "intensity", intensity)


def interpolate_spectrum(
    source: XASSpectrum,
    target_energy: ArrayLike,
    *,
    energy_unit: EnergyUnit,
) -> XASSpectrum:
    """Interpolate onto explicit coordinates wholly inside the closed range overlap.

    The function deliberately rejects partially overlapping grids rather than
    extrapolating, filling, or clipping their out-of-range coordinates.
    """

    if energy_unit != "eV":
        raise ValueError("energy_unit must be exactly 'eV'")
    target = _finite_vector(target_energy, name="target energy")
    _require_strict_monotonic(target)
    overlap_lower = max(float(np.min(source.energy)), float(np.min(target)))
    overlap_upper = min(float(np.max(source.energy)), float(np.max(target)))
    if overlap_lower > overlap_upper or np.any(target < overlap_lower) or np.any(target > overlap_upper):
        raise ValueError("every target coordinate must lie inside the closed overlap")

    if source.energy[0] < source.energy[-1]:
        source_energy = source.energy
        source_intensity = source.intensity
    else:
        source_energy = source.energy[::-1]
        source_intensity = source.intensity[::-1]
    interpolated = np.interp(target, source_energy, source_intensity)
    return XASSpectrum(
        sample_id=source.sample_id,
        group_id=source.group_id,
        energy=target,
        intensity=interpolated,
        energy_unit="eV",
    )
