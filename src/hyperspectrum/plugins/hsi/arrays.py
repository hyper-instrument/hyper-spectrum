"""Explicit, shape-independent axis handling for hyperspectral cubes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from hyperspectrum.denoising.sample import SpectrumSample

SpectralAxisName: TypeAlias = Literal["band", "wavelength"]


@dataclass(frozen=True, slots=True)
class HSIAxisLayout:
    """One explicit permutation between declared HSI axes and model layout."""

    spectral_axis: SpectralAxisName
    original_axis_names: tuple[str, str, str]
    original_shape: tuple[int, int, int]
    model_shape: tuple[int, int, int]
    to_model_permutation: tuple[int, int, int]
    from_model_permutation: tuple[int, int, int]

    def to_model_layout(
        self, values: NDArray[np.generic]
    ) -> NDArray[np.generic]:
        """Transpose declared values to ``(spectral, y, x)`` without reshaping."""

        array = np.asarray(values)
        if array.shape != self.original_shape:
            raise ValueError("HSI values do not match the declared original shape")
        return np.transpose(array, self.to_model_permutation)

    def from_model_layout(
        self, values: NDArray[np.generic]
    ) -> NDArray[np.generic]:
        """Restore model values to the caller's exact declared axis order."""

        array = np.asarray(values)
        if array.shape != self.model_shape:
            raise ValueError("model values do not match the declared HSI model shape")
        return np.transpose(array, self.from_model_permutation)


def resolve_hsi_layout(
    axis_names: Sequence[str],
    axis_units: Sequence[str],
    axis_lengths: Sequence[int],
    *,
    expected_band_count: int | None = None,
    expected_spatial_shape: tuple[int, int] | None = None,
) -> HSIAxisLayout:
    """Validate named physical axes and return their reversible model permutation."""

    names_input = tuple(axis_names)
    units_input = tuple(axis_units)
    lengths_input = tuple(axis_lengths)
    if len(names_input) != 3 or len(units_input) != 3 or len(lengths_input) != 3:
        raise ValueError("HSI requires exactly three aligned axes")
    names = (names_input[0], names_input[1], names_input[2])
    units = (units_input[0], units_input[1], units_input[2])
    lengths = (lengths_input[0], lengths_input[1], lengths_input[2])
    if len(set(names)) != 3:
        raise ValueError("HSI axis names must be unique")
    spectral_names = tuple(
        name for name in names if name in {"band", "wavelength"}
    )
    if len(spectral_names) != 1:
        raise ValueError("HSI requires exactly one band or wavelength axis")
    spectral = cast(SpectralAxisName, spectral_names[0])
    if set(names) != {spectral, "y", "x"}:
        raise ValueError("HSI axes must contain the spectral axis, y, and x")
    spectral_index = names.index(spectral)
    if spectral == "band" and units[spectral_index] != "index":
        raise ValueError("band unit must be exactly index")
    if spectral == "wavelength" and units[spectral_index] == "index":
        raise ValueError("wavelength unit must be a physical unit, not index")
    if any(length < 1 for length in lengths):
        raise ValueError("HSI axis lengths must be positive")

    to_model = (spectral_index, names.index("y"), names.index("x"))
    model_shape = (
        lengths[to_model[0]],
        lengths[to_model[1]],
        lengths[to_model[2]],
    )
    if expected_band_count is not None and model_shape[0] != expected_band_count:
        raise ValueError(f"HSI band count must be {expected_band_count}")
    if expected_spatial_shape is not None and model_shape[1:] != expected_spatial_shape:
        raise ValueError(f"HSI spatial shape must be {expected_spatial_shape}")
    from_model = (to_model.index(0), to_model.index(1), to_model.index(2))
    return HSIAxisLayout(
        spectral_axis=spectral,
        original_axis_names=names,
        original_shape=lengths,
        model_shape=model_shape,
        to_model_permutation=to_model,
        from_model_permutation=from_model,
    )


def validate_hsi_sample(
    sample: SpectrumSample,
    *,
    expected_band_count: int | None = None,
    expected_spatial_shape: tuple[int, int] | None = None,
) -> HSIAxisLayout:
    """Validate one canonical sample's explicit HSI physical axes."""

    if sample.modality != "hyperspectral":
        raise ValueError("HSI sample modality must be hyperspectral")
    return resolve_hsi_layout(
        tuple(axis.name for axis in sample.axes),
        tuple(axis.unit for axis in sample.axes),
        tuple(len(axis.values) for axis in sample.axes),
        expected_band_count=expected_band_count,
        expected_spatial_shape=expected_spatial_shape,
    )
