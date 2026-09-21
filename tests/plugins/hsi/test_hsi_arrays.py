from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hyperspectrum.denoising import SpectrumAxis, SpectrumSample
from hyperspectrum.plugins.hsi.arrays import (
    resolve_hsi_layout,
    validate_hsi_sample,
)


@pytest.mark.parametrize(
    ("names", "units", "shape"),
    [
        (("band", "y", "x"), ("index", "pixel", "pixel"), (3, 2, 4)),
        (("band", "x", "y"), ("index", "pixel", "pixel"), (3, 4, 2)),
        (("y", "band", "x"), ("pixel", "index", "pixel"), (2, 3, 4)),
        (("y", "x", "band"), ("pixel", "pixel", "index"), (2, 4, 3)),
        (("x", "band", "y"), ("pixel", "index", "pixel"), (4, 3, 2)),
        (("x", "y", "band"), ("pixel", "pixel", "index"), (4, 2, 3)),
        (("y", "x", "wavelength"), ("pixel", "pixel", "nm"), (2, 4, 3)),
    ],
)
def test_explicit_layout_round_trips_without_guessing(
    names: tuple[str, str, str],
    units: tuple[str, str, str],
    shape: tuple[int, int, int],
) -> None:
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)

    layout = resolve_hsi_layout(
        names,
        units,
        shape,
        expected_band_count=3,
        expected_spatial_shape=(2, 4),
    )
    model_values = layout.to_model_layout(values)

    assert model_values.shape == (3, 2, 4)
    assert np.array_equal(layout.from_model_layout(model_values), values)


@pytest.mark.parametrize(
    ("names", "units", "message"),
    [
        (("band", "y", "time"), ("index", "pixel", "s"), "y, and x"),
        (
            ("band", "wavelength", "x"),
            ("index", "nm", "pixel"),
            "exactly one",
        ),
        (("band", "y", "y"), ("index", "pixel", "pixel"), "unique"),
        (("z", "y", "x"), ("pixel", "pixel", "pixel"), "band or wavelength"),
        (("band", "y", "x"), ("nm", "pixel", "pixel"), "band unit"),
        (
            ("wavelength", "y", "x"),
            ("index", "pixel", "pixel"),
            "wavelength unit",
        ),
    ],
)
def test_ambiguous_or_invalid_axes_are_rejected(
    names: tuple[str, str, str],
    units: tuple[str, str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_hsi_layout(names, units, (3, 2, 4))


def test_layout_rejects_rank_length_and_model_shape_drift() -> None:
    with pytest.raises(ValueError, match="exactly three aligned axes"):
        resolve_hsi_layout(("band", "x"), ("index", "pixel"), (3, 4))
    with pytest.raises(ValueError, match="positive"):
        resolve_hsi_layout(
            ("band", "y", "x"),
            ("index", "pixel", "pixel"),
            (3, 0, 4),
        )
    with pytest.raises(ValueError, match="band count must be 191"):
        resolve_hsi_layout(
            ("band", "y", "x"),
            ("index", "pixel", "pixel"),
            (190, 64, 64),
            expected_band_count=191,
            expected_spatial_shape=(64, 64),
        )
    with pytest.raises(ValueError, match=r"spatial shape must be \(64, 64\)"):
        resolve_hsi_layout(
            ("band", "y", "x"),
            ("index", "pixel", "pixel"),
            (191, 64, 63),
            expected_band_count=191,
            expected_spatial_shape=(64, 64),
        )

    layout = resolve_hsi_layout(
        ("band", "y", "x"),
        ("index", "pixel", "pixel"),
        (3, 2, 4),
    )
    with pytest.raises(ValueError, match="original shape"):
        layout.to_model_layout(np.ones((3, 4, 2)))
    with pytest.raises(ValueError, match="model shape"):
        layout.from_model_layout(np.ones((3, 4, 2)))


def _axis(name: str, unit: str, length: int) -> SpectrumAxis:
    return SpectrumAxis(
        name=name,
        unit=unit,
        direction="increasing",
        values=np.arange(length, dtype=np.float64),
    )


def _sample(*, modality: str = "hyperspectral") -> SpectrumSample:
    shape = (2, 4, 3)
    return SpectrumSample(
        sample_id="synthetic-hsi",
        group_id="synthetic-scene",
        modality=modality,  # type: ignore[arg-type]
        representation="dense",
        axes=(
            _axis("y", "pixel", 2),
            _axis("x", "pixel", 4),
            _axis("wavelength", "nm", 3),
        ),
        signal=np.arange(np.prod(shape), dtype=np.float64).reshape(shape),
        valid_mask=np.ones(shape, dtype=np.bool_),
        signal_unit="relative_reflectance",
    )


def test_sample_validation_uses_explicit_axes_and_expected_model_shape() -> None:
    layout = validate_hsi_sample(
        _sample(),
        expected_band_count=3,
        expected_spatial_shape=(2, 4),
    )
    assert layout.original_axis_names == ("y", "x", "wavelength")
    assert layout.model_shape == (3, 2, 4)

    with pytest.raises(ValueError, match="modality must be hyperspectral"):
        validate_hsi_sample(_sample(modality="eels"))


def test_text_fixture_is_explicitly_synthetic() -> None:
    fixture = (
        Path(__file__).parents[2] / "fixtures/hsi/denoising-pairs.json"
    )
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "hyperspectrum-hsi-synthetic-fixture/v1"
    assert payload["fixture_kind"] == "synthetic"
    assert payload["axis_names"] == ["band", "y", "x"]
    assert payload["axis_units"] == ["index", "pixel", "pixel"]
