from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from hyperspectrum.contracts.json import FrozenJsonMapping, thaw_json_mapping
from hyperspectrum.denoising import SpectrumAxis, SpectrumSample


def axis(
    values: list[float] | None = None,
    *,
    name: str = "energy",
    unit: str = "eV",
) -> SpectrumAxis:
    return SpectrumAxis(
        name=name,
        unit=unit,
        direction="increasing",
        values=np.array(values or [1.0, 2.0, 3.0]),
    )


def sample(**changes: object) -> SpectrumSample:
    values: dict[str, object] = {
        "sample_id": "xas-1",
        "group_id": "compound-a",
        "modality": "xas",
        "representation": "dense",
        "axes": (axis(),),
        "signal": np.array([2.0, 4.0, 6.0]),
        "valid_mask": np.array([True, True, True]),
        "signal_unit": "mu(E)",
        "channel_labels": (),
        "metadata": {"instrument": {"mode": "transmission"}},
        "provenance": {"source": "scan-1"},
    }
    values.update(changes)
    return SpectrumSample(**values)  # type: ignore[arg-type]


def test_sample_detaches_and_freezes_arrays_and_json() -> None:
    # Break caught: callers could mutate a sample after its digest or split was fixed.
    signal = np.array([2.0, 4.0, 6.0])
    mask = np.array([True, True, True])
    axes = [axis()]
    metadata = {"instrument": {"modes": ["transmission"]}}

    result = sample(
        signal=signal,
        valid_mask=mask,
        axes=axes,
        metadata=metadata,
    )
    signal[0] = 99.0
    mask[0] = False
    axes.clear()
    metadata["instrument"]["modes"].append("fluorescence")

    assert result.signal.tolist() == [2.0, 4.0, 6.0]
    assert result.valid_mask.tolist() == [True, True, True]
    assert len(result.axes) == 1
    assert thaw_json_mapping(cast(FrozenJsonMapping, result.metadata)) == {
        "instrument": {"modes": ["transmission"]}
    }
    with pytest.raises(ValueError, match="read-only"):
        result.signal[0] = 3.0
    with pytest.raises(TypeError):
        result.metadata["new"] = "value"  # type: ignore[index]


def test_dense_sample_requires_explicit_axis_shape_and_mask_shape() -> None:
    # Break caught: a 2-D image could be silently treated as a 1-D spectrum.
    with pytest.raises(ValueError, match="one dimension per axis"):
        sample(signal=np.ones((2, 3)), valid_mask=np.ones((2, 3), dtype=bool))

    with pytest.raises(ValueError, match="valid_mask shape"):
        sample(valid_mask=np.array([True, True]))


def test_multi_channel_signal_requires_explicit_unique_channel_labels() -> None:
    # Break caught: XAS detector channels could be mistaken for physical axes.
    result = sample(
        signal=np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]),
        valid_mask=np.ones((2, 3), dtype=bool),
        channel_labels=("transmission", "fluorescence"),
    )
    assert result.axis_rank == 1
    assert result.channel_count == 2

    with pytest.raises(ValueError, match="channel labels"):
        sample(
            signal=np.ones((2, 3)),
            valid_mask=np.ones((2, 3), dtype=bool),
            channel_labels=("same", "same"),
        )


def test_sparse_mass_spectrum_and_complex_nmr_are_explicit_representations() -> None:
    # Break caught: sparse peaks or phase-bearing NMR could be coerced to dense real arrays.
    mass = sample(
        sample_id="ms-1",
        modality="mass_spectrometry",
        representation="sparse_peaks",
        axes=(axis([100.0, 101.5, 150.0], name="mass_to_charge", unit="m/z"),),
        signal=np.array([20.0, 5.0, 3.0]),
        valid_mask=np.array([True, True, True]),
        signal_unit="count",
    )
    nmr = sample(
        sample_id="nmr-1",
        modality="nmr",
        representation="complex",
        axes=(axis([0.0, 1.0, 2.0], name="time", unit="s"),),
        signal=np.array([1.0 + 2.0j, 2.0 + 0.5j, 3.0 - 1.0j]),
        valid_mask=np.array([True, True, True]),
        signal_unit="V",
    )

    assert mass.representation == "sparse_peaks"
    assert np.iscomplexobj(nmr.signal)

    with pytest.raises(ValueError, match="complex representation"):
        sample(signal=np.array([1.0 + 1.0j, 2.0, 3.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        sample(
            modality="mass_spectrometry",
            representation="sparse_peaks",
            signal=np.ones((1, 3)),
            valid_mask=np.ones((1, 3), dtype=bool),
        )


def test_2d_eels_and_3d_hyperspectral_axes_are_preserved() -> None:
    # Break caught: spectral images could be flattened, losing spatial/physical axes.
    eels = sample(
        modality="eels",
        axes=(
            axis([0.0, 1.0], name="momentum_transfer", unit="1/angstrom"),
            axis([5.0, 6.0, 7.0], name="energy_loss", unit="eV"),
        ),
        signal=np.arange(6.0).reshape(2, 3),
        valid_mask=np.ones((2, 3), dtype=bool),
        signal_unit="count",
    )
    hyperspectral = sample(
        modality="hyperspectral",
        axes=(
            axis([0.0, 1.0], name="y", unit="pixel"),
            axis([0.0, 1.0], name="x", unit="pixel"),
            axis([500.0, 600.0, 700.0], name="wavelength", unit="nm"),
        ),
        signal=np.ones((2, 2, 3)),
        valid_mask=np.ones((2, 2, 3), dtype=bool),
        signal_unit="reflectance",
    )

    assert eels.axis_rank == 2
    assert hyperspectral.axis_rank == 3


@pytest.mark.parametrize(
    "modality",
    [
        "xas",
        "xanes",
        "exafs",
        "raman",
        "ir",
        "nmr",
        "mass_spectrometry",
        "eels",
        "hyperspectral",
    ],
)
def test_supported_modalities_are_explicit(modality: str) -> None:
    result = sample(modality=modality)
    assert result.modality == modality


def test_unknown_modality_and_nonfinite_valid_values_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported modality"):
        sample(modality="mystery")
    with pytest.raises(ValueError, match="valid signal values must be finite"):
        sample(signal=np.array([1.0, np.nan, 3.0]))

    masked = sample(
        signal=np.array([1.0, np.nan, 3.0]),
        valid_mask=np.array([True, False, True]),
    )
    assert np.isnan(masked.signal[1])


def test_sample_rejects_an_empty_valid_domain() -> None:
    # Break caught: an all-invalid sample could enter metrics as a perfect zero error.
    with pytest.raises(ValueError, match="at least one valid signal value"):
        sample(valid_mask=np.array([False, False, False]))
