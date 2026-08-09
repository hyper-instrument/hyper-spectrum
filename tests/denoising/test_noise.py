from __future__ import annotations

import re

import numpy as np
import pytest

from hyperspectrum.denoising import SpectrumAxis, SpectrumSample
from hyperspectrum.denoising.noise import NoiseSpec, inject_noise


def sample(
    signal: np.ndarray, *, representation: str = "dense", modality: str = "xas"
) -> SpectrumSample:
    return SpectrumSample(
        sample_id="sample-1",
        group_id="group-a",
        modality=modality,  # type: ignore[arg-type]
        representation=representation,  # type: ignore[arg-type]
        axes=(
            SpectrumAxis(
                name="energy" if modality != "mass_spectrometry" else "mass_to_charge",
                unit="eV" if modality != "mass_spectrometry" else "m/z",
                direction="increasing",
                values=np.arange(1, signal.shape[-1] + 1, dtype=float),
            ),
        ),
        signal=signal,
        valid_mask=np.ones(signal.shape, dtype=bool),
        signal_unit="count" if modality == "mass_spectrometry" else "mu(E)",
        metadata={},
        provenance={},
    )


def test_gaussian_noise_is_seeded_native_space_and_provenance_complete() -> None:
    # Break caught: noise could be injected after normalization or without a replayable seed.
    clean = sample(np.array([10.0, 20.0, 30.0]))
    spec = NoiseSpec(kind="gaussian_signal", seed=17, parameters={"sigma": 0.5})

    first = inject_noise(clean, spec)
    second = inject_noise(clean, spec)

    np.testing.assert_array_equal(first.sample.signal, second.sample.signal)
    assert not np.array_equal(first.sample.signal, clean.signal)
    assert first.sample.signal_unit == "mu(E)"
    assert first.provenance.seed == 17
    assert first.provenance.parameters == {"sigma": 0.5}
    assert first.provenance.native_signal_unit == "mu(E)"
    assert first.provenance.representation == "dense"
    assert re.fullmatch(r"[0-9a-f]{64}", first.provenance.clean_sample_digest)


def test_poisson_noise_operates_on_nonnegative_count_space() -> None:
    # Break caught: count noise could be approximated in normalized space or admit negative rates.
    clean = sample(
        np.array([10.0, 20.0, 30.0]),
        modality="mass_spectrometry",
        representation="sparse_peaks",
    )
    corrupted = inject_noise(
        clean,
        NoiseSpec(kind="poisson_counts", seed=3, parameters={"count_scale": 2.0}),
    )

    assert corrupted.sample.signal.tolist() == [6.0, 23.0, 24.0]
    assert corrupted.sample.signal_unit == "count"
    assert corrupted.provenance.kind == "poisson_counts"

    with pytest.raises(ValueError, match="non-negative"):
        inject_noise(
            sample(
                np.array([1.0, -1.0, 2.0]),
                modality="mass_spectrometry",
                representation="sparse_peaks",
            ),
            NoiseSpec(kind="poisson_counts", seed=3, parameters={"count_scale": 1.0}),
        )


def test_noise_rejects_complex_or_invalid_parameters_without_coercion() -> None:
    # Break caught: complex phase could be discarded by an accidental float cast.
    complex_nmr = sample(
        np.array([1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 0.5j]),
        representation="complex",
        modality="nmr",
    )

    with pytest.raises(ValueError, match="real-valued"):
        inject_noise(
            complex_nmr,
            NoiseSpec(kind="gaussian_signal", seed=0, parameters={"sigma": 0.1}),
        )
    with pytest.raises(ValueError, match="sigma"):
        inject_noise(
            sample(np.array([1.0, 2.0, 3.0])),
            NoiseSpec(kind="gaussian_signal", seed=0, parameters={"sigma": -0.1}),
        )
    with pytest.raises(ValueError, match="count_scale"):
        NoiseSpec(kind="poisson_counts", seed=0, parameters={"count_scale": 0.0})


def test_noise_models_are_bound_to_declared_signal_domains() -> None:
    # Break caught: Poisson could be applied to absorbance and Gaussian to sparse peaks.
    with pytest.raises(ValueError, match="count signal unit"):
        inject_noise(
            sample(np.array([1.0, 2.0, 3.0])),
            NoiseSpec(kind="poisson_counts", seed=0, parameters={"count_scale": 1.0}),
        )
    with pytest.raises(ValueError, match="dense representation"):
        inject_noise(
            sample(
                np.array([1.0, 2.0, 3.0]),
                modality="mass_spectrometry",
                representation="sparse_peaks",
            ),
            NoiseSpec(kind="gaussian_signal", seed=0, parameters={"sigma": 0.1}),
        )
