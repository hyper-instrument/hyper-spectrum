from __future__ import annotations

import numpy as np
import pytest

from hyperspectrum.plugins.xas.arrays import XASSpectrum
from hyperspectrum.plugins.xas.baselines import (
    BaselineFailure,
    BaselinePrediction,
    gaussian_filter,
    gaussian_process_filter,
    identity_filter,
    moving_average_filter,
    pca_filter,
    savgol_filter,
)


def samples() -> tuple[XASSpectrum, ...]:
    energy = np.linspace(7110.0, 7118.0, 9)
    return tuple(
        XASSpectrum(
            sample_id=f"sample-{index}",
            group_id=("FeO", "Fe2O3", "Fe3O4")[index],
            energy=energy,
            intensity=np.sin(np.linspace(0.0, np.pi, 9)) + 0.03 * index + noise,
            energy_unit="eV",
        )
        for index, noise in enumerate(
            (
                np.array([0.02, -0.01, 0.03, -0.02, 0.0, 0.01, -0.02, 0.01, -0.01]),
                np.array([-0.01, 0.02, -0.02, 0.01, 0.03, -0.01, 0.01, -0.02, 0.02]),
                np.array([0.03, 0.01, -0.01, -0.03, 0.02, 0.0, -0.01, 0.02, -0.02]),
            )
        )
    )


@pytest.mark.parametrize(
    ("method", "runner"),
    [
        ("identity", lambda values: identity_filter(values)),
        ("moving_average", lambda values: moving_average_filter(values, window_size=3)),
        ("gaussian", lambda values: gaussian_filter(values, sigma=1.0)),
        ("savitzky_golay", lambda values: savgol_filter(values, window_length=5, polyorder=2)),
        ("pca", lambda values: pca_filter(values, n_components=2)),
        (
            "gaussian_process",
            lambda values: gaussian_process_filter(values, length_scale=1.5, noise_level=0.05),
        ),
    ],
)
def test_each_baseline_returns_one_finite_prediction_with_original_coordinates_and_ids(
    method: str, runner: object
) -> None:
    # Break caught: a denoiser could reorder/drop samples or emit values on an inferred grid.
    source = samples()
    results = runner(source)  # type: ignore[operator]

    assert len(results) == len(source)
    for original, result in zip(source, results):
        assert isinstance(result, BaselinePrediction)
        assert result.method == method
        assert (result.spectrum.sample_id, result.spectrum.group_id) == (
            original.sample_id,
            original.group_id,
        )
        np.testing.assert_array_equal(result.spectrum.energy, original.energy)
        assert result.spectrum.energy_unit == "eV"
        assert np.isfinite(result.spectrum.intensity).all()


@pytest.mark.parametrize(
    "runner",
    [
        lambda values: identity_filter(values),
        lambda values: moving_average_filter(values, window_size=3),
        lambda values: gaussian_filter(values, sigma=1.0),
        lambda values: savgol_filter(values, window_length=5, polyorder=2),
    ],
)
def test_no_fit_baselines_report_no_training(runner: object) -> None:
    # Break caught: a classical smoother could be misreported as trained or runtime-fitted.
    results = runner(samples())  # type: ignore[operator]

    assert {result.optimization_kind for result in results if isinstance(result, BaselinePrediction)} == {
        "no_training"
    }


@pytest.mark.parametrize(
    "runner",
    [
        lambda values: pca_filter(values, n_components=2),
        lambda values: gaussian_process_filter(values, length_scale=1.5, noise_level=0.05),
    ],
)
def test_noisy_only_fitted_baselines_report_runtime_fit(runner: object) -> None:
    # Break caught: per-run numerical fitting could be mislabeled as model training.
    results = runner(samples())  # type: ignore[operator]

    assert {result.optimization_kind for result in results if isinstance(result, BaselinePrediction)} == {
        "runtime_fit"
    }


def test_identity_returns_an_equal_but_detached_spectrum() -> None:
    # Break caught: identity could alias caller-owned memory and allow output mutation to corrupt input.
    original = samples()[0]
    result = identity_filter((original,))[0]

    assert isinstance(result, BaselinePrediction)
    np.testing.assert_array_equal(result.spectrum.intensity, original.intensity)
    assert not np.shares_memory(result.spectrum.intensity, original.intensity)


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (lambda values: moving_average_filter(values, window_size=4), "odd"),
        (lambda values: gaussian_filter(values, sigma=0.0), "positive"),
        (lambda values: savgol_filter(values, window_length=4, polyorder=2), "odd"),
        (lambda values: pca_filter(values, n_components=4), "n_components"),
        (
            lambda values: gaussian_process_filter(values, length_scale=-1.0, noise_level=0.05),
            "length_scale",
        ),
    ],
)
def test_unsupported_settings_return_one_typed_failure_per_input(
    runner: object, message: str
) -> None:
    # Break caught: invalid settings could raise globally or silently omit affected samples.
    source = samples()
    results = runner(source)  # type: ignore[operator]

    assert len(results) == len(source)
    assert all(isinstance(result, BaselineFailure) for result in results)
    assert all(result.error_type == "unsupported_setting" for result in results)
    assert all(message in result.message for result in results)


def test_pca_rejects_nonidentical_energy_grids_without_interpolation() -> None:
    # Break caught: PCA could combine columns representing different physical energy coordinates.
    source = list(samples())
    source[1] = XASSpectrum(
        sample_id=source[1].sample_id,
        group_id=source[1].group_id,
        energy=source[1].energy + 0.1,
        intensity=source[1].intensity,
        energy_unit="eV",
    )

    results = pca_filter(source, n_components=2)

    assert len(results) == len(source)
    assert all(isinstance(result, BaselineFailure) for result in results)
    assert all(result.error_type == "unsupported_grid" for result in results)


def test_runtime_fits_are_deterministic_for_identical_noisy_inputs() -> None:
    # Break caught: randomized PCA or GP fitting could change predictions between identical runs.
    source = samples()

    first_pca = pca_filter(source, n_components=2)
    second_pca = pca_filter(source, n_components=2)
    first_gp = gaussian_process_filter(source, length_scale=1.5, noise_level=0.05)
    second_gp = gaussian_process_filter(source, length_scale=1.5, noise_level=0.05)

    for first, second in zip(first_pca + first_gp, second_pca + second_gp):
        assert isinstance(first, BaselinePrediction)
        assert isinstance(second, BaselinePrediction)
        np.testing.assert_array_equal(first.spectrum.intensity, second.spectrum.intensity)
