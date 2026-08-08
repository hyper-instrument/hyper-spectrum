"""Deterministic classical denoising baselines for canonical XAS spectra."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter as scipy_savgol_filter

from .arrays import XASSpectrum

OptimizationKind = Literal["no_training", "runtime_fit"]


@dataclass(frozen=True, slots=True)
class BaselinePrediction:
    """One successful baseline output."""

    spectrum: XASSpectrum
    method: str
    optimization_kind: OptimizationKind


@dataclass(frozen=True, slots=True)
class BaselineFailure:
    """One typed baseline failure retaining the input sample coordinates."""

    sample_id: str
    group_id: str
    energy: NDArray[np.float64]
    energy_unit: Literal["eV"]
    method: str
    error_type: str
    message: str

    def __post_init__(self) -> None:
        energy = np.array(self.energy, dtype=np.float64, copy=True)
        energy.setflags(write=False)
        object.__setattr__(self, "energy", energy)


BaselineResult: TypeAlias = BaselinePrediction | BaselineFailure


def _prediction(
    source: XASSpectrum,
    intensity: NDArray[np.float64],
    *,
    method: str,
    optimization_kind: OptimizationKind,
) -> BaselinePrediction:
    return BaselinePrediction(
        spectrum=XASSpectrum(
            sample_id=source.sample_id,
            group_id=source.group_id,
            energy=source.energy,
            intensity=intensity,
            energy_unit="eV",
        ),
        method=method,
        optimization_kind=optimization_kind,
    )


def _failure(
    source: XASSpectrum, *, method: str, error_type: str, message: str
) -> BaselineFailure:
    return BaselineFailure(
        sample_id=source.sample_id,
        group_id=source.group_id,
        energy=source.energy,
        energy_unit="eV",
        method=method,
        error_type=error_type,
        message=message,
    )


def _fail_all(
    samples: Sequence[XASSpectrum], *, method: str, error_type: str, message: str
) -> tuple[BaselineResult, ...]:
    return tuple(_failure(item, method=method, error_type=error_type, message=message) for item in samples)


def _map_filter(
    samples: Sequence[XASSpectrum],
    *,
    method: str,
    transform: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> tuple[BaselineResult, ...]:
    results: list[BaselineResult] = []
    for item in samples:
        try:
            filtered = np.asarray(transform(item.intensity), dtype=np.float64)
            results.append(
                _prediction(item, filtered, method=method, optimization_kind="no_training")
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            results.append(
                _failure(
                    item,
                    method=method,
                    error_type="numerical_failure",
                    message=str(error),
                )
            )
    return tuple(results)


def identity_filter(samples: Sequence[XASSpectrum]) -> tuple[BaselineResult, ...]:
    """Return detached copies of every input spectrum without fitting."""

    return _map_filter(
        samples,
        method="identity",
        transform=lambda intensity: np.array(intensity, copy=True),
    )


def moving_average_filter(
    samples: Sequence[XASSpectrum], *, window_size: int = 3
) -> tuple[BaselineResult, ...]:
    """Apply a centered moving average with truncated endpoint windows."""

    if window_size < 1 or window_size % 2 == 0:
        return _fail_all(
            samples,
            method="moving_average",
            error_type="unsupported_setting",
            message="window_size must be a positive odd integer",
        )
    if any(window_size > len(item.intensity) for item in samples):
        return _fail_all(
            samples,
            method="moving_average",
            error_type="unsupported_setting",
            message="window_size cannot exceed the spectrum length",
        )
    kernel: NDArray[np.float64] = np.ones(window_size, dtype=np.float64)

    def transform(intensity: NDArray[np.float64]) -> NDArray[np.float64]:
        sums = np.convolve(intensity, kernel, mode="same")
        counts = np.convolve(np.ones_like(intensity), kernel, mode="same")
        return sums / counts

    return _map_filter(samples, method="moving_average", transform=transform)


def gaussian_filter(
    samples: Sequence[XASSpectrum], *, sigma: float = 1.0
) -> tuple[BaselineResult, ...]:
    """Apply a one-dimensional Gaussian smoother without fitting."""

    if not np.isfinite(sigma) or sigma <= 0.0:
        return _fail_all(
            samples,
            method="gaussian",
            error_type="unsupported_setting",
            message="sigma must be a finite positive value",
        )
    return _map_filter(
        samples,
        method="gaussian",
        transform=lambda intensity: np.asarray(
            gaussian_filter1d(intensity, sigma=sigma, mode="reflect"), dtype=np.float64
        ),
    )


def savgol_filter(
    samples: Sequence[XASSpectrum], *, window_length: int = 5, polyorder: int = 2
) -> tuple[BaselineResult, ...]:
    """Apply a local Savitzky-Golay filter through a manifest-resolvable wrapper."""

    if window_length < 1 or window_length % 2 == 0:
        return _fail_all(
            samples,
            method="savitzky_golay",
            error_type="unsupported_setting",
            message="window_length must be a positive odd integer",
        )
    if polyorder < 0 or polyorder >= window_length:
        return _fail_all(
            samples,
            method="savitzky_golay",
            error_type="unsupported_setting",
            message="polyorder must be non-negative and less than window_length",
        )
    if any(window_length > len(item.intensity) for item in samples):
        return _fail_all(
            samples,
            method="savitzky_golay",
            error_type="unsupported_setting",
            message="window_length cannot exceed the spectrum length",
        )
    return _map_filter(
        samples,
        method="savitzky_golay",
        transform=lambda intensity: np.asarray(
            scipy_savgol_filter(
                intensity,
                window_length=window_length,
                polyorder=polyorder,
                mode="interp",
            ),
            dtype=np.float64,
        ),
    )


def pca_filter(
    samples: Sequence[XASSpectrum], *, n_components: int = 2
) -> tuple[BaselineResult, ...]:
    """Fit PCA at runtime to the noisy input matrix and reconstruct every row."""

    method = "pca"
    if not samples:
        return ()
    maximum_components = min(len(samples), len(samples[0].intensity))
    if n_components < 1 or n_components > maximum_components:
        return _fail_all(
            samples,
            method=method,
            error_type="unsupported_setting",
            message=f"n_components must be between 1 and {maximum_components}",
        )
    reference_energy = samples[0].energy
    if any(
        item.energy.shape != reference_energy.shape
        or not np.array_equal(item.energy, reference_energy)
        for item in samples[1:]
    ):
        return _fail_all(
            samples,
            method=method,
            error_type="unsupported_grid",
            message="PCA requires identical explicit energy grids",
        )

    noisy_matrix = np.stack([item.intensity for item in samples])
    center = np.mean(noisy_matrix, axis=0)
    try:
        left, singular_values, right = np.linalg.svd(noisy_matrix - center, full_matrices=False)
        reconstructed = (left[:, :n_components] * singular_values[:n_components]) @ right[
            :n_components
        ] + center
    except np.linalg.LinAlgError as error:
        return _fail_all(
            samples,
            method=method,
            error_type="numerical_failure",
            message=str(error),
        )
    return tuple(
        _prediction(item, row, method=method, optimization_kind="runtime_fit")
        for item, row in zip(samples, reconstructed)
    )


def gaussian_process_filter(
    samples: Sequence[XASSpectrum],
    *,
    length_scale: float = 1.5,
    noise_level: float = 0.05,
) -> tuple[BaselineResult, ...]:
    """Fit a fixed-hyperparameter RBF Gaussian process to each noisy input spectrum."""

    method = "gaussian_process"
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        return _fail_all(
            samples,
            method=method,
            error_type="unsupported_setting",
            message="length_scale must be a finite positive value",
        )
    if not np.isfinite(noise_level) or noise_level <= 0.0:
        return _fail_all(
            samples,
            method=method,
            error_type="unsupported_setting",
            message="noise_level must be a finite positive value",
        )

    results: list[BaselineResult] = []
    for item in samples:
        distances = item.energy[:, None] - item.energy[None, :]
        kernel = np.exp(-0.5 * (distances / length_scale) ** 2)
        covariance = kernel + (noise_level**2) * np.eye(len(item.energy))
        try:
            coefficients = np.linalg.solve(covariance, item.intensity)
            fitted = kernel @ coefficients
            results.append(
                _prediction(item, fitted, method=method, optimization_kind="runtime_fit")
            )
        except np.linalg.LinAlgError as error:
            results.append(
                _failure(
                    item,
                    method=method,
                    error_type="numerical_failure",
                    message=str(error),
                )
            )
    return tuple(results)
