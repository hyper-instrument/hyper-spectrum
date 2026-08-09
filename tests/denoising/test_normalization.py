from __future__ import annotations

import re
from typing import cast

import numpy as np
import pytest

from hyperspectrum.denoising import SpectrumAxis, SpectrumSample
from hyperspectrum.denoising.normalization import (
    NormalizationState,
    NormalizedSpectrum,
    default_normalization_registry,
    denormalize,
    fit_normalizer,
    normalize,
    recommended_normalization,
)
from hyperspectrum.denoising.split import SplitEntry, SplitManifest


def sample(
    sample_id: str,
    signal: np.ndarray | list[float] | list[complex],
    *,
    modality: str = "xas",
    representation: str = "dense",
    unit: str = "mu(E)",
    axis_name: str = "energy",
    axis_unit: str = "eV",
) -> SpectrumSample:
    values = np.asarray(signal)
    return SpectrumSample(
        sample_id=sample_id,
        group_id="group-a",
        modality=modality,  # type: ignore[arg-type]
        representation=representation,  # type: ignore[arg-type]
        axes=(
            SpectrumAxis(
                name=axis_name,
                unit=axis_unit,
                direction="increasing",
                values=np.arange(1, values.shape[-1] + 1, dtype=float),
            ),
        ),
        signal=values,
        valid_mask=np.ones(values.shape, dtype=bool),
        signal_unit=unit,
        metadata={},
        provenance={},
    )


def manifest(*entries: tuple[str, str, str]) -> SplitManifest:
    return SplitManifest(
        schema_version="hyperspectrum-split-manifest/v1",
        entries=tuple(
            SplitEntry(sample_id, group_id, split)  # type: ignore[arg-type]
            for sample_id, group_id, split in entries
        ),
    )


def test_per_spectrum_range_is_exact_reversible_and_sample_bound() -> None:
    # Break caught: normalization parameters could be lost or recomputed from the clean target.
    source = sample("xas-1", [2.0, 4.0, 6.0])

    normalized = normalize(source, "per_spectrum_range")

    assert normalized.sample.signal.tolist() == [0.0, 0.5, 1.0]
    assert normalized.sample.signal_unit == "1"
    assert normalized.state.fit_scope == "per_sample"
    assert normalized.state.invertibility == "exact"
    assert normalized.state.sample_id == "xas-1"
    assert normalized.state.parameters == {"offset": 2.0, "scale": 4.0}
    assert re.fullmatch(r"[0-9a-f]{64}", normalized.state.fit_data_digest)

    recovered = denormalize(normalized)
    np.testing.assert_array_equal(recovered.signal, source.signal)
    assert recovered.signal_unit == "mu(E)"

    with pytest.raises(ValueError, match="sample-bound"):
        normalize(
            sample("xas-2", [2.0, 4.0, 6.0]),
            "per_spectrum_range",
            state=normalized.state,
        )


def test_normalized_value_object_rejects_a_foreign_per_sample_state() -> None:
    # Break caught: sample B could be denormalized with sample A's physical parameters.
    first = normalize(sample("xas-a", [2.0, 4.0, 6.0]), "per_spectrum_range")
    second = normalize(sample("xas-b", [20.0, 40.0, 60.0]), "per_spectrum_range")

    with pytest.raises(ValueError, match="sample-bound"):
        NormalizedSpectrum(second.sample, first.state)


@pytest.mark.parametrize("split", ["val", "test"])
def test_dataset_fitted_normalizer_rejects_non_train_fit(split: str) -> None:
    # Break caught: validation/test values could leak into fitted centering and scale.
    with pytest.raises(ValueError, match="train split"):
        fit_normalizer(
            "train_global_standard",
            (sample("held-out", [100.0, 200.0, 300.0]),),
            manifest=manifest(("held-out", "group-a", split)),
        )


def test_train_global_standard_persists_only_training_identity_and_parameters() -> None:
    # Break caught: a fitted state could omit the exact samples/parameters needed for replay.
    train = (sample("train-a", [2.0, 4.0, 6.0]), sample("train-b", [4.0, 6.0, 8.0]))
    held_out = sample("test-z", [100.0, 102.0, 104.0])

    state = fit_normalizer(
        "train_global_standard",
        train,
        manifest=manifest(
            ("train-a", "group-a", "train"),
            ("train-b", "group-a", "train"),
        ),
    )
    transformed = normalize(held_out, "train_global_standard", state=state)

    assert state.training_sample_ids == ("train-a", "train-b")
    assert state.training_group_ids == ("group-a",)
    assert len(state.split_manifest_digest or "") == 64
    assert state.sample_id is None
    assert state.parameters["mean"] == pytest.approx(5.0)
    assert state.parameters["scale"] == pytest.approx(np.sqrt(11.0 / 3.0))
    assert state.to_dict()["training_sample_ids"] == ["train-a", "train-b"]
    assert transformed.state is state
    np.testing.assert_allclose(
        transformed.sample.signal,
        (held_out.signal - 5.0) / np.sqrt(11.0 / 3.0),
    )
    np.testing.assert_allclose(denormalize(transformed).signal, held_out.signal)


def test_fitted_state_rejects_modality_representation_or_native_unit_drift() -> None:
    # Break caught: a scaler fitted in one physical domain could be reused silently elsewhere.
    state = fit_normalizer(
        "train_global_standard",
        (sample("train", [1.0, 2.0, 3.0]),),
        manifest=manifest(("train", "group-a", "train")),
    )

    with pytest.raises(ValueError, match="modality"):
        normalize(
            sample("r", [1.0, 2.0, 3.0], modality="raman"), state.method, state=state
        )
    with pytest.raises(ValueError, match="signal unit"):
        normalize(
            sample("x", [1.0, 2.0, 3.0], unit="absorbance"), state.method, state=state
        )
    with pytest.raises(ValueError, match="representation"):
        normalize(
            sample(
                "n",
                [1.0 + 1.0j, 2.0 + 0.0j, 3.0 - 1.0j],
                modality="nmr",
                representation="complex",
                unit="V",
            ),
            state.method,
            state=state,
        )


def test_tic_and_complex_rms_preserve_sparse_counts_and_nmr_phase() -> None:
    # Break caught: modality defaults could destroy peak ratios or complex phase.
    mass = sample(
        "ms-1",
        [2.0, 3.0, 5.0],
        modality="mass_spectrometry",
        representation="sparse_peaks",
        unit="count",
        axis_name="mass_to_charge",
        axis_unit="m/z",
    )
    nmr = sample(
        "nmr-1",
        [3.0 + 4.0j, 0.0 + 5.0j, 4.0 + 3.0j],
        modality="nmr",
        representation="complex",
        unit="V",
        axis_name="time",
        axis_unit="s",
    )

    mass_norm = normalize(mass, "total_ion_current")
    nmr_norm = normalize(nmr, "complex_rms")

    assert mass_norm.sample.signal.tolist() == pytest.approx([0.2, 0.3, 0.5])
    np.testing.assert_allclose(nmr_norm.sample.signal / nmr.signal, np.full(3, 0.2))
    np.testing.assert_allclose(denormalize(mass_norm).signal, mass.signal)
    np.testing.assert_allclose(denormalize(nmr_norm).signal, nmr.signal)


def test_registry_and_recommendations_are_modality_aware() -> None:
    registry = default_normalization_registry()

    assert registry.get("train_global_standard").fit_scope == "dataset_fitted"
    assert recommended_normalization("xas", "dense") == "per_spectrum_range"
    assert recommended_normalization("raman", "dense") == "per_spectrum_range"
    assert (
        recommended_normalization("mass_spectrometry", "sparse_peaks")
        == "total_ion_current"
    )
    assert recommended_normalization("nmr", "complex") == "complex_rms"
    with pytest.raises(ValueError, match="no recommended normalization"):
        recommended_normalization("nmr", "dense")


def test_degenerate_or_negative_domain_normalization_fails_closed() -> None:
    with pytest.raises(ValueError, match="dynamic range"):
        normalize(sample("flat", [2.0, 2.0, 2.0]), "per_spectrum_range")
    with pytest.raises(ValueError, match="non-negative"):
        normalize(
            sample(
                "ms-negative",
                [2.0, -1.0, 5.0],
                modality="mass_spectrometry",
                representation="sparse_peaks",
                unit="count",
                axis_name="mass_to_charge",
                axis_unit="m/z",
            ),
            "total_ion_current",
        )


def test_normalization_state_cannot_be_constructed_with_unpersistable_parameters() -> (
    None
):
    with pytest.raises(ValueError):
        NormalizationState(
            schema_version="hyperspectrum-normalization-state/v1",
            method="bad",
            fit_scope="per_sample",
            invertibility="exact",
            modality="xas",
            representation="dense",
            signal_unit="mu(E)",
            parameters={"array": np.array([1.0])},
            training_sample_ids=(),
            training_group_ids=(),
            split_manifest_digest=None,
            fit_data_digest="a" * 64,
            sample_id="xas-1",
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"mean": 0.0},
        {"mean": 0.0, "scale": "oops"},
        {"mean": 0.0, "scale": 0.0},
        {"mean": float("nan"), "scale": 1.0},
    ],
)
def test_normalization_state_validates_method_parameters_on_load(
    parameters: dict[str, object],
) -> None:
    # Break caught: malformed persisted state could crash an entire suite during apply.
    with pytest.raises(ValueError, match="parameters"):
        NormalizationState(
            schema_version="hyperspectrum-normalization-state/v1",
            method="train_global_standard",
            fit_scope="dataset_fitted",
            invertibility="exact",
            modality="xas",
            representation="dense",
            signal_unit="mu(E)",
            parameters=parameters,
            training_sample_ids=("train-1",),
            training_group_ids=("group-a",),
            split_manifest_digest="b" * 64,
            fit_data_digest="a" * 64,
            sample_id=None,
        )


def test_reloaded_state_detaches_training_identity_lists() -> None:
    # Break caught: mutating a JSON payload could erase leakage identities in a frozen state.
    original = fit_normalizer(
        "train_global_standard",
        (sample("train-1", [1.0, 2.0, 3.0]),),
        manifest=manifest(("train-1", "group-a", "train")),
    )
    payload = original.to_dict()
    state = NormalizationState(**payload)  # type: ignore[arg-type]

    cast(list[str], payload["training_sample_ids"]).clear()
    cast(list[str], payload["training_group_ids"]).clear()

    assert state.training_sample_ids == ("train-1",)
    assert state.training_group_ids == ("group-a",)
