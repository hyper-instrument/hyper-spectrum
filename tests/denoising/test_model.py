from __future__ import annotations

import numpy as np
import pytest

from hyperspectrum.denoising import SpectrumAxis, SpectrumSample, normalize
from hyperspectrum.denoising.model import (
    CanonicalDenoisingInput,
    CanonicalDenoisingOutput,
    ModelCapabilities,
    validate_model_output,
)


def sample(
    modality: str,
    signal: np.ndarray,
    *,
    representation: str = "dense",
    channel_labels: tuple[str, ...] = (),
) -> SpectrumSample:
    axis_rank = signal.ndim - (1 if channel_labels else 0)
    axes = tuple(
        SpectrumAxis(
            name=f"axis-{index}",
            unit="arb-axis",
            direction="increasing",
            values=np.arange(
                signal.shape[index + (1 if channel_labels else 0)], dtype=float
            ),
        )
        for index in range(axis_rank)
    )
    return SpectrumSample(
        sample_id=f"{modality}-1",
        group_id="group-a",
        modality=modality,  # type: ignore[arg-type]
        representation=representation,  # type: ignore[arg-type]
        axes=axes,
        signal=signal,
        valid_mask=np.ones(signal.shape, dtype=bool),
        signal_unit="native",
        channel_labels=channel_labels,
        metadata={},
        provenance={},
    )


def capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        schema_version="hyperspectrum-denoising-capabilities/v1",
        axis_ranks=(1,),
        representations=("dense",),
        channel_counts=(1,),
        required_normalization="per_spectrum_range",
        native_unit_recovery=True,
    )


def test_capability_is_modality_independent_and_accepts_xas_and_raman() -> None:
    # Break caught: a reusable 1-D model could be hard-coded to one modality name.
    model = capabilities()
    xas = normalize(sample("xas", np.array([1.0, 2.0, 4.0])), "per_spectrum_range")
    raman = normalize(sample("raman", np.array([2.0, 5.0, 8.0])), "per_spectrum_range")

    assert model.check(xas) == ()
    assert model.check(raman) == ()


def test_capability_detaches_runtime_lists() -> None:
    # Break caught: caller mutation could change a model's declared compatibility mid-suite.
    ranks = [1]
    representations = ["dense"]
    channels = [1]
    model = ModelCapabilities(
        schema_version="hyperspectrum-denoising-capabilities/v1",
        axis_ranks=ranks,  # type: ignore[arg-type]
        representations=representations,  # type: ignore[arg-type]
        channel_counts=channels,  # type: ignore[arg-type]
        required_normalization="per_spectrum_range",
        native_unit_recovery=True,
    )
    ranks.clear()
    representations.clear()
    channels.clear()
    normalized = normalize(
        sample("xas", np.array([1.0, 2.0, 4.0])), "per_spectrum_range"
    )

    assert model.check(normalized) == ()


def test_capability_returns_structured_issues_for_complex_nmr_and_2d_eels() -> None:
    # Break caught: incompatible complex/2-D arrays could be flattened or cast silently.
    model = capabilities()
    nmr = normalize(
        sample(
            "nmr",
            np.array([1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 1.0j]),
            representation="complex",
        ),
        "complex_rms",
    )
    eels = normalize(sample("eels", np.arange(6.0).reshape(2, 3)), "per_spectrum_range")

    assert {issue.code for issue in model.check(nmr)} == {
        "representation",
        "normalization",
    }
    assert {issue.code for issue in model.check(eels)} == {"axis_rank"}


def test_capability_checks_channels_normalization_and_native_recovery() -> None:
    multi_channel = normalize(
        sample(
            "xas",
            np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 5.0]]),
            channel_labels=("i0", "mu"),
        ),
        "per_spectrum_range",
    )
    no_inverse = ModelCapabilities(
        schema_version="hyperspectrum-denoising-capabilities/v1",
        axis_ranks=(1,),
        representations=("dense",),
        channel_counts=(1,),
        required_normalization="train_global_standard",
        native_unit_recovery=False,
    )

    assert {issue.code for issue in capabilities().check(multi_channel)} == {
        "channel_count"
    }
    assert {
        issue.code
        for issue in no_inverse.check(multi_channel, require_native_unit_recovery=True)
    } == {
        "channel_count",
        "normalization",
        "native_unit_recovery",
    }


def test_canonical_input_copies_arrays_and_binds_normalization_state() -> None:
    normalized = normalize(
        sample("xas", np.array([1.0, 2.0, 4.0])), "per_spectrum_range"
    )

    model_input = CanonicalDenoisingInput.from_normalized(normalized)

    assert model_input.sample_id == "xas-1"
    assert model_input.modality == "xas"
    assert model_input.axis_names == ("axis-0",)
    assert model_input.axis_units == ("arb-axis",)
    assert model_input.axis_directions == ("increasing",)
    assert model_input.normalization_method == "per_spectrum_range"
    assert len(model_input.normalization_state_digest) == 64
    with pytest.raises(ValueError, match="read-only"):
        model_input.signal[0] = 9.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sample_id": "wrong"}, "sample_id"),
        ({"signal": np.array([1.0, 2.0])}, "shape"),
        ({"valid_mask": np.array([True, False, False])}, "valid_mask"),
        ({"normalization_state_digest": "b" * 64}, "normalization state"),
    ],
)
def test_model_output_validation_rejects_identity_shape_mask_or_state_drift(
    changes: dict[str, object], message: str
) -> None:
    # Break caught: evaluator could score output for the wrong sample/state or repaired shape.
    normalized = normalize(
        sample("xas", np.array([1.0, 2.0, 4.0])), "per_spectrum_range"
    )
    model_input = CanonicalDenoisingInput.from_normalized(normalized)
    values: dict[str, object] = {
        "sample_id": model_input.sample_id,
        "signal": model_input.signal,
        "valid_mask": model_input.valid_mask,
        "normalization_state_digest": model_input.normalization_state_digest,
    }
    values.update(changes)
    output = CanonicalDenoisingOutput(**values)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        validate_model_output(model_input, output)
