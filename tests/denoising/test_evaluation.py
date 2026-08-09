from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from hyperspectrum.denoising import SpectrumAxis, SpectrumSample
from hyperspectrum.denoising.evaluation import DenoisingPair, evaluate_denoising_suite
from hyperspectrum.denoising.model import (
    CanonicalDenoisingInput,
    CanonicalDenoisingOutput,
    ModelCapabilities,
)
from hyperspectrum.denoising.normalization import fit_normalizer
from hyperspectrum.denoising.split import SplitEntry, SplitManifest


def sample(
    sample_id: str,
    modality: str,
    signal: np.ndarray | list[float] | list[complex],
    *,
    representation: str = "dense",
    unit: str = "mu(E)",
) -> SpectrumSample:
    array = np.asarray(signal)
    axes = tuple(
        SpectrumAxis(
            name=f"axis-{index}",
            unit="eV" if modality in {"xas", "xanes", "exafs", "eels"} else "arb-axis",
            direction="increasing",
            values=np.arange(size, dtype=float),
        )
        for index, size in enumerate(array.shape)
    )
    return SpectrumSample(
        sample_id=sample_id,
        group_id=f"group-{sample_id}",
        modality=modality,  # type: ignore[arg-type]
        representation=representation,  # type: ignore[arg-type]
        axes=axes,
        signal=array,
        valid_mask=np.ones(array.shape, dtype=bool),
        signal_unit=unit,
        metadata={},
        provenance={},
    )


def pair(
    sample_id: str,
    modality: str,
    noisy: np.ndarray | list[float] | list[complex],
    clean: np.ndarray | list[float] | list[complex],
    *,
    representation: str = "dense",
    unit: str = "mu(E)",
) -> DenoisingPair:
    noisy_sample = sample(
        sample_id, modality, noisy, representation=representation, unit=unit
    )
    clean_sample = sample(
        sample_id, modality, clean, representation=representation, unit=unit
    )
    split_manifest = SplitManifest(
        schema_version="hyperspectrum-split-manifest/v1",
        entries=(SplitEntry(sample_id, noisy_sample.group_id, "test"),),
    )
    return DenoisingPair(
        noisy=noisy_sample,
        clean=clean_sample,
        assignment=split_manifest.assignment_for(sample_id, noisy_sample.group_id),
    )


class IdentityModel:
    capabilities = ModelCapabilities(
        schema_version="hyperspectrum-denoising-capabilities/v1",
        axis_ranks=(1,),
        representations=("dense",),
        channel_counts=(1,),
        required_normalization="per_spectrum_range",
        native_unit_recovery=True,
    )

    def predict(self, model_input: CanonicalDenoisingInput) -> CanonicalDenoisingOutput:
        return CanonicalDenoisingOutput(
            sample_id=model_input.sample_id,
            signal=model_input.signal,
            valid_mask=model_input.valid_mask,
            normalization_state_digest=model_input.normalization_state_digest,
        )


def test_one_model_runs_xas_and_raman_and_macro_weights_modalities_equally() -> None:
    # Break caught: reuse could be fake modality-specific dispatch, or macro could weight sample-rich XAS more.
    report = evaluate_denoising_suite(
        IdentityModel(),
        (
            pair("xas-perfect", "xas", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
            pair("xas-noisy", "xas", [0.0, 2.0, 2.0], [0.0, 1.0, 2.0]),
            pair(
                "raman-noisy",
                "raman",
                [0.0, 20.0, 20.0],
                [0.0, 10.0, 20.0],
                unit="count",
            ),
        ),
    )

    xas = report.for_modality("xas")
    raman = report.for_modality("raman")
    one_error = math.sqrt(0.25 / 3.0)
    assert xas.normalized_rmse == pytest.approx(one_error / 2.0)
    assert raman.normalized_rmse == pytest.approx(one_error)
    assert report.macro_normalized_rmse == pytest.approx(
        (one_error / 2.0 + one_error) / 2.0
    )
    assert report.coverage == pytest.approx(1.0)
    assert (xas.evaluated_count, raman.evaluated_count) == (2, 1)
    assert report.native_macro_aggregation == "not_applicable_across_physical_units"


def test_dual_space_metrics_restore_native_units_before_scoring() -> None:
    # Break caught: a metric labelled physical could actually be computed on normalized values.
    report = evaluate_denoising_suite(
        IdentityModel(),
        (pair("xas-noisy", "xas", [0.0, 2.0, 2.0], [0.0, 1.0, 2.0]),),
    )
    result = report.sample_results[0]

    assert result.status == "evaluated"
    assert result.normalized_metrics is not None
    assert result.native_metrics is not None
    assert result.normalized_metrics.rmse == pytest.approx(math.sqrt(0.25 / 3.0))
    assert result.normalized_metrics.mae == pytest.approx(1.0 / 6.0)
    assert result.normalized_metrics.unit == "1"
    assert result.native_metrics.rmse == pytest.approx(math.sqrt(1.0 / 3.0))
    assert result.native_metrics.mae == pytest.approx(1.0 / 3.0)
    assert result.native_metrics.unit == "mu(E)"


def test_incompatible_complex_nmr_and_2d_eels_are_skipped_not_zero_scored() -> None:
    # Break caught: unsupported modalities could be flattened or enter a leaderboard as zero errors.
    report = evaluate_denoising_suite(
        IdentityModel(),
        (
            pair("xas-ok", "xas", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
            pair(
                "nmr-complex",
                "nmr",
                [1.0 + 1.0j, 2.0 + 1.0j, 3.0 + 1.0j],
                [1.0 + 1.0j, 2.0 + 1.0j, 3.0 + 1.0j],
                representation="complex",
                unit="V",
            ),
            pair(
                "eels-2d",
                "eels",
                np.ones((2, 3)),
                np.ones((2, 3)),
                unit="count",
            ),
        ),
    )

    statuses = {result.sample_id: result for result in report.sample_results}
    assert statuses["nmr-complex"].status == "skipped"
    assert set(statuses["nmr-complex"].issue_codes) == {
        "representation",
        "normalization",
    }
    assert statuses["eels-2d"].status == "skipped"
    assert statuses["eels-2d"].issue_codes == ("axis_rank",)
    assert report.for_modality("nmr").normalized_rmse is None
    assert report.for_modality("eels").normalized_rmse is None
    assert report.macro_normalized_rmse == pytest.approx(0.0)
    assert report.coverage == pytest.approx(1.0 / 3.0)
    assert (report.evaluated_count, report.skipped_count, report.failed_count) == (
        1,
        2,
        0,
    )
    payload = report.to_dict()
    assert payload["schema_version"] == "hyperspectrum-denoising-suite/v1"
    assert payload["macro"]["normalized_rmse"] == pytest.approx(0.0)
    assert payload["macro"]["native"] == "not_applicable_across_physical_units"
    nmr_payload = next(
        item for item in payload["per_modality"] if item["modality"] == "nmr"
    )
    assert nmr_payload["normalized_rmse"] is None
    assert nmr_payload["skipped_count"] == 1


class WrongIdentityModel(IdentityModel):
    def predict(self, model_input: CanonicalDenoisingInput) -> CanonicalDenoisingOutput:
        return CanonicalDenoisingOutput(
            sample_id="wrong-sample",
            signal=model_input.signal,
            valid_mask=model_input.valid_mask,
            normalization_state_digest=model_input.normalization_state_digest,
        )


def test_malformed_model_output_is_accounted_as_failure() -> None:
    # Break caught: invalid model outputs could disappear from coverage and metric denominators.
    report = evaluate_denoising_suite(
        WrongIdentityModel(),
        (pair("xas-1", "xas", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),),
    )

    result = report.sample_results[0]
    assert result.status == "failed"
    assert result.issue_codes == ("invalid_model_output",)
    assert (report.evaluated_count, report.skipped_count, report.failed_count) == (
        0,
        0,
        1,
    )
    assert report.coverage == 0.0
    assert report.macro_normalized_rmse is None


def test_pair_contract_rejects_axis_unit_mask_or_identity_mismatch() -> None:
    noisy = sample("same", "xas", np.array([1.0, 2.0, 3.0]))
    wrong_id = sample("different", "xas", np.array([1.0, 2.0, 3.0]))
    manifest = SplitManifest(
        schema_version="hyperspectrum-split-manifest/v1",
        entries=(SplitEntry("same", noisy.group_id, "test"),),
    )
    assignment = manifest.assignment_for("same", noisy.group_id)
    with pytest.raises(ValueError, match="sample identity"):
        DenoisingPair(noisy=noisy, clean=wrong_id, assignment=assignment)

    wrong_unit = sample("same", "xas", np.array([1.0, 2.0, 3.0]), unit="absorbance")
    with pytest.raises(ValueError, match="signal unit"):
        DenoisingPair(noisy=noisy, clean=wrong_unit, assignment=assignment)


def test_dataset_fitted_state_must_share_manifest_and_exclude_held_out_groups() -> None:
    # Break caught: an evaluator could accept a state fitted on held-out identities.
    train_noisy = sample("train-1", "xas", np.array([0.0, 1.0, 2.0]))
    test_pair = pair("test-1", "xas", [0.0, 2.0, 2.0], [0.0, 1.0, 2.0])
    shared_manifest = SplitManifest(
        schema_version="hyperspectrum-split-manifest/v1",
        entries=(
            SplitEntry("train-1", train_noisy.group_id, "train"),
            SplitEntry("test-1", test_pair.noisy.group_id, "test"),
        ),
    )
    state = fit_normalizer(
        "train_global_standard", (train_noisy,), manifest=shared_manifest
    )
    bound_pair = DenoisingPair(
        noisy=test_pair.noisy,
        clean=test_pair.clean,
        assignment=shared_manifest.assignment_for(
            test_pair.noisy.sample_id, test_pair.noisy.group_id
        ),
    )

    report = evaluate_denoising_suite(
        IdentityModel(),
        (bound_pair,),
        normalization_methods={"xas": "train_global_standard"},
        normalization_states={"xas": state},
    )
    assert report.sample_results[0].status == "skipped"
    assert report.sample_results[0].issue_codes == ("normalization",)

    valid = evaluate_denoising_suite(
        ModelWithGlobalNormalization(),
        (bound_pair,),
        normalization_methods={"xas": "train_global_standard"},
        normalization_states={"xas": state},
    )
    assert valid.sample_results[0].status == "evaluated"

    leaked_state = replace(state, training_group_ids=(test_pair.noisy.group_id,))
    leaked = evaluate_denoising_suite(
        ModelWithGlobalNormalization(),
        (bound_pair,),
        normalization_methods={"xas": "train_global_standard"},
        normalization_states={"xas": leaked_state},
    )
    assert leaked.sample_results[0].status == "failed"
    assert leaked.sample_results[0].issue_codes == ("split_leakage",)

    mismatched = evaluate_denoising_suite(
        ModelWithGlobalNormalization(),
        (test_pair,),
        normalization_methods={"xas": "train_global_standard"},
        normalization_states={"xas": state},
    )
    assert mismatched.sample_results[0].status == "failed"
    assert mismatched.sample_results[0].issue_codes == ("split_manifest_mismatch",)


class ModelWithGlobalNormalization(IdentityModel):
    capabilities = ModelCapabilities(
        schema_version="hyperspectrum-denoising-capabilities/v1",
        axis_ranks=(1,),
        representations=("dense",),
        channel_counts=(1,),
        required_normalization="train_global_standard",
        native_unit_recovery=True,
    )
