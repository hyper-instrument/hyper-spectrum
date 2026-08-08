from __future__ import annotations

import math

import numpy as np
import pytest

from hyperspectrum.plugins.xas.arrays import XASSpectrum
from hyperspectrum.plugins.xas.baselines import BaselineFailure, BaselinePrediction
from hyperspectrum.plugins.xas.metrics import (
    METRIC_DIRECTION,
    METRIC_KEY,
    MetricFailure,
    SampleMetric,
    aggregate_grouped_metrics,
    normalized_spectrum_rmse,
    score_normalized_spectrum_rmse,
)


def spectrum(
    sample_id: str,
    group_id: str,
    intensity: list[float],
    *,
    energy: list[float] | None = None,
) -> XASSpectrum:
    return XASSpectrum(
        sample_id=sample_id,
        group_id=group_id,
        energy=np.array(energy or [1.0, 2.0, 3.0]),
        intensity=np.array(intensity),
        energy_unit="eV",
    )


def test_normalized_spectrum_rmse_uses_target_dynamic_range() -> None:
    # Break caught: normalization could use prediction range or divide before taking RMSE.
    true = spectrum("s1", "A", [0.0, 1.0, 2.0])
    predicted = spectrum("s1", "A", [0.0, 2.0, 2.0])

    assert normalized_spectrum_rmse(predicted, true) == pytest.approx(math.sqrt(1.0 / 3.0) / 2.0)
    assert (METRIC_KEY, METRIC_DIRECTION) == ("normalized_spectrum_rmse", "min")


@pytest.mark.parametrize(
    ("predicted", "true", "message"),
    [
        (
            spectrum("s1", "A", [0.0, 1.0, 2.0], energy=[1.0, 2.0, 3.0]),
            spectrum("s1", "A", [0.0, 1.0, 2.0], energy=[1.0, 2.5, 3.0]),
            "energy axes",
        ),
        (spectrum("s1", "A", [1.0, 1.0, 1.0]), spectrum("s1", "A", [1.0, 1.0, 1.0]), "dynamic range"),
    ],
)
def test_normalized_spectrum_rmse_rejects_axis_mismatch_or_degenerate_target(
    predicted: XASSpectrum, true: XASSpectrum, message: str
) -> None:
    # Break caught: invalid pairs could receive plausible scores through alignment or epsilon hiding.
    with pytest.raises(ValueError, match=message):
        normalized_spectrum_rmse(predicted, true)


def test_normalized_spectrum_rmse_rejects_nonfinite_derived_target_range() -> None:
    # Break caught: finite target values could overflow to an infinite normalization range.
    true = spectrum("s1", "A", [-1e308, 0.0, 1e308])
    predicted = spectrum("s1", "A", [-1e308, 0.0, 1e308])

    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="finite target dynamic range"),
    ):
        normalized_spectrum_rmse(predicted, true)


def test_normalized_spectrum_rmse_rejects_nonfinite_derived_residual() -> None:
    # Break caught: subtraction of finite prediction and target values could overflow silently.
    true = spectrum("s1", "A", [-1e308, -1e308 + 1e292, -1e308 + 2e292])
    predicted = spectrum("s1", "A", [1e308, 1e308, 1e308])

    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="finite residual"),
    ):
        normalized_spectrum_rmse(predicted, true)


def test_normalized_spectrum_rmse_stably_scores_large_finite_residuals() -> None:
    # Break caught: naive residual squaring could turn a valid large score into infinity.
    true = spectrum("s1", "A", [0.0, 1.0, 2.0])
    predicted = spectrum("s1", "A", [1e200, 1e200, 1e200])

    with np.errstate(over="raise", invalid="raise"):
        score = normalized_spectrum_rmse(predicted, true)

    assert np.isfinite(score)
    assert score == pytest.approx(5e199)


def test_group_aggregate_means_samples_then_weights_compound_groups_equally() -> None:
    # Break caught: a group with more spectra could receive more weight in the primary score.
    records = (
        SampleMetric("a1", "A", 1.0),
        SampleMetric("a2", "A", 3.0),
        SampleMetric("b1", "B", 10.0),
        SampleMetric("c1", "C", 4.0),
        SampleMetric("c2", "C", 6.0),
        SampleMetric("c3", "C", 8.0),
    )

    aggregate = aggregate_grouped_metrics(records, bootstrap_iterations=8, seed=7)

    assert aggregate.value == pytest.approx(6.0)
    assert aggregate.group_means == {"A": 2.0, "B": 10.0, "C": 6.0}
    assert (aggregate.sample_count, aggregate.group_count, aggregate.failure_count) == (6, 3, 0)


def test_grouped_bootstrap_is_seeded_and_resamples_groups_not_spectra() -> None:
    # Break caught: bootstrap could be nondeterministic or sample spectra across group boundaries.
    records = (
        SampleMetric("a1", "A", 1.0),
        SampleMetric("a2", "A", 3.0),
        SampleMetric("b1", "B", 10.0),
        SampleMetric("c1", "C", 4.0),
        SampleMetric("c2", "C", 6.0),
        SampleMetric("c3", "C", 8.0),
    )

    first = aggregate_grouped_metrics(records, bootstrap_iterations=8, seed=7)
    second = aggregate_grouped_metrics(records, bootstrap_iterations=8, seed=7)

    assert first.bootstrap_values == pytest.approx(
        (22 / 3, 22 / 3, 10 / 3, 10 / 3, 6.0, 14 / 3, 6.0, 14 / 3)
    )
    assert first.confidence_interval == pytest.approx((10 / 3, 22 / 3))
    assert second == first


def test_aggregate_reports_failures_separately_without_silent_drop() -> None:
    # Break caught: failed samples could disappear while the reported count still implies full coverage.
    records = (
        SampleMetric("a1", "A", 2.0),
        MetricFailure("a2", "A", "invalid_metric", "axis mismatch"),
        SampleMetric("b1", "B", 4.0),
    )

    aggregate = aggregate_grouped_metrics(records, bootstrap_iterations=4, seed=0)

    assert aggregate.value == pytest.approx(3.0)
    assert (aggregate.sample_count, aggregate.failure_count, aggregate.total_count) == (2, 1, 3)
    assert aggregate.failures[0].sample_id == "a2"


def test_scoring_turns_every_prediction_or_baseline_failure_into_one_metric_record() -> None:
    # Break caught: baseline failures could be omitted before aggregation.
    truth = (spectrum("s1", "A", [0.0, 1.0, 2.0]), spectrum("s2", "B", [0.0, 1.0, 2.0]))
    prediction = BaselinePrediction(
        spectrum=spectrum("s1", "A", [0.0, 1.0, 2.0]), method="identity", optimization_kind="no_training"
    )
    failure = BaselineFailure(
        sample_id="s2",
        group_id="B",
        energy=np.array([1.0, 2.0, 3.0]),
        energy_unit="eV",
        method="identity",
        error_type="unsupported_setting",
        message="deliberate failure",
    )

    records = score_normalized_spectrum_rmse((prediction, failure), truth)

    assert len(records) == 2
    assert isinstance(records[0], SampleMetric)
    assert isinstance(records[1], MetricFailure)
    assert records[1].sample_id == "s2"
