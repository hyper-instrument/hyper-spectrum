"""Quantitative XAS denoising metrics and compound-grouped aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from .arrays import XASSpectrum
from .baselines import BaselineFailure, BaselineResult

METRIC_KEY: Literal["normalized_spectrum_rmse"] = "normalized_spectrum_rmse"
METRIC_DIRECTION: Literal["min"] = "min"
DEFAULT_BOOTSTRAP_SEED = 0


def normalized_spectrum_rmse(predicted: XASSpectrum, true: XASSpectrum) -> float:
    """Return RMSE divided by the target spectrum's dynamic range."""

    if predicted.sample_id != true.sample_id or predicted.group_id != true.group_id:
        raise ValueError("prediction and target sample/group IDs must match")
    if predicted.energy.shape != true.energy.shape or not np.array_equal(
        predicted.energy, true.energy
    ):
        raise ValueError("prediction and target energy axes must match exactly")
    if not np.isfinite(predicted.intensity).all() or not np.isfinite(true.intensity).all():
        raise ValueError("prediction and target arrays must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        dynamic_range = float(np.max(true.intensity) - np.min(true.intensity))
    if not np.isfinite(dynamic_range):
        raise ValueError("target must have a finite target dynamic range")
    if dynamic_range <= 1e-12:
        raise ValueError("target dynamic range must be greater than 1e-12")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = predicted.intensity - true.intensity
    if not np.isfinite(residual).all():
        raise ValueError("prediction and target must produce a finite residual")

    residual_scale = float(np.max(np.abs(residual)))
    if not np.isfinite(residual_scale):
        raise ValueError("residual scale must be finite")
    if residual_scale == 0.0:
        rmse = 0.0
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scaled_residual = residual / residual_scale
            scaled_mean_square = float(np.mean(scaled_residual * scaled_residual))
            rmse = float(residual_scale * np.sqrt(scaled_mean_square))
        if not np.isfinite(scaled_residual).all() or not np.isfinite(scaled_mean_square):
            raise ValueError("scaled residual statistics must be finite")
    if not np.isfinite(rmse):
        raise ValueError("RMSE must be finite")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        score = float(rmse / dynamic_range)
    if not np.isfinite(score):
        raise ValueError("normalized spectrum RMSE must be finite")
    return score


@dataclass(frozen=True, slots=True)
class SampleMetric:
    """One valid per-sample metric value."""

    sample_id: str
    group_id: str
    value: float

    def __post_init__(self) -> None:
        if not self.sample_id or not self.group_id:
            raise ValueError("metric sample_id and group_id must be non-empty")
        if not np.isfinite(self.value):
            raise ValueError("metric value must be finite")


@dataclass(frozen=True, slots=True)
class MetricFailure:
    """One accounted sample that could not receive a metric."""

    sample_id: str
    group_id: str
    error_type: str
    message: str


MetricRecord: TypeAlias = SampleMetric | MetricFailure


@dataclass(frozen=True, slots=True)
class GroupedMetricAggregate:
    """Primary score plus deterministic grouped-bootstrap evidence and coverage."""

    key: Literal["normalized_spectrum_rmse"]
    direction: Literal["min"]
    value: float
    group_means: dict[str, float]
    bootstrap_values: tuple[float, ...]
    confidence_interval: tuple[float, float]
    sample_count: int
    group_count: int
    failure_count: int
    total_count: int
    failures: tuple[MetricFailure, ...]
    bootstrap_seed: int


def score_normalized_spectrum_rmse(
    predictions: Sequence[BaselineResult], truths: Sequence[XASSpectrum]
) -> tuple[MetricRecord, ...]:
    """Score outputs by sample ID while explicitly accounting for every failure."""

    truth_by_id: dict[str, XASSpectrum] = {}
    for truth in truths:
        if truth.sample_id in truth_by_id:
            raise ValueError(f"duplicate target sample_id: {truth.sample_id}")
        truth_by_id[truth.sample_id] = truth

    records: list[MetricRecord] = []
    seen_predictions: set[str] = set()
    for result in predictions:
        sample_id = result.sample_id if isinstance(result, BaselineFailure) else result.spectrum.sample_id
        group_id = result.group_id if isinstance(result, BaselineFailure) else result.spectrum.group_id
        if sample_id in seen_predictions:
            raise ValueError(f"duplicate prediction sample_id: {sample_id}")
        seen_predictions.add(sample_id)
        if isinstance(result, BaselineFailure):
            records.append(MetricFailure(sample_id, group_id, result.error_type, result.message))
            continue
        target = truth_by_id.get(sample_id)
        if target is None:
            records.append(
                MetricFailure(sample_id, group_id, "missing_target", "no target exists for sample")
            )
            continue
        try:
            value = normalized_spectrum_rmse(result.spectrum, target)
        except ValueError as error:
            records.append(MetricFailure(sample_id, group_id, "invalid_metric", str(error)))
        else:
            records.append(SampleMetric(sample_id, group_id, value))

    for truth in truths:
        if truth.sample_id not in seen_predictions:
            records.append(
                MetricFailure(
                    truth.sample_id,
                    truth.group_id,
                    "missing_prediction",
                    "no prediction or typed baseline failure exists for sample",
                )
            )
    return tuple(records)


def aggregate_grouped_metrics(
    records: Sequence[MetricRecord],
    *,
    bootstrap_iterations: int = 1000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> GroupedMetricAggregate:
    """Mean within compounds, then unweighted mean and bootstrap across compounds."""

    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("metric records must contain unique sample IDs")

    failures = tuple(record for record in records if isinstance(record, MetricFailure))
    successful = tuple(record for record in records if isinstance(record, SampleMetric))
    if not successful:
        raise ValueError("at least one successful sample metric is required")
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for record in successful:
        grouped[record.group_id].append(record.value)
    group_means = {
        group_id: float(np.mean(values)) for group_id, values in sorted(grouped.items())
    }
    means = np.array(tuple(group_means.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(means), size=(bootstrap_iterations, len(means)))
    bootstrap_values_array = np.mean(means[indices], axis=1)
    tail = (1.0 - confidence_level) / 2.0
    confidence_interval_array = np.quantile(
        bootstrap_values_array, np.array([tail, 1.0 - tail])
    )
    return GroupedMetricAggregate(
        key=METRIC_KEY,
        direction=METRIC_DIRECTION,
        value=float(np.mean(means)),
        group_means=group_means,
        bootstrap_values=tuple(float(value) for value in bootstrap_values_array),
        confidence_interval=(
            float(confidence_interval_array[0]),
            float(confidence_interval_array[1]),
        ),
        sample_count=len(successful),
        group_count=len(group_means),
        failure_count=len(failures),
        total_count=len(records),
        failures=failures,
        bootstrap_seed=seed,
    )
