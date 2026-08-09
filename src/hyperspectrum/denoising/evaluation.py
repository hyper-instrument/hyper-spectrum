"""Cross-modality denoising evaluation in normalized and native signal spaces."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .model import (
    CanonicalDenoisingInput,
    CompatibilityIssue,
    DenoisingModel,
    validate_model_output,
)
from .normalization import (
    NormalizationState,
    NormalizedSpectrum,
    denormalize,
    normalize,
    recommended_normalization,
)
from .sample import SpectrumSample
from .split import SplitAssignment


@dataclass(frozen=True, slots=True)
class DenoisingPair:
    """One noisy input and clean/proxy target on the exact same physical grid."""

    noisy: SpectrumSample
    clean: SpectrumSample
    assignment: SplitAssignment

    def __post_init__(self) -> None:
        if (
            self.noisy.sample_id != self.clean.sample_id
            or self.noisy.group_id != self.clean.group_id
        ):
            raise ValueError("noisy and clean sample identity must match")
        if (
            self.assignment.sample_id != self.noisy.sample_id
            or self.assignment.group_id != self.noisy.group_id
        ):
            raise ValueError("split assignment must match noisy sample identity")
        if self.noisy.modality != self.clean.modality:
            raise ValueError("noisy and clean modality must match")
        if self.noisy.representation != self.clean.representation:
            raise ValueError("noisy and clean representation must match")
        if self.noisy.signal_unit != self.clean.signal_unit:
            raise ValueError("noisy and clean signal unit must match")
        if self.noisy.signal.shape != self.clean.signal.shape:
            raise ValueError("noisy and clean signal shape must match")
        if self.noisy.channel_labels != self.clean.channel_labels:
            raise ValueError("noisy and clean channel labels must match")
        if len(self.noisy.axes) != len(self.clean.axes):
            raise ValueError("noisy and clean axes must match")
        for noisy_axis, clean_axis in zip(
            self.noisy.axes, self.clean.axes, strict=True
        ):
            if (
                noisy_axis.name != clean_axis.name
                or noisy_axis.unit != clean_axis.unit
                or noisy_axis.direction != clean_axis.direction
                or not np.array_equal(noisy_axis.values, clean_axis.values)
            ):
                raise ValueError("noisy and clean axes must match exactly")
        if not np.array_equal(self.noisy.valid_mask, self.clean.valid_mask):
            raise ValueError("noisy and clean valid masks must match exactly")

    @property
    def split(self) -> Literal["train", "val", "test"]:
        """Return digest-bound split membership for compatibility with callers."""
        return self.assignment.split


@dataclass(frozen=True, slots=True)
class MetricValues:
    """RMSE and MAE measured in one explicit signal unit."""

    rmse: float
    mae: float
    unit: str


@dataclass(frozen=True, slots=True)
class SampleEvaluation:
    """One accounted sample: evaluated, compatibility-skipped, or failed."""

    sample_id: str
    group_id: str
    modality: str
    status: Literal["evaluated", "skipped", "failed"]
    normalized_metrics: MetricValues | None
    native_metrics: MetricValues | None
    issue_codes: tuple[str, ...]
    message: str | None


@dataclass(frozen=True, slots=True)
class ModalityEvaluation:
    """Per-modality metrics and coverage without cross-modality sample weighting."""

    modality: str
    sample_count: int
    evaluated_count: int
    skipped_count: int
    failed_count: int
    coverage: float
    normalized_rmse: float | None
    normalized_mae: float | None
    native_rmse: float | None
    native_mae: float | None
    native_unit: str | None


@dataclass(frozen=True, slots=True)
class DenoisingSuiteReport:
    """Per-sample evidence, per-modality means, macro means, and honest coverage."""

    sample_results: tuple[SampleEvaluation, ...]
    modality_results: tuple[ModalityEvaluation, ...]
    evaluated_count: int
    skipped_count: int
    failed_count: int
    coverage: float
    macro_normalized_rmse: float | None
    macro_normalized_mae: float | None
    native_macro_aggregation: Literal["not_applicable_across_physical_units"]

    def for_modality(self, modality: str) -> ModalityEvaluation:
        for result in self.modality_results:
            if result.modality == modality:
                return result
        raise KeyError(modality)

    def to_dict(self) -> dict[str, Any]:
        """Return an agent/ACE-friendly JSON payload without inventing skipped scores."""

        def metric_payload(metric: MetricValues | None) -> dict[str, object] | None:
            if metric is None:
                return None
            return {"rmse": metric.rmse, "mae": metric.mae, "unit": metric.unit}

        return {
            "schema_version": "hyperspectrum-denoising-suite/v1",
            "coverage": {
                "value": self.coverage,
                "evaluated_count": self.evaluated_count,
                "skipped_count": self.skipped_count,
                "failed_count": self.failed_count,
                "total_count": len(self.sample_results),
            },
            "macro": {
                "normalized_rmse": self.macro_normalized_rmse,
                "normalized_mae": self.macro_normalized_mae,
                "native": self.native_macro_aggregation,
            },
            "per_modality": [
                {
                    "modality": result.modality,
                    "sample_count": result.sample_count,
                    "evaluated_count": result.evaluated_count,
                    "skipped_count": result.skipped_count,
                    "failed_count": result.failed_count,
                    "coverage": result.coverage,
                    "normalized_rmse": result.normalized_rmse,
                    "normalized_mae": result.normalized_mae,
                    "native_rmse": result.native_rmse,
                    "native_mae": result.native_mae,
                    "native_unit": result.native_unit,
                }
                for result in self.modality_results
            ],
            "samples": [
                {
                    "sample_id": result.sample_id,
                    "group_id": result.group_id,
                    "modality": result.modality,
                    "status": result.status,
                    "normalized_metrics": metric_payload(result.normalized_metrics),
                    "native_metrics": metric_payload(result.native_metrics),
                    "issue_codes": list(result.issue_codes),
                    "message": result.message,
                }
                for result in self.sample_results
            ],
        }


def _metrics(predicted: SpectrumSample, target: SpectrumSample) -> MetricValues:
    if predicted.signal.shape != target.signal.shape or not np.array_equal(
        predicted.valid_mask, target.valid_mask
    ):
        raise ValueError("metric arrays and masks must match exactly")
    residual = predicted.signal[predicted.valid_mask] - target.signal[target.valid_mask]
    magnitudes = np.abs(residual)
    if not np.isfinite(magnitudes).all():
        raise ValueError("metric residual must be finite")
    scale = float(np.max(magnitudes)) if len(magnitudes) else 0.0
    if scale == 0.0:
        rmse = 0.0
        mae = 0.0
    else:
        scaled = magnitudes / scale
        rmse = float(scale * np.sqrt(np.mean(scaled * scaled)))
        mae = float(scale * np.mean(scaled))
    if not np.isfinite(rmse) or not np.isfinite(mae):
        raise ValueError("metric values must be finite")
    return MetricValues(rmse=rmse, mae=mae, unit=predicted.signal_unit)


def _failed(pair: DenoisingPair, code: str, message: str) -> SampleEvaluation:
    return SampleEvaluation(
        sample_id=pair.noisy.sample_id,
        group_id=pair.noisy.group_id,
        modality=pair.noisy.modality,
        status="failed",
        normalized_metrics=None,
        native_metrics=None,
        issue_codes=(code,),
        message=message,
    )


def _skipped(
    pair: DenoisingPair, compatibility: Sequence[CompatibilityIssue]
) -> SampleEvaluation:
    issues = tuple(compatibility)
    return SampleEvaluation(
        sample_id=pair.noisy.sample_id,
        group_id=pair.noisy.group_id,
        modality=pair.noisy.modality,
        status="skipped",
        normalized_metrics=None,
        native_metrics=None,
        issue_codes=tuple(issue.code for issue in issues),
        message="; ".join(issue.message for issue in issues),
    )


def _evaluate_pair(
    model: DenoisingModel,
    pair: DenoisingPair,
    *,
    method: str,
    state: NormalizationState | None,
) -> SampleEvaluation:
    if state is not None and state.fit_scope == "dataset_fitted":
        if state.split_manifest_digest != pair.assignment.manifest_digest:
            return _failed(
                pair,
                "split_manifest_mismatch",
                "dataset-fitted normalization and evaluation pair use different split manifests",
            )
        if pair.split != "train" and (
            pair.noisy.sample_id in state.training_sample_ids
            or pair.noisy.group_id in state.training_group_ids
        ):
            return _failed(
                pair,
                "split_leakage",
                "held-out sample/group overlaps the normalization training partition",
            )
    compatibility = model.capabilities.check_sample(
        pair.noisy,
        normalization_method=method,
        require_native_unit_recovery=True,
    )
    if compatibility:
        return _skipped(pair, compatibility)
    try:
        normalized_noisy = normalize(pair.noisy, method, state=state)
        normalized_clean = normalize(pair.clean, method, state=normalized_noisy.state)
    except ValueError as error:
        return _failed(pair, "normalization_error", str(error))

    model_input = CanonicalDenoisingInput.from_normalized(normalized_noisy)
    try:
        output = model.predict(model_input)
        validate_model_output(model_input, output)
    except Exception as error:  # noqa: BLE001 - model adapters are an isolation boundary.
        return _failed(pair, "invalid_model_output", str(error))

    try:
        normalized_prediction = NormalizedSpectrum(
            normalized_noisy.sample.with_signal(output.signal),
            normalized_noisy.state,
        )
        normalized_metrics = _metrics(
            normalized_prediction.sample, normalized_clean.sample
        )
        native_prediction = denormalize(normalized_prediction)
        native_metrics = _metrics(native_prediction, pair.clean)
    except ValueError as error:
        return _failed(pair, "metric_error", str(error))
    return SampleEvaluation(
        sample_id=pair.noisy.sample_id,
        group_id=pair.noisy.group_id,
        modality=pair.noisy.modality,
        status="evaluated",
        normalized_metrics=normalized_metrics,
        native_metrics=native_metrics,
        issue_codes=(),
        message=None,
    )


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _aggregate_modality(
    modality: str, results: tuple[SampleEvaluation, ...]
) -> ModalityEvaluation:
    evaluated = tuple(result for result in results if result.status == "evaluated")
    normalized = tuple(
        result.normalized_metrics
        for result in evaluated
        if result.normalized_metrics is not None
    )
    native = tuple(
        result.native_metrics
        for result in evaluated
        if result.native_metrics is not None
    )
    native_units = {metric.unit for metric in native}
    if len(native_units) > 1:
        raise ValueError(f"native metric units drifted within modality {modality}")
    return ModalityEvaluation(
        modality=modality,
        sample_count=len(results),
        evaluated_count=len(evaluated),
        skipped_count=sum(result.status == "skipped" for result in results),
        failed_count=sum(result.status == "failed" for result in results),
        coverage=len(evaluated) / len(results),
        normalized_rmse=_mean(tuple(metric.rmse for metric in normalized)),
        normalized_mae=_mean(tuple(metric.mae for metric in normalized)),
        native_rmse=_mean(tuple(metric.rmse for metric in native)),
        native_mae=_mean(tuple(metric.mae for metric in native)),
        native_unit=next(iter(native_units)) if native_units else None,
    )


def evaluate_denoising_suite(
    model: DenoisingModel,
    pairs: Sequence[DenoisingPair],
    *,
    normalization_methods: Mapping[str, str] | None = None,
    normalization_states: Mapping[str, NormalizationState] | None = None,
) -> DenoisingSuiteReport:
    """Evaluate one model across modalities with model-external adapters and metrics."""
    if not pairs:
        raise ValueError("a denoising suite requires at least one pair")
    keys = tuple((pair.noisy.modality, pair.noisy.sample_id) for pair in pairs)
    if len(keys) != len(set(keys)):
        raise ValueError("suite sample IDs must be unique within each modality")

    results: list[SampleEvaluation] = []
    for pair in pairs:
        try:
            method = (
                normalization_methods[pair.noisy.modality]
                if normalization_methods is not None
                and pair.noisy.modality in normalization_methods
                else recommended_normalization(
                    pair.noisy.modality, pair.noisy.representation
                )
            )
        except ValueError as error:
            results.append(_failed(pair, "normalization_error", str(error)))
            continue
        state = (
            normalization_states.get(pair.noisy.modality)
            if normalization_states is not None
            else None
        )
        results.append(_evaluate_pair(model, pair, method=method, state=state))

    grouped: defaultdict[str, list[SampleEvaluation]] = defaultdict(list)
    for result in results:
        grouped[result.modality].append(result)
    modality_results = tuple(
        _aggregate_modality(modality, tuple(grouped[modality]))
        for modality in sorted(grouped)
    )
    evaluated_count = sum(result.status == "evaluated" for result in results)
    skipped_count = sum(result.status == "skipped" for result in results)
    failed_count = sum(result.status == "failed" for result in results)
    normalized_rmse = tuple(
        result.normalized_rmse
        for result in modality_results
        if result.normalized_rmse is not None
    )
    normalized_mae = tuple(
        result.normalized_mae
        for result in modality_results
        if result.normalized_mae is not None
    )
    return DenoisingSuiteReport(
        sample_results=tuple(results),
        modality_results=modality_results,
        evaluated_count=evaluated_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        coverage=evaluated_count / len(results),
        macro_normalized_rmse=_mean(normalized_rmse),
        macro_normalized_mae=_mean(normalized_mae),
        native_macro_aggregation="not_applicable_across_physical_units",
    )
