"""Leakage-safe, modality-aware normalization for spectral denoising."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import numpy as np

from hyperspectrum.contracts.json import (
    FrozenJsonMapping,
    freeze_json_mapping,
    thaw_json_mapping,
)

from .sample import (
    SpectralModality,
    SpectrumRepresentation,
    SpectrumSample,
    samples_digest,
)
from .split import SplitManifest

FitScope: TypeAlias = Literal["per_sample", "dataset_fitted"]
Invertibility: TypeAlias = Literal["exact", "approximate", "none"]
Split: TypeAlias = Literal["train", "val", "test"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATE_PARAMETER_KEYS = {
    "identity_raw": ("offset", "scale"),
    "per_spectrum_range": ("offset", "scale"),
    "total_ion_current": ("offset", "scale"),
    "complex_rms": ("offset", "scale"),
    "train_global_standard": ("mean", "scale"),
}
_METHOD_FIT_SCOPES: dict[str, FitScope] = {
    "identity_raw": "per_sample",
    "per_spectrum_range": "per_sample",
    "total_ion_current": "per_sample",
    "complex_rms": "per_sample",
    "train_global_standard": "dataset_fitted",
}


@dataclass(frozen=True, slots=True)
class NormalizationDefinition:
    """A registry entry describing compatibility and fitted-state semantics."""

    name: str
    fit_scope: FitScope
    invertibility: Invertibility
    modalities: frozenset[str]
    representations: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "modalities", frozenset(self.modalities))
        object.__setattr__(self, "representations", frozenset(self.representations))
        if not self.name.strip():
            raise ValueError("normalization name must be non-empty")
        if not self.modalities or not self.representations:
            raise ValueError("normalization compatibility must be non-empty")

    def require_compatible(self, sample: SpectrumSample) -> None:
        if sample.representation not in self.representations:
            raise ValueError(
                f"normalization {self.name} does not support representation {sample.representation}"
            )
        if sample.modality not in self.modalities:
            raise ValueError(
                f"normalization {self.name} does not support modality {sample.modality}"
            )


@dataclass(frozen=True, slots=True)
class NormalizationState:
    """Persistable authority for replaying one normalization exactly."""

    schema_version: Literal["hyperspectrum-normalization-state/v1"]
    method: str
    fit_scope: FitScope
    invertibility: Invertibility
    modality: SpectralModality
    representation: SpectrumRepresentation
    signal_unit: str
    parameters: FrozenJsonMapping | Mapping[str, object]
    training_sample_ids: tuple[str, ...]
    training_group_ids: tuple[str, ...]
    split_manifest_digest: str | None
    fit_data_digest: str
    sample_id: str | None

    def __post_init__(self) -> None:
        training_sample_ids = tuple(self.training_sample_ids)
        training_group_ids = tuple(self.training_group_ids)
        object.__setattr__(self, "training_sample_ids", training_sample_ids)
        object.__setattr__(self, "training_group_ids", training_group_ids)
        if self.schema_version != "hyperspectrum-normalization-state/v1":
            raise ValueError("unsupported normalization state schema")
        if not self.method.strip() or not self.signal_unit.strip():
            raise ValueError("normalization method and signal unit must be non-empty")
        if _SHA256.fullmatch(self.fit_data_digest) is None:
            raise ValueError("fit_data_digest must be a lowercase SHA-256")
        expected_scope = _METHOD_FIT_SCOPES.get(self.method)
        if expected_scope is None:
            raise ValueError(
                f"normalization parameters use unknown method {self.method}"
            )
        if self.fit_scope != expected_scope:
            raise ValueError("normalization parameters do not match method fit scope")
        if self.invertibility != "exact":
            raise ValueError(
                "built-in normalization parameters require exact inversion"
            )
        if len(training_sample_ids) != len(set(training_sample_ids)):
            raise ValueError("training_sample_ids must be unique")
        if len(training_group_ids) != len(set(training_group_ids)):
            raise ValueError("training_group_ids must be unique")
        if any(not sample_id.strip() for sample_id in training_sample_ids):
            raise ValueError("training_sample_ids must be non-empty")
        if any(not group_id.strip() for group_id in training_group_ids):
            raise ValueError("training_group_ids must be non-empty")
        if self.fit_scope == "per_sample":
            if self.sample_id is None or not self.sample_id.strip():
                raise ValueError("per-sample normalization must bind a sample_id")
            if self.training_sample_ids:
                raise ValueError(
                    "per-sample normalization cannot claim training samples"
                )
            if self.training_group_ids or self.split_manifest_digest is not None:
                raise ValueError(
                    "per-sample normalization cannot claim a training split manifest"
                )
        elif self.fit_scope == "dataset_fitted":
            if self.sample_id is not None:
                raise ValueError(
                    "dataset-fitted normalization cannot bind one sample_id"
                )
            if not self.training_sample_ids:
                raise ValueError(
                    "dataset-fitted normalization requires training_sample_ids"
                )
            if not self.training_group_ids:
                raise ValueError(
                    "dataset-fitted normalization requires training_group_ids"
                )
            if (
                self.split_manifest_digest is None
                or _SHA256.fullmatch(self.split_manifest_digest) is None
            ):
                raise ValueError(
                    "dataset-fitted normalization requires a split_manifest_digest"
                )
        else:
            raise ValueError("unsupported normalization fit_scope")
        try:
            parameters = freeze_json_mapping(self.parameters)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"normalization parameters are invalid: {error}"
            ) from error
        expected_keys = set(_STATE_PARAMETER_KEYS[self.method])
        if set(parameters) != expected_keys:
            raise ValueError(
                f"normalization parameters must contain exactly {sorted(expected_keys)}"
            )
        for key in expected_keys:
            value = parameters[key]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(float(value))
            ):
                raise ValueError(
                    f"normalization parameters.{key} must be finite numeric"
                )
        scale = parameters["scale"]
        if not isinstance(scale, (int, float)) or float(scale) <= 1e-12:
            raise ValueError(
                "normalization parameters.scale must be greater than 1e-12"
            )
        object.__setattr__(self, "parameters", parameters)

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-safe persistence payload."""
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "fit_scope": self.fit_scope,
            "invertibility": self.invertibility,
            "modality": self.modality,
            "representation": self.representation,
            "signal_unit": self.signal_unit,
            "parameters": thaw_json_mapping(cast(FrozenJsonMapping, self.parameters)),
            "training_sample_ids": list(self.training_sample_ids),
            "training_group_ids": list(self.training_group_ids),
            "split_manifest_digest": self.split_manifest_digest,
            "fit_data_digest": self.fit_data_digest,
            "sample_id": self.sample_id,
        }


@dataclass(frozen=True, slots=True)
class NormalizedSpectrum:
    """A dimensionless sample paired with the state needed for native recovery."""

    sample: SpectrumSample
    state: NormalizationState

    def __post_init__(self) -> None:
        if self.sample.signal_unit != "1":
            raise ValueError("normalized samples must use the dimensionless unit '1'")
        if self.sample.modality != self.state.modality:
            raise ValueError("normalized sample modality must match state")
        if self.sample.representation != self.state.representation:
            raise ValueError("normalized sample representation must match state")
        if (
            self.state.fit_scope == "per_sample"
            and self.state.sample_id != self.sample.sample_id
        ):
            raise ValueError("per-sample normalization state is sample-bound")


class NormalizationRegistry:
    """Immutable lookup boundary for normalization declarations."""

    def __init__(self, definitions: tuple[NormalizationDefinition, ...]) -> None:
        entries = {definition.name: definition for definition in definitions}
        if len(entries) != len(definitions):
            raise ValueError("normalization names must be unique")
        self._entries = entries

    def get(self, name: str) -> NormalizationDefinition:
        try:
            return self._entries[name]
        except KeyError as error:
            raise ValueError(f"unknown normalization: {name}") from error


_REAL_DENSE_MODALITIES = frozenset(
    {"xas", "xanes", "exafs", "raman", "ir", "eels", "hyperspectral"}
)
_DEFAULT_DEFINITIONS = (
    NormalizationDefinition(
        "identity_raw",
        "per_sample",
        "exact",
        _REAL_DENSE_MODALITIES,
        frozenset({"dense"}),
    ),
    NormalizationDefinition(
        "per_spectrum_range",
        "per_sample",
        "exact",
        _REAL_DENSE_MODALITIES,
        frozenset({"dense"}),
    ),
    NormalizationDefinition(
        "total_ion_current",
        "per_sample",
        "exact",
        frozenset({"mass_spectrometry"}),
        frozenset({"sparse_peaks"}),
    ),
    NormalizationDefinition(
        "complex_rms", "per_sample", "exact", frozenset({"nmr"}), frozenset({"complex"})
    ),
    NormalizationDefinition(
        "train_global_standard",
        "dataset_fitted",
        "exact",
        _REAL_DENSE_MODALITIES,
        frozenset({"dense"}),
    ),
)


def default_normalization_registry() -> NormalizationRegistry:
    """Return the built-in M0 registry."""
    return NormalizationRegistry(_DEFAULT_DEFINITIONS)


def recommended_normalization(modality: str, representation: str) -> str:
    """Return an explicit conservative default or refuse an unknown combination."""
    if (
        modality in {"xas", "xanes", "exafs", "raman", "ir", "eels", "hyperspectral"}
        and representation == "dense"
    ):
        return "per_spectrum_range"
    if modality == "mass_spectrometry" and representation == "sparse_peaks":
        return "total_ion_current"
    if modality == "nmr" and representation == "complex":
        return "complex_rms"
    raise ValueError(
        f"no recommended normalization for modality={modality}, representation={representation}"
    )


def _state(
    definition: NormalizationDefinition,
    sample: SpectrumSample,
    *,
    parameters: dict[str, float],
) -> NormalizationState:
    return NormalizationState(
        schema_version="hyperspectrum-normalization-state/v1",
        method=definition.name,
        fit_scope=definition.fit_scope,
        invertibility=definition.invertibility,
        modality=sample.modality,
        representation=sample.representation,
        signal_unit=sample.signal_unit,
        parameters=freeze_json_mapping(parameters),
        training_sample_ids=(),
        training_group_ids=(),
        split_manifest_digest=None,
        fit_data_digest=samples_digest((sample,)),
        sample_id=sample.sample_id,
    )


def _derive_per_sample_state(
    definition: NormalizationDefinition, sample: SpectrumSample
) -> NormalizationState:
    values = sample.signal[sample.valid_mask]
    if definition.name == "identity_raw":
        return _state(definition, sample, parameters={"offset": 0.0, "scale": 1.0})
    if definition.name == "per_spectrum_range":
        minimum = float(np.min(values))
        scale = float(np.max(values) - minimum)
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError("per-spectrum dynamic range must be greater than 1e-12")
        return _state(
            definition, sample, parameters={"offset": minimum, "scale": scale}
        )
    if definition.name == "total_ion_current":
        if np.any(values < 0.0):
            raise ValueError(
                "total-ion-current normalization requires non-negative values"
            )
        scale = float(np.sum(values))
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError("total ion current must be greater than 1e-12")
        return _state(definition, sample, parameters={"offset": 0.0, "scale": scale})
    if definition.name == "complex_rms":
        scale = float(np.sqrt(np.mean(np.abs(values) ** 2)))
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError("complex RMS must be greater than 1e-12")
        return _state(definition, sample, parameters={"offset": 0.0, "scale": scale})
    raise ValueError(
        f"normalization {definition.name} has no per-sample implementation"
    )


def fit_normalizer(
    method: str,
    samples: tuple[SpectrumSample, ...],
    *,
    manifest: SplitManifest,
    registry: NormalizationRegistry | None = None,
) -> NormalizationState:
    """Fit a dataset-scoped normalizer using the train split only."""
    definition = (registry or default_normalization_registry()).get(method)
    if definition.fit_scope != "dataset_fitted":
        raise ValueError(
            f"normalization {method} is per-sample and must not be dataset-fitted"
        )
    if not samples:
        raise ValueError("at least one training sample is required")
    sample_ids = tuple(sample.sample_id for sample in samples)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("training sample IDs must be unique")
    reference = samples[0]
    for sample in samples:
        try:
            assignment = manifest.assignment_for(sample.sample_id, sample.group_id)
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"training sample is not bound to the supplied split manifest: {error}"
            ) from error
        if assignment.split != "train":
            raise ValueError(
                "dataset-fitted normalization may fit only the train split"
            )
        definition.require_compatible(sample)
        if sample.modality != reference.modality:
            raise ValueError("fitted samples must share one modality")
        if sample.representation != reference.representation:
            raise ValueError("fitted samples must share one representation")
        if sample.signal_unit != reference.signal_unit:
            raise ValueError("fitted samples must share one signal unit")
    values = np.concatenate(
        tuple(sample.signal[sample.valid_mask] for sample in samples)
    )
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("training standard deviation must be greater than 1e-12")
    return NormalizationState(
        schema_version="hyperspectrum-normalization-state/v1",
        method=definition.name,
        fit_scope=definition.fit_scope,
        invertibility=definition.invertibility,
        modality=reference.modality,
        representation=reference.representation,
        signal_unit=reference.signal_unit,
        parameters=freeze_json_mapping({"mean": mean, "scale": scale}),
        training_sample_ids=sample_ids,
        training_group_ids=tuple(dict.fromkeys(sample.group_id for sample in samples)),
        split_manifest_digest=manifest.digest,
        fit_data_digest=samples_digest(samples),
        sample_id=None,
    )


def _require_state_matches(sample: SpectrumSample, state: NormalizationState) -> None:
    if sample.representation != state.representation:
        raise ValueError("normalization state representation does not match sample")
    if sample.modality != state.modality:
        raise ValueError("normalization state modality does not match sample")
    if sample.signal_unit != state.signal_unit:
        raise ValueError("normalization state signal unit does not match sample")
    if state.fit_scope == "per_sample" and state.sample_id != sample.sample_id:
        raise ValueError("per-sample normalization state is sample-bound")


def _numeric_parameter(state: NormalizationState, key: str) -> float:
    value = state.parameters[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"normalization parameter {key} must be numeric")
    return float(value)


def normalize(
    sample: SpectrumSample,
    method: str,
    *,
    state: NormalizationState | None = None,
    registry: NormalizationRegistry | None = None,
) -> NormalizedSpectrum:
    """Apply either a sample-derived or already-fitted normalization state."""
    definition = (registry or default_normalization_registry()).get(method)
    definition.require_compatible(sample)
    if state is None:
        if definition.fit_scope == "dataset_fitted":
            raise ValueError("dataset-fitted normalization requires a fitted state")
        state = _derive_per_sample_state(definition, sample)
    else:
        if state.method != method:
            raise ValueError(
                "normalization state method does not match requested method"
            )
        if (
            state.fit_scope != definition.fit_scope
            or state.invertibility != definition.invertibility
        ):
            raise ValueError("normalization state declaration does not match registry")
        _require_state_matches(sample, state)
    scale = _numeric_parameter(state, "scale")
    offset_key = "mean" if "mean" in state.parameters else "offset"
    offset = _numeric_parameter(state, offset_key)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = (sample.signal - offset) / scale
    if not np.isfinite(transformed[sample.valid_mask]).all():
        raise ValueError("normalization produced non-finite valid values")
    return NormalizedSpectrum(sample.with_signal(transformed, signal_unit="1"), state)


def denormalize(normalized: NormalizedSpectrum) -> SpectrumSample:
    """Recover native signal values and units when exact inversion is declared."""
    state = normalized.state
    if state.invertibility != "exact":
        raise ValueError("native-unit recovery requires exact normalization inversion")
    scale = _numeric_parameter(state, "scale")
    offset_key = "mean" if "mean" in state.parameters else "offset"
    offset = _numeric_parameter(state, offset_key)
    with np.errstate(over="ignore", invalid="ignore"):
        native = normalized.sample.signal * scale + offset
    if not np.isfinite(native[normalized.sample.valid_mask]).all():
        raise ValueError("normalization inverse produced non-finite valid values")
    return normalized.sample.with_signal(native, signal_unit=state.signal_unit)
