"""Model-independent contracts for reusable spectral denoising."""

from .evaluation import (
    DenoisingPair,
    DenoisingSuiteReport,
    MetricValues,
    ModalityEvaluation,
    SampleEvaluation,
    evaluate_denoising_suite,
)
from .model import (
    CanonicalDenoisingInput,
    CanonicalDenoisingOutput,
    CompatibilityIssue,
    DenoisingModel,
    ModelCapabilities,
    normalization_state_digest,
    validate_model_output,
)
from .noise import CorruptedSpectrum, NoiseProvenance, NoiseSpec, inject_noise
from .normalization import (
    NormalizationDefinition,
    NormalizationRegistry,
    NormalizationState,
    NormalizedSpectrum,
    default_normalization_registry,
    denormalize,
    fit_normalizer,
    normalize,
    recommended_normalization,
)
from .sample import (
    SpectralModality,
    SpectrumAxis,
    SpectrumRepresentation,
    SpectrumSample,
)
from .split import SplitAssignment, SplitEntry, SplitManifest

__all__ = [
    "CanonicalDenoisingInput",
    "CanonicalDenoisingOutput",
    "CompatibilityIssue",
    "CorruptedSpectrum",
    "DenoisingModel",
    "DenoisingPair",
    "DenoisingSuiteReport",
    "MetricValues",
    "ModalityEvaluation",
    "ModelCapabilities",
    "NoiseProvenance",
    "NoiseSpec",
    "NormalizationDefinition",
    "NormalizationRegistry",
    "NormalizationState",
    "NormalizedSpectrum",
    "SampleEvaluation",
    "SpectralModality",
    "SpectrumAxis",
    "SpectrumRepresentation",
    "SpectrumSample",
    "SplitAssignment",
    "SplitEntry",
    "SplitManifest",
    "default_normalization_registry",
    "denormalize",
    "evaluate_denoising_suite",
    "fit_normalizer",
    "inject_noise",
    "normalization_state_digest",
    "normalize",
    "recommended_normalization",
    "validate_model_output",
]
