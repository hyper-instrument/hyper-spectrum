"""Immutable public contracts shared by HyperSpectrum components."""

from .artifact import ArtifactRef, AxisSpec
from .observation import ObservationBundle
from .prediction import PredictionBundle, PredictionBundleV2, PredictionBundleV3
from .task import MetricSpec, TaskSpec

__all__ = [
    "ArtifactRef",
    "AxisSpec",
    "MetricSpec",
    "ObservationBundle",
    "PredictionBundle",
    "PredictionBundleV2",
    "PredictionBundleV3",
    "TaskSpec",
]
