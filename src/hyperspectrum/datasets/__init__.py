"""Dataset-specific adapters that materialize strict scientific contracts."""

from .xanes_spec import (
    BenchmarkAssetIdentity,
    MaterializationResult,
    ParsedXanesSpectrum,
    SourceIntegrityError,
    SpectrumRejected,
    load_benchmark_asset_identity,
    load_denoising_pairs,
    materialize_xanes_spec,
    parse_xanes_spec,
)

__all__ = [
    "BenchmarkAssetIdentity",
    "MaterializationResult",
    "ParsedXanesSpectrum",
    "SourceIntegrityError",
    "SpectrumRejected",
    "load_benchmark_asset_identity",
    "load_denoising_pairs",
    "materialize_xanes_spec",
    "parse_xanes_spec",
]
