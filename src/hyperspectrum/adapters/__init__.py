"""Thin adapters for pinned external spectroscopy models.

Submodules stay lazy so registry discovery and traditional baselines never import
optional model runtimes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .hypersigma import HyperSIGMADenoiseAdapter, denoise_cubes

__all__ = ["HyperSIGMADenoiseAdapter", "denoise_cubes"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .hypersigma import HyperSIGMADenoiseAdapter, denoise_cubes

        return {
            "HyperSIGMADenoiseAdapter": HyperSIGMADenoiseAdapter,
            "denoise_cubes": denoise_cubes,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
