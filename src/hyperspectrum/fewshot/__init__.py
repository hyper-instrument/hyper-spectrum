"""HyperData few-shot XAS solver: k-NN baseline plus the seven-file delivery.

The solver maps simulated XAS spectra to experimental ones and back
("sim2exp" / "exp2sim") on the organiser's fixed 133-point energy grid. The
public pieces are:

- :mod:`hyperspectrum.fewshot.knn` — the reference k-NN baseline, ported with
  its numerics intact (``PairIndex``, ``predict_many``).
- :mod:`hyperspectrum.fewshot.delivery` — writes and size-checks the seven
  files the organiser's evaluator consumes.
- :mod:`hyperspectrum.fewshot.solver` — the fixed ``prepare`` / ``train`` /
  ``predict`` CLI a kind-agnostic kit drives as a subprocess.
"""

from __future__ import annotations

from .delivery import DELIVERY_FILES, LIMITS, Usage, write_delivery
from .knn import (
    GRID_POINTS,
    NEIGHBOR_COUNT,
    PairIndex,
    candidate_indices,
    predict_many,
    reverse_direction,
)

__all__ = [
    "DELIVERY_FILES",
    "GRID_POINTS",
    "LIMITS",
    "NEIGHBOR_COUNT",
    "PairIndex",
    "Usage",
    "candidate_indices",
    "predict_many",
    "reverse_direction",
    "write_delivery",
]
