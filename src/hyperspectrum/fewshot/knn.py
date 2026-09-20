"""k-nearest-neighbour baseline for sim<->exp XAS mapping.

This is a faithful port of the organiser's reference baseline. The numerics
are deliberately unchanged: candidates are chosen by absorber and edge,
ranked by mean absolute distance with the sample id as tie-break
(``np.lexsort``), an exact match (distance <= 1e-12) is decisive, and the
remaining neighbours are averaged with inverse-square weights on a distance
floored at 1e-6. All arithmetic is float64; predictions are emitted as float32.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

GRID_POINTS = 133
NEIGHBOR_COUNT = 8
DISTANCE_POWER = 2.0
EXACT_MATCH_TOLERANCE = 1e-12
DISTANCE_FLOOR = 1e-6
VALID_DIRECTIONS = frozenset({"sim2exp", "exp2sim"})

#: Keys every paired public pool carries; the index only consumes some of them.
PAIRED_POOL_KEYS = (
    "sample_id",
    "group_id",
    "energy",
    "simulation",
    "experiment",
    "absorber_atomic_number",
    "edge_code",
)
INDEX_POOL_KEYS = (
    "sample_id",
    "simulation",
    "experiment",
    "absorber_atomic_number",
    "edge_code",
)
QUERY_KEYS = (
    "sample_id",
    "direction",
    "spectrum",
    "absorber_atomic_number",
    "edge_code",
)

_DIRECTION_KEYS = {
    "sim2exp": ("simulation", "experiment"),
    "exp2sim": ("experiment", "simulation"),
}


def reverse_direction(direction: str) -> str:
    """Return the mapping direction that undoes ``direction``."""
    if direction == "sim2exp":
        return "exp2sim"
    if direction == "exp2sim":
        return "sim2exp"
    raise ValueError(
        f"unknown direction {direction!r}; expected one of {sorted(VALID_DIRECTIONS)}"
    )


def candidate_indices(
    train_atomic_numbers: NDArray[np.int16],
    train_edge_codes: NDArray[np.int8],
    atomic_number: int,
    edge_code: int,
) -> NDArray[np.int64]:
    """Same absorber and edge, else same edge, else every training pair."""
    exact = np.flatnonzero(
        (train_atomic_numbers == atomic_number) & (train_edge_codes == edge_code)
    )
    if exact.size:
        return np.asarray(exact, dtype=np.int64)
    same_edge = np.flatnonzero(train_edge_codes == edge_code)
    if same_edge.size:
        return np.asarray(same_edge, dtype=np.int64)
    return np.arange(len(train_edge_codes), dtype=np.int64)


def _spectra(
    values: Any, *, name: str, count: int | None = None, dtype: Any = np.float32
) -> NDArray[Any]:
    """Validate an ``(N, 133)`` numeric block; ``dtype=None`` keeps the native one."""
    spectra = np.asarray(values) if dtype is None else np.asarray(values, dtype=dtype)
    if spectra.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric, got dtype {spectra.dtype}")
    if spectra.ndim != 2 or spectra.shape[1] != GRID_POINTS:
        raise ValueError(
            f"{name} must have shape (N, {GRID_POINTS}), got {spectra.shape}"
        )
    if count is not None and spectra.shape[0] != count:
        raise ValueError(f"{name} has {spectra.shape[0]} rows but {count} sample ids")
    if not np.isfinite(spectra).all():
        raise ValueError(f"{name} must be finite")
    return spectra


def _codes(values: Any, *, name: str, count: int, dtype: type) -> NDArray[Any]:
    codes = np.asarray(values, dtype=dtype)
    if codes.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},), got {codes.shape}")
    return codes


def _require_keys(mapping: Mapping[str, Any], keys: Sequence[str], what: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{what} is missing keys: {', '.join(missing)}")


@dataclass(frozen=True)
class PairIndex:
    """Every (simulation, experiment) pair the baseline may draw neighbours from.

    Built from one or more paired pools and stably sorted by ``sample_id`` so
    the index — and every prediction — is independent of pool order.
    """

    sample_id: NDArray[np.str_]
    simulation: NDArray[np.float32]
    experiment: NDArray[np.float32]
    atomic_number: NDArray[np.int16]
    edge_code: NDArray[np.int8]

    def __len__(self) -> int:
        return int(self.sample_id.shape[0])

    @classmethod
    def from_pools(cls, pools: Sequence[Mapping[str, NDArray[Any]]]) -> PairIndex:
        """Validate and concatenate paired pools into one sorted index."""
        ids: list[NDArray[np.str_]] = []
        simulations: list[NDArray[np.float32]] = []
        experiments: list[NDArray[np.float32]] = []
        atomic_numbers: list[NDArray[np.int16]] = []
        edge_codes: list[NDArray[np.int8]] = []
        for position, pool in enumerate(pools):
            what = f"pool[{position}]"
            _require_keys(pool, INDEX_POOL_KEYS, what)
            sample_id = np.asarray(pool["sample_id"]).astype(str)
            if sample_id.ndim != 1:
                raise ValueError(f"{what} sample_id must be one-dimensional")
            count = int(sample_id.shape[0])
            ids.append(sample_id)
            simulations.append(
                _spectra(pool["simulation"], name=f"{what} simulation", count=count)
            )
            experiments.append(
                _spectra(pool["experiment"], name=f"{what} experiment", count=count)
            )
            atomic_numbers.append(
                _codes(
                    pool["absorber_atomic_number"],
                    name=f"{what} absorber_atomic_number",
                    count=count,
                    dtype=np.int16,
                )
            )
            edge_codes.append(
                _codes(
                    pool["edge_code"],
                    name=f"{what} edge_code",
                    count=count,
                    dtype=np.int8,
                )
            )
        if not ids or sum(len(chunk) for chunk in ids) == 0:
            raise ValueError("cannot build an empty pair index")
        sample_id = np.concatenate(ids)
        order = np.argsort(sample_id, kind="stable")
        return cls(
            sample_id=sample_id[order],
            simulation=np.concatenate(simulations)[order],
            experiment=np.concatenate(experiments)[order],
            atomic_number=np.concatenate(atomic_numbers)[order],
            edge_code=np.concatenate(edge_codes)[order],
        )

    def save(self, path: Path) -> None:
        """Persist the index as an ``.npz`` holding exactly ``INDEX_POOL_KEYS``.

        No ``allow_pickle`` kwarg: numpy < 2.4 would store it as a member of
        that name. Nothing here needs pickling, and :meth:`load` refuses it.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            sample_id=self.sample_id,
            simulation=self.simulation,
            experiment=self.experiment,
            absorber_atomic_number=self.atomic_number,
            edge_code=self.edge_code,
        )

    @classmethod
    def load(cls, path: Path) -> PairIndex:
        """Load an index written by :meth:`save`, re-validating it."""
        with np.load(path, allow_pickle=False) as archive:
            unexpected = sorted(set(archive.files) - set(INDEX_POOL_KEYS))
            if unexpected:
                raise ValueError(
                    f"{path} has unexpected members: {', '.join(unexpected)}"
                )
            pool = {key: archive[key] for key in archive.files}
        return cls.from_pools([pool])


def _query_arrays(
    queries: Mapping[str, NDArray[Any]],
) -> tuple[NDArray[np.str_], NDArray[Any], NDArray[np.int16], NDArray[np.int8]]:
    _require_keys(queries, ("direction", "spectrum"), "queries")
    directions = np.asarray(queries["direction"]).astype(str)
    if directions.ndim != 1:
        raise ValueError("queries direction must be one-dimensional")
    count = int(directions.shape[0])
    invalid = sorted(set(directions.tolist()) - VALID_DIRECTIONS)
    if invalid:
        raise ValueError(
            f"queries contain unknown direction values {invalid}; expected "
            f"{sorted(VALID_DIRECTIONS)}"
        )
    spectra = _spectra(
        queries["spectrum"], name="queries spectrum", count=count, dtype=None
    )
    _require_keys(queries, ("absorber_atomic_number", "edge_code"), "queries")
    atomic_numbers = _codes(
        queries["absorber_atomic_number"],
        name="queries absorber_atomic_number",
        count=count,
        dtype=np.int16,
    )
    edge_codes = _codes(
        queries["edge_code"], name="queries edge_code", count=count, dtype=np.int8
    )
    return directions, spectra, atomic_numbers, edge_codes


def predict_many(
    queries: Mapping[str, NDArray[Any]], index: PairIndex
) -> NDArray[np.float32]:
    """Predict the opposite-domain spectrum for every query row.

    ``queries`` needs ``direction`` (``"sim2exp"`` / ``"exp2sim"``),
    ``spectrum`` of shape ``(N, 133)``, ``absorber_atomic_number`` and
    ``edge_code``. Returns an ``(N, 133)`` float32 array in query order.
    """
    directions, spectra, atomic_numbers, edge_codes = _query_arrays(queries)
    train_ids = index.sample_id
    train_atomic_numbers = index.atomic_number
    train_edge_codes = index.edge_code
    predictions: list[NDArray[np.float32]] = []
    for position, direction_value in enumerate(directions.tolist()):
        candidates = candidate_indices(
            train_atomic_numbers,
            train_edge_codes,
            int(atomic_numbers[position]),
            int(edge_codes[position]),
        )
        input_key, output_key = _DIRECTION_KEYS[direction_value]
        query_spectrum = np.asarray(spectra[position], dtype=np.float64)
        candidate_spectra = np.asarray(
            getattr(index, input_key)[candidates], dtype=np.float64
        )
        distances = np.mean(np.abs(candidate_spectra - query_spectrum), axis=1)
        ranked = np.lexsort((train_ids[candidates], distances))
        selected = ranked[: min(NEIGHBOR_COUNT, len(ranked))]
        selected_distances = distances[selected]
        exact = np.flatnonzero(selected_distances <= EXACT_MATCH_TOLERANCE)
        if exact.size:
            selected = selected[exact[:1]]
            selected_distances = distances[selected]
        weights = 1.0 / np.maximum(selected_distances, DISTANCE_FLOOR) ** DISTANCE_POWER
        selected_targets = np.asarray(
            getattr(index, output_key)[candidates[selected]], dtype=np.float64
        )
        prediction = np.average(selected_targets, axis=0, weights=weights)
        predictions.append(prediction.astype(np.float32))
    if not predictions:
        return np.zeros((0, GRID_POINTS), dtype=np.float32)
    return np.stack(predictions)
