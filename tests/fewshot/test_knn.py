from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hyperspectrum.fewshot.knn import (
    GRID_POINTS,
    PairIndex,
    candidate_indices,
    predict_many,
    reverse_direction,
)


def constant(value: float) -> np.ndarray:
    return np.full(GRID_POINTS, value, dtype=np.float32)


def pool(
    ids: list[str],
    simulation: list[np.ndarray],
    experiment: list[np.ndarray],
    atomic_numbers: list[int],
    edge_codes: list[int],
) -> dict[str, Any]:
    return {
        "sample_id": np.asarray(ids),
        "group_id": np.asarray([f"g-{sample_id}" for sample_id in ids]),
        "energy": np.linspace(-3.0, 30.0, GRID_POINTS, dtype=np.float32),
        "simulation": np.stack(simulation).astype(np.float32),
        "experiment": np.stack(experiment).astype(np.float32),
        "absorber_atomic_number": np.asarray(atomic_numbers, dtype=np.int16),
        "edge_code": np.asarray(edge_codes, dtype=np.int8),
    }


def queries(
    directions: list[str],
    spectra: list[np.ndarray],
    atomic_numbers: list[int],
    edge_codes: list[int],
) -> dict[str, Any]:
    return {
        "sample_id": np.asarray([f"q-{index}" for index in range(len(directions))]),
        "direction": np.asarray(directions),
        "energy": np.linspace(-3.0, 30.0, GRID_POINTS, dtype=np.float32),
        "spectrum": np.stack(spectra).astype(np.float32),
        "absorber_atomic_number": np.asarray(atomic_numbers, dtype=np.int16),
        "edge_code": np.asarray(edge_codes, dtype=np.int8),
    }


def three_pair_index() -> PairIndex:
    return PairIndex.from_pools(
        [
            pool(
                ["p-0", "p-1", "p-2"],
                [constant(0.0), constant(1.0), constant(3.0)],
                [constant(10.0), constant(20.0), constant(30.0)],
                [29, 29, 29],
                [1, 1, 1],
            )
        ]
    )


def test_prediction_is_inverse_square_weighted_average_of_neighbours() -> None:
    index = three_pair_index()
    query = queries(["sim2exp"], [constant(0.25)], [29], [1])

    prediction = predict_many(query, index)

    # Mean absolute distances to the three constant training spectra.
    d0, d1, d2 = 0.25, 0.75, 2.75
    w0, w1, w2 = 1 / d0**2, 1 / d1**2, 1 / d2**2
    expected = (w0 * 10.0 + w1 * 20.0 + w2 * 30.0) / (w0 + w1 + w2)
    assert prediction.shape == (1, GRID_POINTS)
    assert prediction.dtype == np.float32
    np.testing.assert_allclose(prediction[0], expected, rtol=1e-6)


def test_exp2sim_swaps_input_and_output_domains() -> None:
    index = three_pair_index()
    query = queries(["exp2sim"], [constant(10.0)], [29], [1])

    prediction = predict_many(query, index)

    # Exact match on the experiment spectrum of p-0 returns its simulation.
    np.testing.assert_array_equal(prediction[0], constant(0.0))


def test_exact_match_is_decisive_and_takes_lexicographically_first_id() -> None:
    query_spectrum = constant(2.0)
    index = PairIndex.from_pools(
        [
            pool(
                ["p-b", "p-a", "p-far"],
                [query_spectrum, query_spectrum, constant(2.5)],
                [constant(1.0), constant(5.0), constant(100.0)],
                [29, 29, 29],
                [1, 1, 1],
            )
        ]
    )
    query = queries(["sim2exp"], [query_spectrum], [29], [1])

    prediction = predict_many(query, index)

    np.testing.assert_array_equal(prediction[0], constant(5.0))


def test_same_atomic_number_and_edge_preferred_over_same_edge() -> None:
    query_spectrum = constant(2.0)
    index = PairIndex.from_pools(
        [
            pool(
                ["same-z", "other-z"],
                [constant(9.0), query_spectrum],
                [constant(1.0), constant(2.0)],
                [29, 26],
                [1, 1],
            )
        ]
    )
    query = queries(["sim2exp"], [query_spectrum], [29], [1])

    prediction = predict_many(query, index)

    # The exact match under another absorber is never a candidate.
    np.testing.assert_array_equal(prediction[0], constant(1.0))


def test_candidates_fall_back_to_same_edge_then_to_everything() -> None:
    train_z = np.asarray([29, 26, 26], dtype=np.int16)
    train_edge = np.asarray([1, 1, 2], dtype=np.int8)

    np.testing.assert_array_equal(candidate_indices(train_z, train_edge, 29, 1), [0])
    np.testing.assert_array_equal(candidate_indices(train_z, train_edge, 99, 1), [0, 1])
    np.testing.assert_array_equal(
        candidate_indices(train_z, train_edge, 99, 7), [0, 1, 2]
    )
    assert candidate_indices(train_z, train_edge, 99, 7).dtype == np.int64


def test_prediction_uses_same_edge_fallback_then_all_pairs() -> None:
    index = PairIndex.from_pools(
        [
            pool(
                ["edge-1", "edge-2"],
                [constant(0.0), constant(0.0)],
                [constant(1.0), constant(2.0)],
                [29, 26],
                [1, 2],
            )
        ]
    )
    same_edge = queries(["sim2exp"], [constant(0.0)], [99], [2])
    np.testing.assert_array_equal(predict_many(same_edge, index)[0], constant(2.0))

    # Unknown edge: both pairs are exact matches, the first id wins.
    all_pairs = queries(["sim2exp"], [constant(0.0)], [0], [0])
    np.testing.assert_array_equal(predict_many(all_pairs, index)[0], constant(1.0))


def test_at_most_eight_neighbours_are_averaged() -> None:
    ids = [f"p-{index:02d}" for index in range(10)]
    index = PairIndex.from_pools(
        [
            pool(
                ids,
                [constant(1.0 + step) for step in range(10)],
                [constant(100.0 * step) for step in range(10)],
                [29] * 10,
                [1] * 10,
            )
        ]
    )
    query = queries(["sim2exp"], [constant(0.0)], [29], [1])

    prediction = predict_many(query, index)

    weights = [1 / (1.0 + step) ** 2 for step in range(8)]
    expected = sum(w * 100.0 * step for step, w in enumerate(weights)) / sum(weights)
    np.testing.assert_allclose(prediction[0], expected, rtol=1e-6)


def test_invalid_direction_raises_value_error() -> None:
    index = three_pair_index()
    query = queries(["sideways"], [constant(0.0)], [29], [1])

    with pytest.raises(ValueError, match="direction"):
        predict_many(query, index)


def test_spectrum_with_wrong_grid_raises_value_error() -> None:
    index = three_pair_index()
    query = queries(["sim2exp"], [constant(0.0)], [29], [1])
    query["spectrum"] = query["spectrum"][:, :-1]

    with pytest.raises(ValueError, match="spectrum"):
        predict_many(query, index)


def test_prediction_is_deterministic_and_index_order_independent() -> None:
    first = pool(["p-1"], [constant(0.5)], [constant(5.0)], [29], [1])
    second = pool(["p-0"], [constant(0.0)], [constant(0.0)], [29], [1])
    index_a = PairIndex.from_pools([first, second])
    index_b = PairIndex.from_pools([second, first])
    query = queries(["sim2exp"] * 3, [constant(0.2)] * 3, [29] * 3, [1] * 3)

    assert list(index_a.sample_id) == ["p-0", "p-1"]
    assert index_a.simulation.tobytes() == index_b.simulation.tobytes()
    assert (
        predict_many(query, index_a).tobytes() == predict_many(query, index_b).tobytes()
    )


def test_pair_index_round_trips_through_npz(tmp_path: Path) -> None:
    index = three_pair_index()
    path = tmp_path / "index.npz"

    index.save(path)
    loaded = PairIndex.load(path)

    assert list(loaded.sample_id) == list(index.sample_id)
    assert loaded.simulation.dtype == np.float32
    assert loaded.atomic_number.dtype == np.int16
    assert loaded.edge_code.dtype == np.int8
    np.testing.assert_array_equal(loaded.experiment, index.experiment)
    assert len(loaded) == 3


def test_saved_index_has_exactly_the_five_members(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    three_pair_index().save(path)

    # In particular no stray ``allow_pickle`` member, which numpy < 2.4 would
    # have written had the kwarg been passed to ``np.savez``.
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == [
            "absorber_atomic_number.npy",
            "edge_code.npy",
            "experiment.npy",
            "sample_id.npy",
            "simulation.npy",
        ]


def test_load_rejects_unexpected_members(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    three_pair_index().save(path)
    # Reproduce the stray member numpy < 2.4 writes for a savez(allow_pickle=...)
    # kwarg (newer numpy consumes the kwarg, so append the member by hand).
    with (
        zipfile.ZipFile(path, "a") as archive,
        archive.open("allow_pickle.npy", "w") as member,
    ):
        np.lib.format.write_array(member, np.asarray(False))

    with pytest.raises(ValueError, match="allow_pickle"):
        PairIndex.load(path)


def test_from_pools_rejects_missing_keys_bad_shapes_and_non_finite() -> None:
    good = pool(["p-0"], [constant(0.0)], [constant(1.0)], [29], [1])

    missing = dict(good)
    del missing["edge_code"]
    with pytest.raises(ValueError, match="edge_code"):
        PairIndex.from_pools([missing])

    narrow = dict(good)
    narrow["simulation"] = good["simulation"][:, :10]
    with pytest.raises(ValueError, match="simulation"):
        PairIndex.from_pools([narrow])

    non_finite = dict(good)
    non_finite["experiment"] = np.full((1, GRID_POINTS), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        PairIndex.from_pools([non_finite])

    with pytest.raises(ValueError, match="empty"):
        PairIndex.from_pools([])


def test_reverse_direction() -> None:
    assert reverse_direction("sim2exp") == "exp2sim"
    assert reverse_direction("exp2sim") == "sim2exp"
    with pytest.raises(ValueError, match="direction"):
        reverse_direction("both")
