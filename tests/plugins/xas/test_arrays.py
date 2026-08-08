"""Canonical XAS array tests and deterministic synthetic fixture generator.

Regenerate the committed test-only fixture from the repository root with::

    PYTHONPATH=src .venv/bin/python tests/plugins/xas/test_arrays.py --generate-fixture

The generator uses only literal NumPy operations and writes keys in a fixed order.
The fixture metadata explicitly records that these are synthetic spectra.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from hyperspectrum.plugins.xas.arrays import XASSpectrum, interpolate_spectrum

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = ROOT / "tests/fixtures/xas/denoising-pairs.npz"


def write_synthetic_fixture(path: Path) -> None:
    """Write the small deterministic XAS denoising fixture used by tests."""
    energy_axis = np.linspace(7110.0, 7118.0, 9, dtype=np.float64)
    group_offsets = np.repeat(np.array([0.0, 0.08, -0.05]), 2)
    replicate_offsets = np.tile(np.array([-0.01, 0.01]), 3)
    edge = 1.0 / (1.0 + np.exp(-(energy_axis - 7114.0) * 1.4))
    peak = 0.22 * np.exp(-0.5 * ((energy_axis - 7115.0) / 0.75) ** 2)
    clean = np.stack(
        [edge + peak + group + replicate for group, replicate in zip(group_offsets, replicate_offsets)]
    )
    noise_patterns = np.array(
        [
            [0.02, -0.01, 0.03, -0.02, 0.00, 0.01, -0.02, 0.01, -0.01],
            [-0.01, 0.02, -0.02, 0.01, 0.03, -0.01, 0.01, -0.02, 0.02],
            [0.03, 0.01, -0.01, -0.03, 0.02, 0.00, -0.01, 0.02, -0.02],
            [-0.02, 0.00, 0.02, -0.01, 0.01, 0.03, -0.03, 0.01, 0.00],
            [0.01, -0.03, 0.00, 0.02, -0.01, 0.02, 0.01, -0.01, 0.03],
            [0.00, 0.02, -0.03, 0.01, 0.02, -0.02, 0.03, -0.01, 0.01],
        ],
        dtype=np.float64,
    )
    metadata = json.dumps(
        {
            "description": "Deterministic tiny XAS denoising pairs for tests only",
            "energy_unit": "eV",
            "synthetic": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        energy=np.tile(energy_axis, (6, 1)),
        noisy=clean + noise_patterns,
        clean=clean,
        sample_ids=np.array(["feo-1", "feo-2", "fe2o3-1", "fe2o3-2", "fe3o4-1", "fe3o4-2"]),
        group_ids=np.array(["FeO", "FeO", "Fe2O3", "Fe2O3", "Fe3O4", "Fe3O4"]),
        energy_unit=np.array("eV"),
        metadata_json=np.array(metadata),
    )


def spectrum(**changes: object) -> XASSpectrum:
    values: dict[str, object] = {
        "sample_id": "sample-1",
        "group_id": "FeO",
        "energy": np.array([7110.0, 7111.0, 7112.0]),
        "intensity": np.array([0.1, 0.4, 0.9]),
        "energy_unit": "eV",
    }
    values.update(changes)
    return XASSpectrum(**values)  # type: ignore[arg-type]


def test_canonical_spectrum_preserves_explicit_coordinates_and_ids() -> None:
    # Break caught: canonicalization could replace IDs or infer an energy axis from array shape.
    item = spectrum()

    assert item.sample_id == "sample-1"
    assert item.group_id == "FeO"
    np.testing.assert_array_equal(item.energy, [7110.0, 7111.0, 7112.0])
    assert item.energy_unit == "eV"


@pytest.mark.parametrize(
    "energy",
    [
        np.array([7110.0, 7111.0, 7111.0]),
        np.array([7110.0, 7112.0, 7111.0]),
    ],
)
def test_canonical_spectrum_rejects_energy_that_is_not_strictly_monotonic(
    energy: np.ndarray,
) -> None:
    # Break caught: duplicate or reversing coordinates could enter interpolation silently.
    with pytest.raises(ValueError, match="strictly monotonic"):
        spectrum(energy=energy)


def test_canonical_spectrum_accepts_strictly_decreasing_energy() -> None:
    # Break caught: valid descending instrument coordinates could be rejected as unordered.
    item = spectrum(
        energy=np.array([7112.0, 7111.0, 7110.0]),
        intensity=np.array([0.9, 0.4, 0.1]),
    )

    np.testing.assert_array_equal(item.energy, [7112.0, 7111.0, 7110.0])


def test_canonical_spectrum_requires_exact_ev_unit() -> None:
    # Break caught: energy units could be guessed or implicitly converted from a label.
    with pytest.raises(ValueError, match="exactly 'eV'"):
        spectrum(energy_unit="keV")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("energy", np.array([7110.0, np.nan, 7112.0])),
        ("intensity", np.array([0.1, np.inf, 0.9])),
    ],
)
def test_canonical_spectrum_rejects_non_finite_arrays(field: str, value: np.ndarray) -> None:
    # Break caught: a NaN or infinity could contaminate filtering and metrics.
    with pytest.raises(ValueError, match="finite"):
        spectrum(**{field: value})


def test_interpolation_uses_explicit_target_coordinates_and_preserves_ids() -> None:
    # Break caught: interpolation could synthesize coordinates or detach sample/group identity.
    result = interpolate_spectrum(spectrum(), np.array([7110.5, 7111.5]), energy_unit="eV")

    np.testing.assert_allclose(result.intensity, [0.25, 0.65])
    np.testing.assert_array_equal(result.energy, [7110.5, 7111.5])
    assert (result.sample_id, result.group_id, result.energy_unit) == ("sample-1", "FeO", "eV")


def test_interpolation_rejects_any_target_coordinate_outside_closed_overlap() -> None:
    # Break caught: interpolation could extrapolate, fill edges, or silently clip a partial overlap.
    with pytest.raises(ValueError, match="closed overlap"):
        interpolate_spectrum(spectrum(), np.array([7109.5, 7110.5]), energy_unit="eV")


def test_committed_fixture_is_synthetic_grouped_and_byte_reproducible(tmp_path: Path) -> None:
    # Break caught: hand-edited or nondeterministic fixture bytes could lose provenance or grouping.
    regenerated = tmp_path / "denoising-pairs.npz"
    write_synthetic_fixture(regenerated)

    assert regenerated.read_bytes() == FIXTURE_PATH.read_bytes()
    with np.load(regenerated, allow_pickle=False) as fixture:
        assert json.loads(str(fixture["metadata_json"]))["synthetic"] is True
        assert str(fixture["energy_unit"]) == "eV"
        assert set(fixture["group_ids"].tolist()) == {"FeO", "Fe2O3", "Fe3O4"}
        assert fixture["energy"].shape == fixture["noisy"].shape == fixture["clean"].shape


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-fixture", action="store_true")
    args = parser.parse_args()
    if not args.generate_fixture:
        parser.error("pass --generate-fixture to write the deterministic synthetic fixture")
    write_synthetic_fixture(FIXTURE_PATH)


if __name__ == "__main__":
    _main()
