from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hyperspectrum.datasets.cu_cha import (
    CuChaSpectrumIdentity,
    ParsedCuChaSpectrum,
    SpectrumRejected,
    build_grouped_balanced_split,
    derive_cu_cha_seed,
    load_cu_cha_denoising_pairs,
    materialize_cu_cha,
    parse_cu_cha_dat,
    parse_cu_cha_identity,
    poisson_thin_transmission,
)


def _dat_bytes(
    *,
    energy_kev: np.ndarray | None = None,
    mu_trans: np.ndarray | None = None,
    i0: np.ndarray | None = None,
    i1: np.ndarray | None = None,
    i2: np.ndarray | None = None,
    header: str = "musst_enestep, mu_trans, mu_ref, I0, I1, I2",
) -> bytes:
    if energy_kev is None:
        energy_kev = np.array([8.80, 8.85, 8.90, 8.95, 9.00], dtype=np.float64)
    if i0 is None:
        i0 = np.array([100_000, 101_000, 102_000, 103_000, 104_000], dtype=np.float64)
    if mu_trans is None:
        mu_trans = np.array([0.01, 0.03, 0.08, 0.15, 0.22], dtype=np.float64)
    if i1 is None:
        i1 = i0 / np.exp(mu_trans)
    if i2 is None:
        i2 = np.array([50_000, 50_100, 50_200, 50_300, 50_400], dtype=np.float64)
    mu_ref = np.array([0.5, 0.52, 0.54, 0.56, 0.58], dtype=np.float64)
    rows = np.column_stack((energy_kev, mu_trans, mu_ref, i0, i1, i2))
    lines = ["# T = 200", "# start_time = 2022-07-17T07:43:47+02:00", header]
    lines.extend(" ".join(f"{value:.17g}" for value in row) for row in rows)
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_parser_converts_kev_and_preserves_full_count_proxy_channels(
    tmp_path: Path,
) -> None:
    # Break caught: a generic DAT fallback could keep keV or discard count channels.
    source = tmp_path / "scan.dat"
    source.write_bytes(_dat_bytes())

    spectrum = parse_cu_cha_dat(source)

    assert spectrum.columns == (
        "musst_enestep",
        "mu_trans",
        "mu_ref",
        "I0",
        "I1",
        "I2",
    )
    np.testing.assert_allclose(spectrum.energy_ev, [8800, 8850, 8900, 8950, 9000])
    np.testing.assert_allclose(spectrum.mu_trans, [0.01, 0.03, 0.08, 0.15, 0.22])
    np.testing.assert_allclose(spectrum.mu_trans, np.log(spectrum.i0 / spectrum.i1))
    assert spectrum.energy_ev.flags.writeable is False
    assert spectrum.i2.flags.writeable is False


@pytest.mark.parametrize(
    ("content", "reason_code"),
    [
        (_dat_bytes(header="energy, mu_trans, mu_ref, I0, I1, I2"), "columns"),
        (
            _dat_bytes(energy_kev=np.array([8.8, 8.9, 8.85, 8.95, 9.0])),
            "energy_not_increasing",
        ),
        (
            _dat_bytes(i0=np.array([100_000, 101_000, 0, 103_000, 104_000])),
            "nonpositive_counts",
        ),
        (
            _dat_bytes(i2=np.array([50_000, 50_100, 50_200, -1, 50_400])),
            "nonpositive_counts",
        ),
        (
            _dat_bytes(
                mu_trans=np.array([0.01, 0.03, 0.08, 0.15, 0.60]),
                i1=np.array([100_000, 101_000, 102_000, 103_000, 104_000])
                / np.exp(np.array([0.01, 0.03, 0.08, 0.15, 0.22])),
            ),
            "mu_trans_mismatch",
        ),
    ],
)
def test_parser_fails_closed_on_invalid_scientific_contract(
    tmp_path: Path, content: bytes, reason_code: str
) -> None:
    # Break caught: malformed axes/counts or a false proxy target could be admitted.
    source = tmp_path / "scan.dat"
    source.write_bytes(content)

    with pytest.raises(SpectrumRejected) as error:
        parse_cu_cha_dat(source)

    assert error.value.code == reason_code


def test_relative_path_identity_uses_experiment_and_condition_segment() -> None:
    # Break caught: grouping by individual scan would leak adjacent operando states.
    identity = parse_cu_cha_identity(
        "data_txt/High-Cu_cycles_SO2_O2/100_at_199C_Proc5_SO2_O2_cycles_1st_SO2.dat"
    )

    assert identity.sample_id == (
        "data_txt/High-Cu_cycles_SO2_O2/100_at_199C_Proc5_SO2_O2_cycles_1st_SO2"
    )
    assert identity.experiment == "High-Cu_cycles_SO2_O2"
    assert identity.condition_segment == "Proc5_SO2_O2_cycles_1st_SO2"
    assert identity.group_id == ("High-Cu_cycles_SO2_O2/Proc5_SO2_O2_cycles_1st_SO2")
    assert identity.cu_loading == "High-Cu"
    assert identity.protocol_family == "cycles"
    assert identity.scan_index == 100
    assert identity.temperature_c == 199


@pytest.mark.parametrize(
    "relative_path",
    [
        "/data_txt/High-Cu_cycles/a.dat",
        "data_txt\\High-Cu_cycles\\a.dat",
        "data_txt/../High-Cu_cycles/a.dat",
        "data_txt/Other/100_at_199C_Proc5.dat",
        "data_txt/High-Cu_unknown_SO2/100_at_199C_Proc5.dat",
        "data_txt/Low-Cu_cycles/bad-name.dat",
    ],
)
def test_relative_path_identity_fails_closed(relative_path: str) -> None:
    with pytest.raises(ValueError):
        parse_cu_cha_identity(relative_path)


def _identity(
    sample_id: str,
    group_id: str,
    *,
    cu_loading: str,
    protocol_family: str,
) -> CuChaSpectrumIdentity:
    return CuChaSpectrumIdentity(
        sample_id=sample_id,
        experiment=group_id.split("/", maxsplit=1)[0],
        condition_segment=group_id.split("/", maxsplit=1)[1],
        group_id=group_id,
        cu_loading=cu_loading,
        protocol_family=protocol_family,
        scan_index=1,
        temperature_c=200,
    )


def test_grouped_balanced_split_is_order_independent_and_leakage_safe() -> None:
    identities = tuple(
        _identity(
            f"sample-{loading}-{protocol}-{group}-{index}",
            f"{loading}_{protocol}_experiment/{group}",
            cu_loading=loading,
            protocol_family=protocol,
        )
        for loading, protocol in (("High-Cu", "exposure"), ("Low-Cu", "cycles"))
        for group, size in (("a", 7), ("b", 5), ("c", 4), ("d", 3))
        for index in range(size)
    )

    first = build_grouped_balanced_split(identities)
    second = build_grouped_balanced_split(tuple(reversed(identities)))

    assert first.digest == second.digest
    assert first.entries == second.entries
    assert {entry.split for entry in first.entries} == {"train", "val", "test"}
    group_splits: dict[str, set[str]] = {}
    for entry in first.entries:
        group_splits.setdefault(entry.group_id, set()).add(entry.split)
    assert all(len(splits) == 1 for splits in group_splits.values())
    identity_by_sample = {identity.sample_id: identity for identity in identities}
    stratum_splits: dict[tuple[str, str], set[str]] = {}
    for entry in first.entries:
        identity = identity_by_sample[entry.sample_id]
        stratum_splits.setdefault(
            (identity.cu_loading, identity.protocol_family), set()
        ).add(entry.split)
    assert all(splits == {"train", "val", "test"} for splits in stratum_splits.values())


def test_grouped_balanced_split_requires_three_groups_and_unique_samples() -> None:
    with pytest.raises(ValueError, match="at least three"):
        build_grouped_balanced_split(
            (
                _identity(
                    "sample-a",
                    "High-Cu_exposure_exp/a",
                    cu_loading="High-Cu",
                    protocol_family="exposure",
                ),
                _identity(
                    "sample-b",
                    "High-Cu_exposure_exp/b",
                    cu_loading="High-Cu",
                    protocol_family="exposure",
                ),
            )
        )
    duplicate = _identity(
        "duplicate",
        "High-Cu_exposure_exp/a",
        cu_loading="High-Cu",
        protocol_family="exposure",
    )
    with pytest.raises(ValueError, match="unique"):
        build_grouped_balanced_split(
            (
                duplicate,
                duplicate,
                _identity(
                    "b",
                    "High-Cu_exposure_exp/b",
                    cu_loading="High-Cu",
                    protocol_family="exposure",
                ),
                _identity(
                    "c",
                    "High-Cu_exposure_exp/c",
                    cu_loading="High-Cu",
                    protocol_family="exposure",
                ),
            )
        )


def test_grouped_balanced_split_rejects_one_undercovered_stratum() -> None:
    identities = tuple(
        _identity(
            f"sample-{protocol}-{group}",
            f"High-Cu_{protocol}_experiment/{group}",
            cu_loading="High-Cu",
            protocol_family=protocol,
        )
        for protocol, groups in (("exposure", ("a", "b", "c")), ("cycles", ("d", "e")))
        for group in groups
    )

    with pytest.raises(ValueError, match="stratum.*at least three"):
        build_grouped_balanced_split(identities)


def test_seed_binds_every_reproducibility_input() -> None:
    arguments = {
        "dataset_version": "synthetic-version-id",
        "source_sha256": "a" * 64,
        "source_path": "data_txt/High-Cu_exp/1_at_200C_segment.dat",
        "dose_fraction": 0.25,
        "contract_version": "hyperspectrum-cu-cha-poisson/v1",
        "numpy_version": "2.4.6",
    }
    baseline = derive_cu_cha_seed(**arguments)

    assert baseline == derive_cu_cha_seed(**arguments)
    assert 0 <= baseline < 2**64
    replacements = (
        ("dataset_version", "another-version"),
        ("source_sha256", "b" * 64),
        ("source_path", "data_txt/High-Cu_exp/2_at_200C_segment.dat"),
        ("dose_fraction", 0.50),
        ("contract_version", "hyperspectrum-cu-cha-poisson/v2"),
    )
    for name, value in replacements:
        changed = dict(arguments)
        changed[name] = value
        assert derive_cu_cha_seed(**changed) != baseline

    incompatible = dict(arguments)
    incompatible["numpy_version"] = "2.4.7"
    with pytest.raises(ValueError, match="official NumPy Poisson ABI.*2.4.6"):
        derive_cu_cha_seed(**incompatible)


def test_poisson_thinning_is_exactly_reproducible_and_uses_both_channels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scan.dat"
    source.write_bytes(_dat_bytes())
    spectrum = parse_cu_cha_dat(source)
    seed = 123456789

    first = poisson_thin_transmission(
        spectrum,
        dose_fraction=0.25,
        seed=seed,
        expected_numpy_version="2.4.6",
    )
    second = poisson_thin_transmission(
        spectrum,
        dose_fraction=0.25,
        seed=seed,
        expected_numpy_version="2.4.6",
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    expected_i0 = generator.poisson(0.25 * spectrum.i0).astype(np.float64) / 0.25
    expected_i1 = generator.poisson(0.25 * spectrum.i1).astype(np.float64) / 0.25

    np.testing.assert_array_equal(first.noisy_i0, second.noisy_i0)
    np.testing.assert_array_equal(first.noisy_i1, second.noisy_i1)
    np.testing.assert_array_equal(first.noisy_i0, expected_i0)
    np.testing.assert_array_equal(first.noisy_i1, expected_i1)
    np.testing.assert_allclose(first.noisy_mu_trans, np.log(expected_i0 / expected_i1))
    assert first.noisy_mu_trans.flags.writeable is False
    assert first.dose_fraction == 0.25
    assert first.seed == seed


@pytest.mark.skipif(
    np.__version__ != "2.4.6",
    reason="this golden vector owns the official NumPy 2.4.6 Poisson ABI",
)
def test_numpy_246_pcg64_poisson_distribution_golden_vector() -> None:
    # Break caught: Generator.poisson is not covered by NumPy's compatibility promise.
    i0 = np.array([10, 1_000, 25_000, 100_000, 250_000], dtype=np.float64)
    i1 = np.array([20, 2_000, 50_000, 200_000, 500_000], dtype=np.float64)
    spectrum = ParsedCuChaSpectrum(
        energy_ev=np.arange(5, dtype=np.float64) + 1.0,
        mu_trans=np.log(i0 / i1),
        mu_ref=np.zeros(5, dtype=np.float64),
        i0=i0,
        i1=i1,
        i2=np.ones(5, dtype=np.float64),
    )

    thinned = poisson_thin_transmission(
        spectrum,
        dose_fraction=1.0,
        seed=0xC0DEC0DE,
        expected_numpy_version="2.4.6",
    )

    np.testing.assert_array_equal(thinned.noisy_i0, [8, 987, 24_795, 100_193, 250_603])
    np.testing.assert_array_equal(
        thinned.noisy_i1, [21, 1_981, 50_296, 200_749, 500_595]
    )


def test_poisson_thinning_fails_closed_on_numpy_version_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scan.dat"
    source.write_bytes(_dat_bytes())
    spectrum = parse_cu_cha_dat(source)

    with pytest.raises(ValueError, match="exact NumPy runtime version"):
        poisson_thin_transmission(
            spectrum,
            dose_fraction=0.25,
            seed=123456789,
            expected_numpy_version="2.4.7",
        )


def test_poisson_thinning_rejects_zero_counts_without_epsilon_correction() -> None:
    low_count = np.ones(5, dtype=np.float64)
    spectrum = ParsedCuChaSpectrum(
        energy_ev=np.arange(5, dtype=np.float64) + 1.0,
        mu_trans=np.zeros(5, dtype=np.float64),
        mu_ref=np.zeros(5, dtype=np.float64),
        i0=low_count,
        i1=low_count,
        i2=low_count,
    )

    with pytest.raises(SpectrumRejected) as error:
        poisson_thin_transmission(
            spectrum,
            dose_fraction=0.10,
            seed=0,
            expected_numpy_version="2.4.6",
        )

    assert error.value.code == "zero_thinned_count"


def _source_fixture(root: Path, *, reverse_declaration: bool = False) -> Path:
    declarations: list[dict[str, object]] = []
    source_groups = (
        (loading, protocol, group)
        for loading in ("High-Cu", "Low-Cu")
        for protocol in ("exposure", "cycles")
        for group in ("alpha", "beta", "gamma")
    )
    for group_index, (loading, protocol, group) in enumerate(source_groups):
        relative = (
            f"data_txt/{loading}_{protocol}_{group}/"
            f"{group_index + 1}_at_200C_Proc_{loading}_{protocol}_{group}.dat"
        )
        content = _dat_bytes(
            mu_trans=np.array([0.01, 0.03, 0.08, 0.15, 0.22]) + group_index * 0.001,
        )
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
        declarations.append(
            {
                "path": relative,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    declaration = {
        "schema_version": "hyperspectrum-cu-cha-source-declaration/v1",
        "dataset": {
            "code": "zenodo-10159154",
            "version": "v1",
            "version_id": "synthetic-version-id",
        },
        "files": list(reversed(declarations)) if reverse_declaration else declarations,
    }
    declaration_file = root.parent / f"declaration-{reverse_declaration}.json"
    declaration_file.write_text(json.dumps(declaration), encoding="utf-8")
    return declaration_file


def _all_output_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _profile_sha256(root: Path) -> str:
    return hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


def test_materializer_builds_three_honest_tracks_with_shared_split(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    declaration = _source_fixture(source_root)
    output = tmp_path / "output"

    result = materialize_cu_cha(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=output,
    )

    root_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["schema_version"] == "hyperspectrum-xas-denoising-profile/v2"
    assert root_manifest["dataset"]["code"] == "zenodo-10159154"
    assert root_manifest["target"] == {
        "ground_truth_kind": "proxy_full_count",
        "is_physical_noiseless_ground_truth": False,
        "signal": "mu_trans_full_count",
    }
    assert root_manifest["compatibility"]["ace_v3"] == "requires_profile_v2_adapter"
    assert root_manifest["execution_boundary"] == {
        "candidate_container_asset": "inference_only_test_split",
        "leaderboard_scored_split": "test",
        "scorer_only_asset": "benchmark_with_proxy_target",
    }
    assert root_manifest["corruption_runtime"] == {
        "schema_version": "hyperspectrum-numpy-poisson-abi/v1",
        "numpy_version": "2.4.6",
        "bit_generator": "numpy.random.PCG64",
        "distribution": "numpy.random.Generator.poisson",
        "draw_order": ["I0", "I1"],
    }
    assert [track["dose_fraction"] for track in root_manifest["tracks"]] == [
        0.10,
        0.25,
        0.50,
    ]
    assert root_manifest["split"]["strata"] == ["cu_loading", "protocol_family"]
    assert len(root_manifest["split"]["stratum_counts"]) == 4
    for summary in root_manifest["split"]["stratum_counts"]:
        assert summary["group_count"] == 3
        assert summary["sample_count"] == 3
        assert summary["split_group_counts"] == {"train": 1, "val": 1, "test": 1}
        assert summary["split_sample_counts"] == {"train": 1, "val": 1, "test": 1}
    assert len(result.tracks) == 3
    assert result.profile_sha256 == _profile_sha256(output)
    assert result.to_dict()["schema_version"] == (
        "hyperspectrum-cu-cha-materialization-result/v2"
    )
    for track in result.tracks:
        track_manifest = json.loads(track.manifest_path.read_text(encoding="utf-8"))
        assert (
            track_manifest["corruption"]["distribution_abi"]
            == root_manifest["corruption_runtime"]
        )

    pairs_by_dose = {
        dose: load_cu_cha_denoising_pairs(
            output,
            dose_fraction=dose,
            expected_profile_sha256=_profile_sha256(output),
        )
        for dose in (0.10, 0.25, 0.50)
    }
    for pairs in pairs_by_dose.values():
        assert len(pairs) == 12
        assert {pair.split for pair in pairs} == {"train", "val", "test"}
        assert all(
            pair.clean.provenance["ground_truth_kind"] == "proxy_full_count"
            and pair.clean.provenance["is_physical_noiseless_ground_truth"] is False
            for pair in pairs
        )
    for index in range(12):
        np.testing.assert_array_equal(
            pairs_by_dose[0.10][index].clean.signal,
            pairs_by_dose[0.50][index].clean.signal,
        )
    assert any(
        not np.array_equal(low.noisy.signal, high.noisy.signal)
        for low, high in zip(pairs_by_dose[0.10], pairs_by_dose[0.50], strict=True)
    )
    with np.load(
        output / "tracks/dose-0.10/inference.npz", allow_pickle=False
    ) as asset:
        assert set(asset.files) == {
            "energy",
            "energy_unit",
            "group_ids",
            "noisy",
            "sample_ids",
        }


def _split_by_sample(output: Path) -> dict[str, str]:
    payload = json.loads((output / "split.json").read_text(encoding="utf-8"))
    return {entry["sample_id"]: entry["split"] for entry in payload["entries"]}


def test_candidate_inference_asset_carries_only_the_scored_test_rows(
    tmp_path: Path,
) -> None:
    # Break caught: an inference asset carrying train/val rows hands a candidate
    # container rows it is not asked to answer, and every prediction it makes for
    # them is split leakage the scorer must refuse. The candidate cannot filter —
    # the split is a scorer-only asset — so the filtering has to happen here.
    source_root = tmp_path / "source"
    declaration = _source_fixture(source_root)
    output = tmp_path / "output"

    materialize_cu_cha(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=output,
    )

    splits = _split_by_sample(output)
    expected_test_ids = sorted(
        sample_id for sample_id, split in splits.items() if split == "test"
    )
    assert 0 < len(expected_test_ids) < len(splits)

    for track in ("dose-0.10", "dose-0.25", "dose-0.50"):
        track_directory = output / "tracks" / track
        with np.load(track_directory / "inference.npz", allow_pickle=False) as asset:
            inference = {
                name: np.array(asset[name], copy=True) for name in asset.files
            }
        with np.load(track_directory / "benchmark.npz", allow_pickle=False) as asset:
            benchmark = {
                name: np.array(asset[name], copy=True) for name in asset.files
            }

        assert inference["sample_ids"].tolist() == expected_test_ids
        assert not {
            str(value)
            for value in inference["sample_ids"]
        } & {sample_id for sample_id, split in splits.items() if split != "test"}

        rows = [
            index
            for index, sample_id in enumerate(benchmark["sample_ids"].tolist())
            if splits[str(sample_id)] == "test"
        ]
        np.testing.assert_array_equal(inference["energy"], benchmark["energy"][rows])
        np.testing.assert_array_equal(inference["noisy"], benchmark["noisy"][rows])
        assert inference["group_ids"].tolist() == benchmark["group_ids"][rows].tolist()
        assert str(inference["energy_unit"]) == "eV"

        # The scorer half keeps every row: only the candidate's view narrows.
        assert len(benchmark["sample_ids"]) == len(splits)

        track_manifest = json.loads(
            (track_directory / "manifest.json").read_text(encoding="utf-8")
        )
        assert track_manifest["inference_asset"]["contract"] == (
            "hyperspectrum-local-inference-npz/v2"
        )


def test_loader_rejects_an_inference_asset_that_readmits_untested_rows(
    tmp_path: Path,
) -> None:
    # Break caught: a re-materialization that quietly went back to every row would
    # otherwise differ from a compliant one only by digest, and a digest tells an
    # operator that something moved, not that leakage came back.
    source_root = tmp_path / "source"
    declaration = _source_fixture(source_root)
    output = tmp_path / "output"
    materialize_cu_cha(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=output,
    )
    track_directory = output / "tracks/dose-0.25"
    with np.load(track_directory / "benchmark.npz", allow_pickle=False) as loaded:
        benchmark = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    inference_path = track_directory / "inference.npz"
    np.savez(
        inference_path,
        energy=benchmark["energy"],
        energy_unit=np.asarray("eV"),
        group_ids=benchmark["group_ids"],
        noisy=benchmark["noisy"],
        sample_ids=benchmark["sample_ids"],
    )
    inference_sha256 = hashlib.sha256(inference_path.read_bytes()).hexdigest()
    track_manifest_path = track_directory / "manifest.json"
    track_manifest = json.loads(track_manifest_path.read_text(encoding="utf-8"))
    track_manifest["inference_asset"]["sha256"] = inference_sha256
    track_manifest_path.write_text(json.dumps(track_manifest), encoding="utf-8")
    root_path = output / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["tracks"][1]["inference_asset_sha256"] = inference_sha256
    root_path.write_text(json.dumps(root), encoding="utf-8")

    with pytest.raises(ValueError, match="inference.*test"):
        load_cu_cha_denoising_pairs(
            output,
            dose_fraction=0.25,
            expected_profile_sha256=_profile_sha256(output),
        )


def test_materializer_fails_closed_outside_official_numpy_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    declaration = _source_fixture(source_root)
    output = tmp_path / "output"
    monkeypatch.setattr(np, "__version__", "2.4.7")

    with pytest.raises(ValueError, match="official NumPy Poisson ABI.*2.4.6"):
        materialize_cu_cha(
            source_root=source_root,
            source_declaration_file=declaration,
            output_directory=output,
        )

    assert not output.exists()


def test_materialization_is_byte_deterministic_across_declaration_order(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    first_declaration = _source_fixture(source_root)
    second_declaration = _source_fixture(source_root, reverse_declaration=True)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    materialize_cu_cha(
        source_root=source_root,
        source_declaration_file=first_declaration,
        output_directory=first_output,
    )
    materialize_cu_cha(
        source_root=source_root,
        source_declaration_file=second_declaration,
        output_directory=second_output,
    )

    assert _all_output_bytes(first_output) == _all_output_bytes(second_output)


@pytest.mark.parametrize("drift", ["size", "sha256"])
def test_source_integrity_fails_before_creating_output(
    tmp_path: Path, drift: str
) -> None:
    source_root = tmp_path / "source"
    declaration_file = _source_fixture(source_root)
    declaration = json.loads(declaration_file.read_text(encoding="utf-8"))
    declaration["files"][0][drift] = 0 if drift == "size" else "0" * 64
    declaration_file.write_text(json.dumps(declaration), encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="integrity"):
        materialize_cu_cha(
            source_root=source_root,
            source_declaration_file=declaration_file,
            output_directory=output,
        )

    assert not output.exists()


def test_source_integrity_rejects_symlink_even_when_declared_bytes_match(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    declaration_file = _source_fixture(source_root)
    declaration = json.loads(declaration_file.read_text(encoding="utf-8"))
    linked_source = source_root / declaration["files"][0]["path"]
    outside = tmp_path / "outside.dat"
    outside.write_bytes(linked_source.read_bytes())
    linked_source.unlink()
    linked_source.symlink_to(outside)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="regular files"):
        materialize_cu_cha(
            source_root=source_root,
            source_declaration_file=declaration_file,
            output_directory=output,
        )

    assert not output.exists()


def test_source_integrity_rejects_symlink_swap_between_scan_and_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root = tmp_path / "source"
    declaration_file = _source_fixture(source_root)
    declaration = json.loads(declaration_file.read_text(encoding="utf-8"))
    target = source_root / declaration["files"][0]["path"]
    outside = tmp_path / "outside-secret.dat"
    outside.write_bytes(b"must-not-be-read")
    original_open = os.open
    swapped = False

    def racing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if dir_fd is not None and os.fspath(path) == target.name and not swapped:
            target.unlink()
            target.symlink_to(outside)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="secure no-follow"):
        materialize_cu_cha(
            source_root=source_root,
            source_declaration_file=declaration_file,
            output_directory=output,
        )

    assert swapped is True
    assert not output.exists()


def test_output_is_atomically_cleaned_when_bundle_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root = tmp_path / "source"
    declaration_file = _source_fixture(source_root)
    output = tmp_path / "output"
    original_write_bytes = Path.write_bytes

    def fail_on_inference(path: Path, data: bytes) -> int:
        if path.name == "inference.npz":
            raise OSError("synthetic disk failure")
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_on_inference)

    with pytest.raises(OSError, match="synthetic disk failure"):
        materialize_cu_cha(
            source_root=source_root,
            source_declaration_file=declaration_file,
            output_directory=output,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".output.tmp-*")) == []


def test_loader_rejects_tampered_benchmark_asset(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    declaration = _source_fixture(source_root)
    output = tmp_path / "output"
    materialize_cu_cha(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=output,
    )
    asset = output / "tracks/dose-0.25/benchmark.npz"
    asset.write_bytes(asset.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="SHA-256"):
        load_cu_cha_denoising_pairs(
            output,
            dose_fraction=0.25,
            expected_profile_sha256=_profile_sha256(output),
        )


def test_loader_requires_external_profile_trust_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    declaration = _source_fixture(source_root)
    output = tmp_path / "output"
    materialize_cu_cha(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=output,
    )
    trusted_sha256 = _profile_sha256(output)
    root_path = output / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["tracks"][0]["benchmark_asset_sha256"] = "0" * 64
    root_path.write_text(json.dumps(root), encoding="utf-8")

    with pytest.raises(ValueError, match="profile manifest SHA-256"):
        load_cu_cha_denoising_pairs(
            output,
            dose_fraction=0.10,
            expected_profile_sha256=trusted_sha256,
        )


def test_loader_fails_closed_before_poisson_regeneration_on_numpy_abi_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    declaration = _source_fixture(source_root)
    output = tmp_path / "output"
    materialize_cu_cha(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=output,
    )
    trusted_sha256 = _profile_sha256(output)
    monkeypatch.setattr(np, "__version__", f"{np.__version__}+different")

    with pytest.raises(ValueError, match="exact NumPy runtime version"):
        load_cu_cha_denoising_pairs(
            output,
            dose_fraction=0.25,
            expected_profile_sha256=trusted_sha256,
        )


def test_loader_rejects_self_consistent_scientific_benchmark_tamper(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    declaration = _source_fixture(source_root)
    output = tmp_path / "output"
    materialize_cu_cha(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=output,
    )
    benchmark_path = output / "tracks/dose-0.25/benchmark.npz"
    with np.load(benchmark_path, allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    arrays["proxy_full_count"][0, 0] += 1.0
    np.savez(benchmark_path, **arrays)
    benchmark_sha256 = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    track_manifest_path = output / "tracks/dose-0.25/manifest.json"
    track_manifest = json.loads(track_manifest_path.read_text(encoding="utf-8"))
    track_manifest["benchmark_asset"]["sha256"] = benchmark_sha256
    track_manifest_path.write_text(json.dumps(track_manifest), encoding="utf-8")
    root_path = output / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["tracks"][1]["benchmark_asset_sha256"] = benchmark_sha256
    root_path.write_text(json.dumps(root), encoding="utf-8")

    with pytest.raises(ValueError, match="full-count proxy"):
        load_cu_cha_denoising_pairs(
            output,
            dose_fraction=0.25,
            expected_profile_sha256=_profile_sha256(output),
        )


def test_loader_rejects_self_consistent_inference_benchmark_mismatch(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    declaration = _source_fixture(source_root)
    output = tmp_path / "output"
    materialize_cu_cha(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=output,
    )
    inference_path = output / "tracks/dose-0.25/inference.npz"
    with np.load(inference_path, allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    arrays["noisy"][0, 0] += 1.0
    np.savez(inference_path, **arrays)
    inference_sha256 = hashlib.sha256(inference_path.read_bytes()).hexdigest()
    track_manifest_path = output / "tracks/dose-0.25/manifest.json"
    track_manifest = json.loads(track_manifest_path.read_text(encoding="utf-8"))
    track_manifest["inference_asset"]["sha256"] = inference_sha256
    track_manifest_path.write_text(json.dumps(track_manifest), encoding="utf-8")
    root_path = output / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["tracks"][1]["inference_asset_sha256"] = inference_sha256
    root_path.write_text(json.dumps(root), encoding="utf-8")

    with pytest.raises(ValueError, match="inference.*benchmark"):
        load_cu_cha_denoising_pairs(
            output,
            dose_fraction=0.25,
            expected_profile_sha256=_profile_sha256(output),
        )


def test_loader_rejects_self_consistent_but_nonreproducible_poisson_counts(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    declaration = _source_fixture(source_root)
    output = tmp_path / "output"
    materialize_cu_cha(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=output,
    )
    benchmark_path = output / "tracks/dose-0.25/benchmark.npz"
    with np.load(benchmark_path, allow_pickle=False) as loaded:
        benchmark = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    benchmark["noisy_i0_counts"][0, 0] += 4.0
    benchmark["noisy"][0, 0] = np.log(
        benchmark["noisy_i0_counts"][0, 0] / benchmark["noisy_i1_counts"][0, 0]
    )
    np.savez(benchmark_path, **benchmark)
    inference_path = output / "tracks/dose-0.25/inference.npz"
    with np.load(inference_path, allow_pickle=False) as loaded:
        inference = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    inference["noisy"][0, 0] = benchmark["noisy"][0, 0]
    np.savez(inference_path, **inference)
    benchmark_sha256 = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    inference_sha256 = hashlib.sha256(inference_path.read_bytes()).hexdigest()
    track_manifest_path = output / "tracks/dose-0.25/manifest.json"
    track_manifest = json.loads(track_manifest_path.read_text(encoding="utf-8"))
    track_manifest["benchmark_asset"]["sha256"] = benchmark_sha256
    track_manifest["inference_asset"]["sha256"] = inference_sha256
    track_manifest_path.write_text(json.dumps(track_manifest), encoding="utf-8")
    root_path = output / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["tracks"][1]["benchmark_asset_sha256"] = benchmark_sha256
    root["tracks"][1]["inference_asset_sha256"] = inference_sha256
    root_path.write_text(json.dumps(root), encoding="utf-8")

    with pytest.raises(ValueError, match="Poisson draws"):
        load_cu_cha_denoising_pairs(
            output,
            dose_fraction=0.25,
            expected_profile_sha256=_profile_sha256(output),
        )


def test_loader_rejects_tampered_track_manifest_and_inference_asset(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    declaration = _source_fixture(source_root)
    output = tmp_path / "output"
    materialize_cu_cha(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=output,
    )
    track_manifest_path = output / "tracks/dose-0.25/manifest.json"
    track_manifest = json.loads(track_manifest_path.read_text(encoding="utf-8"))
    track_manifest["target"]["ground_truth_kind"] = "clean"
    track_manifest_path.write_text(json.dumps(track_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="track manifest"):
        load_cu_cha_denoising_pairs(
            output,
            dose_fraction=0.25,
            expected_profile_sha256=_profile_sha256(output),
        )

    # Restore the manifest, then prove the target-free execution input is bound too.
    track_manifest["target"]["ground_truth_kind"] = "proxy_full_count"
    track_manifest_path.write_text(json.dumps(track_manifest), encoding="utf-8")
    inference = output / "tracks/dose-0.25/inference.npz"
    inference.write_bytes(inference.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="inference asset SHA-256"):
        load_cu_cha_denoising_pairs(
            output,
            dose_fraction=0.25,
            expected_profile_sha256=_profile_sha256(output),
        )


@pytest.mark.parametrize("field", ["dataset", "source", "split"])
def test_loader_rejects_malformed_root_manifest_objects(
    tmp_path: Path, field: str
) -> None:
    source_root = tmp_path / "source"
    declaration = _source_fixture(source_root)
    output = tmp_path / "output"
    materialize_cu_cha(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=output,
    )
    root_manifest_path = output / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    root_manifest[field] = []
    root_manifest_path.write_text(json.dumps(root_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="profile manifest"):
        load_cu_cha_denoising_pairs(
            output,
            dose_fraction=0.25,
            expected_profile_sha256=_profile_sha256(output),
        )
