from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hyperspectrum.datasets import xanes_spec
from hyperspectrum.datasets.xanes_spec import (
    SourceIntegrityError,
    SpectrumRejected,
    load_benchmark_asset_identity,
    load_denoising_pairs,
    materialize_xanes_spec,
    parse_xanes_spec,
)

ROOT = Path(__file__).parents[2]
REAL_HEADER = (
    ROOT / "tests" / "fixtures" / "xas" / "spec-real-header.txt"
).read_text(encoding="utf-8")


def _spec_bytes(
    *,
    rows: int = 135,
    energy: np.ndarray | None = None,
    i0: float = 100_000.0,
    ketek: float = 1_000.0,
    scans: int = 1,
) -> bytes:
    if energy is None:
        fraction = np.linspace(0.0, 1.0, rows)
        energy = 5692.994 + (5801.399 - 5692.994) * fraction**1.4
    lines: list[str] = []
    for scan in range(scans):
        header = REAL_HEADER.replace("#S 1", f"#S {scan + 1}")
        lines.extend(header.rstrip().splitlines())
        for index, value in enumerate(energy):
            row = (
                value,
                float(index),
                1.0,
                i0 + index,
                i0 + index - 0.1,
                50.0,
                50.0,
                1.0,
                0.0,
                ketek + index,
                1e-12,
                0.0,
                20.0,
                20.0,
                0.0,
                0.0,
            )
            lines.append(" ".join(f"{item:.12g}" for item in row))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_tree(tmp_path: Path, files: dict[str, bytes]) -> Path:
    source_root = tmp_path / "source"
    source_root.mkdir()
    for relative_path, content in files.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return source_root


def _write_declaration(
    tmp_path: Path,
    *,
    source_root: Path,
    declared_sizes: dict[str, int] | None = None,
    declared_sha256: dict[str, str] | None = None,
    reverse_files: bool = False,
    upstream_revision: str | None = None,
) -> Path:
    paths = sorted(
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    )
    if reverse_files:
        paths.reverse()
    entries = []
    for relative_path in paths:
        content = (source_root / relative_path).read_bytes()
        entries.append(
            {
                "path": relative_path,
                "size": (
                    len(content)
                    if declared_sizes is None
                    else declared_sizes.get(relative_path, len(content))
                ),
                "sha256": (
                    hashlib.sha256(content).hexdigest()
                    if declared_sha256 is None
                    else declared_sha256.get(
                        relative_path, hashlib.sha256(content).hexdigest()
                    )
                ),
            }
        )
    declaration = {
        "schema_version": "hyperspectrum-xanes-source-declaration/v1",
        "dataset": {
            "id": "zenodo-10606662",
            "code": "zenodo-10606662",
            "version": "1.0.0",
            "upstream_revision": upstream_revision,
            "declared_manifest_digest": "a" * 64,
        },
        "expected_file_count": len(entries),
        "files": entries,
    }
    path = tmp_path / ("source-declaration-reversed.json" if reverse_files else "source-declaration.json")
    path.write_text(json.dumps(declaration), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_real_spec_header_and_nonuniform_grid_parse_strictly(tmp_path: Path) -> None:
    path = tmp_path / "Exp2_La05_heating_500_000_000.dat"
    path.write_bytes(_spec_bytes())

    spectrum = parse_xanes_spec(path)

    assert spectrum.columns == (
        "energy",
        "Epoch",
        "Seconds",
        "i0",
        "i0s",
        "xicr",
        "xocr",
        "xlt",
        "xdt",
        "ketek",
        "pico",
        "keteki0",
        "xes_temp1",
        "xes_temp2",
        "Monitor",
        "Detector",
    )
    assert spectrum.energy.shape == (135,)
    assert np.all(np.diff(spectrum.energy) > 0.0)
    assert not np.allclose(np.diff(spectrum.energy), np.diff(spectrum.energy)[0])
    assert np.all(spectrum.i0 > 0.0)
    assert np.all(spectrum.ketek >= 0.0)


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (b"", "scan_count"),
        (_spec_bytes(rows=12), "row_count"),
        (_spec_bytes(scans=2), "scan_count"),
    ],
)
def test_empty_short_and_multiple_scan_spectra_fail_closed(
    tmp_path: Path, content: bytes, code: str
) -> None:
    path = tmp_path / "Exp2_La05_heating_500_000_000.dat"
    path.write_bytes(content)

    with pytest.raises(SpectrumRejected) as error:
        parse_xanes_spec(path)

    assert error.value.code == code


@pytest.mark.parametrize(
    ("mutator", "code"),
    [
        (lambda energy: energy[::-1], "energy_not_increasing"),
        (lambda energy: energy + 100.0, "energy_coverage"),
        (
            lambda energy: np.linspace(5600.0, 5900.0, len(energy)),
            "energy_coverage",
        ),
    ],
)
def test_invalid_energy_fails_closed(
    tmp_path: Path, mutator: Any, code: str
) -> None:
    energy = 5692.994 + (5801.399 - 5692.994) * np.linspace(0.0, 1.0, 135) ** 1.4
    path = tmp_path / "Exp2_La05_heating_500_000_000.dat"
    path.write_bytes(_spec_bytes(energy=mutator(energy)))

    with pytest.raises(SpectrumRejected) as error:
        parse_xanes_spec(path)

    assert error.value.code == code


def test_size_metadata_drift_is_reported_but_sha_matching_file_is_admitted(
    tmp_path: Path,
) -> None:
    files = {
        "Exp2_La05_heating_500_000_000.dat": _spec_bytes(),
        "experimental_conditions.txt": b"Experiment 2: La05\n",
        "Powder_La08": b"#S 1 ascan motor 0 1 10 1\n",
    }
    source_root = _write_tree(tmp_path, files)
    declaration = _write_declaration(
        tmp_path,
        source_root=source_root,
        declared_sizes={"Exp2_La05_heating_500_000_000.dat": 1},
    )

    result = materialize_xanes_spec(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=tmp_path / "bundle",
        global_seed=7,
    )

    qc = _read_json(result.qc_path)
    assert qc["size_metadata_drift_count"] == 1
    by_path = {entry["path"]: entry for entry in qc["files"]}
    assert by_path["Exp2_La05_heating_500_000_000.dat"]["status"] == "admitted"
    assert by_path["Powder_La08"] == {
        "actual_size": len(files["Powder_La08"]),
        "declared_size": len(files["Powder_La08"]),
        "path": "Powder_La08",
        "reason_code": "extensionless_motor_scan",
        "sha256": hashlib.sha256(files["Powder_La08"]).hexdigest(),
        "size_matches": True,
        "status": "excluded",
    }
    assert by_path["experimental_conditions.txt"]["reason_code"] == (
        "conditions_sidecar"
    )


def test_sha_mismatch_refuses_entire_materialization(tmp_path: Path) -> None:
    relative_path = "Exp2_La05_heating_500_000_000.dat"
    source_root = _write_tree(tmp_path, {relative_path: _spec_bytes()})
    declaration = _write_declaration(
        tmp_path,
        source_root=source_root,
        declared_sha256={relative_path: "f" * 64},
    )

    with pytest.raises(SourceIntegrityError, match="SHA-256"):
        materialize_xanes_spec(
            source_root=source_root,
            source_declaration_file=declaration,
            output_directory=tmp_path / "bundle",
        )

    assert not (tmp_path / "bundle").exists()


def test_materializer_parses_the_exact_byte_snapshot_that_was_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative_path = "Exp2_La05_heating_500_000_000.dat"
    original = _spec_bytes(ketek=1_000.0)
    replacement = _spec_bytes(ketek=9_000.0)
    source_root = _write_tree(tmp_path, {relative_path: original})
    declaration = _write_declaration(tmp_path, source_root=source_root)
    original_verify = xanes_spec._verify_source_tree

    def replace_after_verification(
        root: Path, source_declaration: xanes_spec.SourceDeclaration
    ) -> tuple[xanes_spec._VerifiedFile, ...]:
        verified = original_verify(root, source_declaration)
        (root / relative_path).write_bytes(replacement)
        return verified

    monkeypatch.setattr(xanes_spec, "_verify_source_tree", replace_after_verification)

    result = materialize_xanes_spec(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=tmp_path / "bundle",
    )

    with np.load(result.asset_path, allow_pickle=False) as bundle:
        assert bundle["pseudo_clean"][0, 0] == pytest.approx(0.01)
    assert hashlib.sha256(original).hexdigest() in result.manifest_path.read_text()


def test_short_and_empty_spectra_enter_qc_without_entering_bundle(
    tmp_path: Path,
) -> None:
    files = {
        "Exp2_La05_heating_500_000_000.dat": _spec_bytes(),
        "Exp2_La05_heating_500_000_001.dat": _spec_bytes(rows=4),
        "Exp3_La05_heating_800_000_010.dat": b"",
    }
    source_root = _write_tree(tmp_path, files)
    declaration = _write_declaration(tmp_path, source_root=source_root)

    result = materialize_xanes_spec(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=tmp_path / "bundle",
    )

    qc = _read_json(result.qc_path)
    rejected = {
        entry["path"]: entry["reason_code"]
        for entry in qc["files"]
        if entry["status"] == "rejected"
    }
    assert rejected == {
        "Exp2_La05_heating_500_000_001.dat": "row_count",
        "Exp3_La05_heating_800_000_010.dat": "scan_count",
    }
    with np.load(result.asset_path, allow_pickle=False) as bundle:
        assert bundle["sample_ids"].tolist() == [
            "Exp2_La05_heating_500_000_000"
        ]


def test_fixed_composition_groups_prevent_sequence_leakage(tmp_path: Path) -> None:
    files = {
        "Exp2_La05_heating_500_000_000.dat": _spec_bytes(ketek=1_100.0),
        "Exp3_La05_heating_800_000_000.dat": _spec_bytes(ketek=1_200.0),
        "Powder_La05_no_heat_000_000.dat": _spec_bytes(ketek=1_300.0),
        "Exp4_La08_heating_800_000_000.dat": _spec_bytes(ketek=1_400.0),
        "Exp9_La08_heating_800_000_000.dat": _spec_bytes(ketek=1_500.0),
        "Powder_La08_no_heat_000_000.dat": _spec_bytes(ketek=1_600.0),
        "Exp8_Gd05_heating_800_000_000.dat": _spec_bytes(ketek=1_700.0),
        "Powder_Gd05_no_heat_000_000.dat": _spec_bytes(ketek=1_800.0),
    }
    source_root = _write_tree(tmp_path, files)
    declaration = _write_declaration(tmp_path, source_root=source_root)

    result = materialize_xanes_spec(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=tmp_path / "bundle",
    )
    split = _read_json(result.split_path)

    composition_splits: dict[str, set[str]] = {}
    for entry in split["entries"]:
        composition_splits.setdefault(entry["group_id"], set()).add(entry["split"])
    assert composition_splits == {
        "La05": {"train"},
        "La08": {"test"},
        "Gd05": {"val"},
    }
    assert all(len(splits) == 1 for splits in composition_splits.values())


def test_reference_grid_is_first_sorted_valid_train_not_test_or_val(
    tmp_path: Path,
) -> None:
    test_energy = 5692.994 + (5801.399 - 5692.994) * np.linspace(0, 1, 135) ** 1.1
    train_energy = 5692.994 + (5801.399 - 5692.994) * np.linspace(0, 1, 135) ** 1.7
    source_root = _write_tree(
        tmp_path,
        {
            "Exp4_La08_heating_800_000_000.dat": _spec_bytes(energy=test_energy),
            "Exp5_La02_heating_800_000_000.dat": _spec_bytes(energy=train_energy),
            "Exp8_Gd05_heating_800_000_000.dat": _spec_bytes(),
        },
    )
    declaration = _write_declaration(tmp_path, source_root=source_root)

    result = materialize_xanes_spec(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=tmp_path / "bundle",
    )
    manifest = _read_json(result.manifest_path)

    assert manifest["reference_grid"]["source_path"] == (
        "Exp5_La02_heating_800_000_000.dat"
    )
    assert manifest["reference_grid"]["source_sha256"] == hashlib.sha256(
        (source_root / "Exp5_La02_heating_800_000_000.dat").read_bytes()
    ).hexdigest()
    frozen_train_energy = parse_xanes_spec(
        source_root / "Exp5_La02_heating_800_000_000.dat"
    ).energy
    with np.load(result.asset_path, allow_pickle=False) as bundle:
        np.testing.assert_array_equal(bundle["energy"], frozen_train_energy)
        assert not np.array_equal(bundle["energy"], test_energy)


def test_reference_grid_clamping_rejects_more_than_tiny_endpoint_jitter(
    tmp_path: Path,
) -> None:
    reference_energy = 5692.97 + (5801.399 - 5692.97) * np.linspace(0, 1, 135) ** 1.4
    overhanging_energy = 5693.03 + (5801.399 - 5693.03) * np.linspace(0, 1, 135) ** 1.4
    source_root = _write_tree(
        tmp_path,
        {
            "Exp2_La05_heating_500_000_000.dat": _spec_bytes(
                energy=reference_energy
            ),
            "Exp5_La02_heating_800_000_000.dat": _spec_bytes(
                energy=overhanging_energy
            ),
        },
    )
    declaration = _write_declaration(tmp_path, source_root=source_root)

    result = materialize_xanes_spec(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=tmp_path / "bundle",
    )

    qc = _read_json(result.qc_path)
    by_path = {entry["path"]: entry for entry in qc["files"]}
    assert by_path["Exp5_La02_heating_800_000_000.dat"]["status"] == "rejected"
    assert by_path["Exp5_La02_heating_800_000_000.dat"]["reason_code"] == (
        "reference_grid_coverage"
    )
    with np.load(result.asset_path, allow_pickle=False) as bundle:
        assert bundle["sample_ids"].tolist() == [
            "Exp2_La05_heating_500_000_000"
        ]


def test_materialization_is_order_independent_and_dose_deterministic(
    tmp_path: Path,
) -> None:
    source_root = _write_tree(
        tmp_path,
        {
            "Exp2_La05_heating_500_000_000.dat": _spec_bytes(ketek=1_100.0),
            "Exp4_La08_heating_800_000_000.dat": _spec_bytes(ketek=1_400.0),
        },
    )
    declaration = _write_declaration(tmp_path, source_root=source_root)
    reversed_declaration = _write_declaration(
        tmp_path, source_root=source_root, reverse_files=True
    )

    first = materialize_xanes_spec(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=tmp_path / "bundle-a",
        dose_fraction=0.25,
        global_seed=42,
    )
    second = materialize_xanes_spec(
        source_root=source_root,
        source_declaration_file=reversed_declaration,
        output_directory=tmp_path / "bundle-b",
        dose_fraction=0.25,
        global_seed=42,
    )
    lower_dose = materialize_xanes_spec(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=tmp_path / "bundle-c",
        dose_fraction=0.1,
        global_seed=42,
    )

    assert first.asset_path.read_bytes() == second.asset_path.read_bytes()
    assert first.benchmark_asset_digest == second.benchmark_asset_digest
    assert first.benchmark_asset_digest != lower_dose.benchmark_asset_digest
    with np.load(first.asset_path, allow_pickle=False) as bundle:
        dose = 0.25
        assert np.all(bundle["noisy_i0_counts"] * dose <= np.rint(bundle["i0_counts"]))
        assert np.all(
            bundle["noisy_ketek_counts"] * dose <= np.rint(bundle["ketek_counts"])
        )
    with np.load(lower_dose.asset_path, allow_pickle=False) as bundle:
        dose = 0.1
        assert np.all(bundle["noisy_i0_counts"] * dose <= np.rint(bundle["i0_counts"]))


def test_digest_classes_are_separate_and_unknown_revision_is_explicit(
    tmp_path: Path,
) -> None:
    source_root = _write_tree(
        tmp_path,
        {"Exp2_La05_heating_500_000_000.dat": _spec_bytes()},
    )
    declaration = _write_declaration(
        tmp_path, source_root=source_root, upstream_revision=None
    )

    result = materialize_xanes_spec(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=tmp_path / "bundle",
    )
    manifest = _read_json(result.manifest_path)

    assert len(
        {
            result.source_dataset_digest,
            result.source_content_manifest_digest,
            result.benchmark_asset_digest,
        }
    ) == 3
    assert manifest["dataset"]["upstream_revision"] is None
    assert manifest["source_dataset_digest"] == result.source_dataset_digest
    assert manifest["source_content_manifest_digest"] == (
        result.source_content_manifest_digest
    )
    assert manifest["benchmark_asset_digest"] == result.benchmark_asset_digest
    assert hashlib.sha256(result.asset_path.read_bytes()).hexdigest() == (
        result.benchmark_asset_digest
    )


def test_benchmark_identity_loader_recomputes_every_source_identity(
    tmp_path: Path,
) -> None:
    source_root = _write_tree(
        tmp_path,
        {"Exp2_La05_heating_500_000_000.dat": _spec_bytes()},
    )
    declaration = _write_declaration(tmp_path, source_root=source_root)
    result = materialize_xanes_spec(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=tmp_path / "bundle",
    )

    identity = load_benchmark_asset_identity(result.manifest_path)

    assert identity.declared_manifest_digest == "a" * 64
    assert identity.source_dataset_digest == result.source_dataset_digest
    assert identity.source_content_manifest_digest == (
        result.source_content_manifest_digest
    )

    manifest = _read_json(result.manifest_path)
    manifest["source_dataset_digest"] = "f" * 64
    tampered_dataset = tmp_path / "tampered-dataset.json"
    tampered_dataset.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source dataset digest"):
        load_benchmark_asset_identity(tampered_dataset)

    manifest = _read_json(result.manifest_path)
    manifest["source_content_manifest"]["files"][0]["actual_size"] += 1
    tampered_content = tmp_path / "tampered-content.json"
    tampered_content.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source content manifest digest"):
        load_benchmark_asset_identity(tampered_content)


def test_bundle_loads_as_denoising_pairs_with_honest_target_semantics(
    tmp_path: Path,
) -> None:
    source_root = _write_tree(
        tmp_path,
        {
            "Exp2_La05_heating_500_000_000.dat": _spec_bytes(),
            "Exp4_La08_heating_800_000_000.dat": _spec_bytes(ketek=1_500.0),
        },
    )
    declaration = _write_declaration(tmp_path, source_root=source_root)
    result = materialize_xanes_spec(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=tmp_path / "bundle",
    )

    pairs = load_denoising_pairs(result.output_directory)

    assert [pair.split for pair in pairs] == ["train", "test"]
    for pair in pairs:
        assert pair.noisy.modality == "xanes"
        assert pair.clean.provenance["target_semantics"] == (
            "pseudo-clean frozen measurement"
        )
        assert pair.clean.provenance["is_physical_noiseless_ground_truth"] is False
        assert pair.clean.signal_unit == "ketek/i0 ratio"
        np.testing.assert_array_equal(
            pair.noisy.axes[0].values, pair.clean.axes[0].values
        )
        assert pair.assignment.manifest_digest == result.split_manifest_digest


def test_bundle_loader_parses_the_same_asset_snapshot_that_it_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = _write_tree(
        tmp_path,
        {"Exp2_La05_heating_500_000_000.dat": _spec_bytes()},
    )
    declaration = _write_declaration(tmp_path, source_root=source_root)
    result = materialize_xanes_spec(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=tmp_path / "bundle",
    )
    with np.load(result.asset_path, allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    original_noisy = np.array(arrays["noisy"], copy=True)
    arrays["noisy"] = original_noisy + 1.0
    replacement = tmp_path / "replacement.npz"
    np.savez(replacement, **arrays)
    original_hash = xanes_spec._sha256_file

    def replace_after_hash(path: Path) -> str:
        digest = original_hash(path)
        if path == result.asset_path:
            path.write_bytes(replacement.read_bytes())
        return digest

    monkeypatch.setattr(xanes_spec, "_sha256_file", replace_after_hash)

    pairs = load_denoising_pairs(result.output_directory)

    np.testing.assert_array_equal(pairs[0].noisy.signal, original_noisy[0])
