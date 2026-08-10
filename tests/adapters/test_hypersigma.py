from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import BinaryIO

import pytest

from hyperspectrum.adapters.hypersigma import (
    HYPERSIGMA_CODE_LICENSE,
    HYPERSIGMA_SOURCE_COMMIT,
    HYPERSIGMA_SOURCE_FILES,
    HYPERSIGMA_WEIGHTS,
    WeightAssetContract,
    load_verified_state_dict,
    main,
    verification_report,
    verify_source_tree,
)


class FakeTorch:
    def __init__(self, loaded: object | None = None) -> None:
        self.calls: list[tuple[BinaryIO, str, bool]] = []
        self.loaded = (
            {"net": {"layer.weight": object()}} if loaded is None else loaded
        )

    def load(
        self,
        stream: BinaryIO,
        *,
        map_location: str,
        weights_only: bool,
    ) -> object:
        assert stream.tell() == 0
        assert not stream.closed
        self.calls.append((stream, map_location, weights_only))
        return self.loaded


def small_contract(payload: bytes) -> WeightAssetContract:
    return WeightAssetContract(
        variant="gaussian",
        repository="test/weights",
        revision="1" * 40,
        asset_id="test-gaussian",
        source_url="https://example.test/weight.pth",
        filename="weight.pth",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        license="Apache-2.0",
    )


def test_official_source_and_weight_contract_is_exact() -> None:
    # Break caught: a mutable revision, similarly named checkpoint, or altered
    # upstream source file could run under the official HyperSIGMA identity.
    assert HYPERSIGMA_SOURCE_COMMIT == "07e9ea24e3072fcb5c3a92a2bcb8185e43b295b9"
    assert HYPERSIGMA_CODE_LICENSE == "Apache-2.0"
    assert dict(HYPERSIGMA_SOURCE_FILES) == {
        "ImageDenoising/models/hypersigma/model.py": (
            "66c2165b1e04aa7df7c995f3d9ee84a332aa189572364ffb74be8a3b89711c1f"
        ),
        "ImageDenoising/models/hypersigma/Spatial.py": (
            "b9834e915333c9f7b8c48d818a0a7bf995c6c8966a1f9b1b3299e642bdb2256b"
        ),
        "ImageDenoising/models/hypersigma/Spectral.py": (
            "c2fbbdcfbf75622c7ecfd88a3ff7c42295f19684e24b6022fd65e1f04050da92"
        ),
        "ImageDenoising/models/hypersigma/Spatial_route.py": (
            "6b80337dfa94253504936584a85d62b5c81db380ba45ef52bc1a59cf0026883f"
        ),
        "ImageDenoising/models/hypersigma/Spectral_route.py": (
            "617916efbe01fff98248f7b95bcd53bc9265b14f425aed2f90d0a8ee800931b3"
        ),
    }


def test_source_tree_verifies_each_expected_file(tmp_path: Path) -> None:
    # Break caught: checking only the repository revision could accept locally
    # edited Python files after checkout.
    expected: dict[str, str] = {}
    for relative, payload in {
        "ImageDenoising/models/hypersigma/model.py": b"model",
        "ImageDenoising/models/hypersigma/Spatial.py": b"spatial",
    }.items():
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(payload)
        expected[relative] = hashlib.sha256(payload).hexdigest()

    verified = verify_source_tree(tmp_path, expected_files=expected)

    assert verified.root == tmp_path.resolve()
    assert verified.commit == HYPERSIGMA_SOURCE_COMMIT
    assert dict(verified.file_sha256) == expected


def test_source_tree_reports_the_exact_bad_path(tmp_path: Path) -> None:
    # Break caught: a modified source file could pass without a path-specific
    # expected-versus-actual diagnostic.
    relative = "ImageDenoising/models/hypersigma/model.py"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"wrong")
    expected = {relative: hashlib.sha256(b"right").hexdigest()}

    with pytest.raises(ValueError, match=r"model\.py.*expected.*actual"):
        verify_source_tree(tmp_path, expected_files=expected)


def test_source_tree_reports_the_exact_missing_path(tmp_path: Path) -> None:
    # Break caught: an incomplete source checkout could reach dynamic import.
    relative = "ImageDenoising/models/hypersigma/Spectral.py"

    with pytest.raises(ValueError, match=r"cannot read verified source.*Spectral\.py"):
        verify_source_tree(tmp_path, expected_files={relative: "0" * 64})


def test_load_uses_a_verified_open_descriptor(tmp_path: Path) -> None:
    # Break caught: deserialization could use a different pathname lookup from
    # the descriptor whose bytes were hashed.
    payload = b"safe fake checkpoint"
    path = tmp_path / "weight.pth"
    path.write_bytes(payload)
    fake_torch = FakeTorch()

    state, identity = load_verified_state_dict(
        path, small_contract(payload), fake_torch
    )

    assert tuple(state) == ("layer.weight",)
    assert identity.sha256 == hashlib.sha256(payload).hexdigest()
    assert identity.path == path.resolve()
    assert [(call[1], call[2]) for call in fake_torch.calls] == [("cpu", True)]
    assert fake_torch.calls[0][0].closed


def test_digest_mismatch_stops_before_torch(tmp_path: Path) -> None:
    # Break caught: untrusted checkpoint bytes could reach torch.load before
    # their declared SHA-256 identity is proven.
    path = tmp_path / "weight.pth"
    path.write_bytes(b"changed!")
    fake_torch = FakeTorch()

    with pytest.raises(ValueError, match="weight SHA-256 mismatch"):
        load_verified_state_dict(path, small_contract(b"expected"), fake_torch)

    assert fake_torch.calls == []


def test_size_mismatch_stops_before_torch(tmp_path: Path) -> None:
    # Break caught: truncated or extended checkpoint bytes could reach torch.load.
    path = tmp_path / "weight.pth"
    path.write_bytes(b"short")
    fake_torch = FakeTorch()

    with pytest.raises(ValueError, match="weight size mismatch"):
        load_verified_state_dict(path, small_contract(b"expected"), fake_torch)

    assert fake_torch.calls == []


@pytest.mark.parametrize(
    ("loaded", "error_type", "message"),
    [
        (["not", "a", "mapping"], TypeError, "checkpoint must be a mapping"),
        ({"weights": {}}, ValueError, "top-level net"),
        (
            {"net": ["not", "a", "mapping"]},
            TypeError,
            "net must be a state-dictionary",
        ),
    ],
)
def test_checkpoint_requires_a_net_state_dictionary(
    tmp_path: Path,
    loaded: object,
    error_type: type[Exception],
    message: str,
) -> None:
    # Break caught: an unrelated safe-deserializable object could be mistaken for
    # the official checkpoint state dictionary.
    payload = b"safe fake checkpoint"
    path = tmp_path / "weight.pth"
    path.write_bytes(payload)

    with pytest.raises(error_type, match=message):
        load_verified_state_dict(path, small_contract(payload), FakeTorch(loaded))


def test_static_verification_report_is_json_serializable_without_local_assets() -> None:
    # Break caught: the manifest's no-argument verification command could require
    # torch or locally downloaded research assets merely to report its contract.
    report = verification_report()

    assert report["schema_version"] == "hyperspectrum-hypersigma-verification/v1"
    assert report["source"] == {
        "commit": HYPERSIGMA_SOURCE_COMMIT,
        "license": "Apache-2.0",
        "files": dict(HYPERSIGMA_SOURCE_FILES),
        "status": "declared",
    }
    weights = report["weights"]
    assert isinstance(weights, dict)
    assert set(weights) == {"gaussian", "complex"}
    assert weights["gaussian"]["sha256"] == HYPERSIGMA_WEIGHTS["gaussian"].sha256
    assert weights["complex"]["size_bytes"] == 2266408098
    assert weights["gaussian"]["status"] == "declared"
    assert json.loads(json.dumps(report)) == report


def test_verify_cli_outputs_the_static_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Break caught: the exact verify command declared in tool.yaml could stop
    # working or emit non-machine-readable output.
    exit_code = main(["--verify"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == verification_report()
    assert captured.err == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ["--verify", "--weight", "unknown=/tmp/weight.pth"],
        [
            "--verify",
            "--weight",
            "gaussian=/tmp/one.pth",
            "--weight",
            "gaussian=/tmp/two.pth",
        ],
    ],
)
def test_verify_cli_rejects_unknown_or_duplicate_weight_variants(
    arguments: list[str],
) -> None:
    # Break caught: a misspelled or repeated variant could silently verify the
    # wrong checkpoint while omitting the intended one.
    with pytest.raises(SystemExit) as caught:
        main(arguments)

    assert caught.value.code == 2
    assert HYPERSIGMA_WEIGHTS == {
        "gaussian": WeightAssetContract(
            variant="gaussian",
            repository="WHU-Sigma/HyperSIGMA",
            revision="e0567395fbdfddbae994695baf5fc73358a1ec3c",
            asset_id="hf-whu-sigma-hypersigma-gaussian-e0567395",
            source_url=(
                "https://huggingface.co/WHU-Sigma/HyperSIGMA/resolve/"
                "e0567395fbdfddbae994695baf5fc73358a1ec3c/"
                "Denoising_models/hypersigma_gaussian_noise_model.pth"
            ),
            filename="hypersigma_gaussian_noise_model.pth",
            size_bytes=2266408098,
            sha256=(
                "dc101cfe7d462d721eb46395b1621d82103cf4e72166d8591b5619cb2d10e806"
            ),
            license="Apache-2.0",
        ),
        "complex": WeightAssetContract(
            variant="complex",
            repository="WHU-Sigma/HyperSIGMA",
            revision="e0567395fbdfddbae994695baf5fc73358a1ec3c",
            asset_id="hf-whu-sigma-hypersigma-complex-e0567395",
            source_url=(
                "https://huggingface.co/WHU-Sigma/HyperSIGMA/resolve/"
                "e0567395fbdfddbae994695baf5fc73358a1ec3c/"
                "Denoising_models/hypersigma_complex_noise_model.pth"
            ),
            filename="hypersigma_complex_noise_model.pth",
            size_bytes=2266408098,
            sha256=(
                "8b1162aae6811af67d287b9271e74154db5448c5e2fd6df953dae5d7807bbe7e"
            ),
            license="Apache-2.0",
        ),
    }
