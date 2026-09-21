from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import BinaryIO

import numpy as np
import pytest
from numpy.typing import NDArray

from hyperspectrum import adapters
from hyperspectrum.adapters import hypersigma as adapter_module
from hyperspectrum.adapters.hypersigma import (
    HYPERSIGMA_CODE_LICENSE,
    HYPERSIGMA_SOURCE_COMMIT,
    HYPERSIGMA_SOURCE_FILES,
    HYPERSIGMA_WEIGHTS,
    HyperSIGMADenoiseAdapter,
    HyperSIGMARuntimeIdentity,
    VerifiedSourceTree,
    VerifiedWeightIdentity,
    WeightAssetContract,
    _build_official_model,
    _construct_official_model,
    _TorchRuntime,
    denoise_cubes,
    load_verified_state_dict,
    main,
    verification_report,
    verify_source_tree,
)
from hyperspectrum.denoising import (
    DenoisingPair,
    SpectrumAxis,
    SpectrumSample,
    SplitEntry,
    SplitManifest,
    evaluate_denoising_suite,
)
from hyperspectrum.denoising.model import CanonicalDenoisingInput

_IMPORT_MODULE = importlib.import_module


class FakeSafeGlobalsContext:
    def __init__(self, serialization: FakeSerialization) -> None:
        self.serialization = serialization

    def __enter__(self) -> None:
        self.serialization.active = True

    def __exit__(self, *args: object) -> None:
        self.serialization.active = False


class FakeSerialization:
    def __init__(self) -> None:
        self.active = False
        self.allowlist: list[object] = []

    def safe_globals(self, allowlist: list[object]) -> FakeSafeGlobalsContext:
        self.allowlist = allowlist
        return FakeSafeGlobalsContext(self)


class FakeTorch:
    def __init__(self, loaded: object | None = None) -> None:
        self.calls: list[tuple[BinaryIO, str, bool]] = []
        self.serialization = FakeSerialization()
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


class FakeNumPyScalarTorch(FakeTorch):
    def load(
        self,
        stream: BinaryIO,
        *,
        map_location: str,
        weights_only: bool,
    ) -> object:
        legacy_names = {
            entry[1]
            for entry in self.serialization.allowlist
            if isinstance(entry, tuple) and len(entry) == 2
        }
        if not self.serialization.active:
            raise RuntimeError("NumPy checkpoint global was not scoped")
        if "numpy.core.multiarray.scalar" not in legacy_names:
            raise RuntimeError("legacy NumPy scalar global was not allowlisted")
        if np.dtype not in self.serialization.allowlist:
            raise RuntimeError("NumPy dtype global was not allowlisted")
        return super().load(
            stream, map_location=map_location, weights_only=weights_only
        )


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


class FakeRuntime:
    def __init__(self, *, output_shape: tuple[int, ...] | None = None) -> None:
        self.seen_shapes: list[tuple[int, ...]] = []
        self.output_shape = output_shape

    def predict(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        self.seen_shapes.append(values.shape)
        if self.output_shape is not None:
            return np.zeros(self.output_shape, dtype=np.float64)
        return values + 0.125


class FakeDevice:
    def __init__(self, name: str) -> None:
        self.name = name
        self.type = "cuda" if name.startswith("cuda") else "cpu"

    def __str__(self) -> str:
        return self.name


class FakeTensor:
    def __init__(self, values: np.ndarray, events: list[object]) -> None:
        self.values = np.array(values, copy=True)
        self.events = events

    def unsqueeze(self, axis: int) -> FakeTensor:
        self.events.append(("unsqueeze", axis))
        return FakeTensor(np.expand_dims(self.values, axis), self.events)

    def to(self, device: object) -> FakeTensor:
        self.events.append(("tensor-to", str(device)))
        return self

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return np.array(self.values, copy=True)


class FakeIncompatible:
    def __init__(
        self,
        *,
        missing_keys: tuple[str, ...] = (),
        unexpected_keys: tuple[str, ...] = (),
    ) -> None:
        self.missing_keys = missing_keys
        self.unexpected_keys = unexpected_keys


class FakeModel:
    def __init__(
        self,
        events: list[object],
        *,
        incompatible: FakeIncompatible | None = None,
        output_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.events = events
        self.incompatible = incompatible or FakeIncompatible()
        self.output_shape = output_shape
        self.training = True

    def load_state_dict(self, state: object, *, strict: bool) -> FakeIncompatible:
        self.events.append(("load-state", state, strict))
        return self.incompatible

    def requires_grad_(self, enabled: bool) -> FakeModel:
        self.events.append(("requires-grad", enabled))
        return self

    def eval(self) -> FakeModel:
        self.training = False
        self.events.append("eval")
        return self

    def to(self, device: object) -> FakeModel:
        self.events.append(("model-to", str(device)))
        return self

    def __call__(self, tensor: FakeTensor) -> FakeTensor:
        self.events.append(("model-call", tensor.values.shape, tensor.values.dtype))
        if self.output_shape is None:
            return FakeTensor(tensor.values + np.float32(0.25), self.events)
        return FakeTensor(np.zeros(self.output_shape, dtype=np.float32), self.events)


class FakeInferenceMode:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def __enter__(self) -> None:
        self.events.append("inference-enter")

    def __exit__(self, *args: object) -> None:
        self.events.append("inference-exit")


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeRuntimeTorch(FakeTorch):
    __version__ = "test-torch"
    cuda = FakeCuda()

    def __init__(self, events: list[object]) -> None:
        super().__init__({"net": {"layer.weight": object()}})
        self.events = events

    def load(
        self,
        stream: BinaryIO,
        *,
        map_location: str,
        weights_only: bool,
    ) -> object:
        self.events.append(("torch-load", map_location, weights_only))
        return super().load(
            stream, map_location=map_location, weights_only=weights_only
        )

    def device(self, name: str) -> FakeDevice:
        self.events.append(("device", name))
        return FakeDevice(name)

    def from_numpy(self, values: np.ndarray) -> FakeTensor:
        self.events.append(("from-numpy", values.shape, values.dtype))
        return FakeTensor(values, self.events)

    def inference_mode(self) -> FakeInferenceMode:
        return FakeInferenceMode(self.events)


class FakeSpatial:
    def init_weights(self, path: str) -> None:
        raise AssertionError(f"private spatial pretrain was accessed: {path}")


class FakeSpectral:
    def init_weights(self, path: str) -> None:
        raise AssertionError(f"private spectral pretrain was accessed: {path}")


def fake_upstream_modules(
    *, constructor_error: Exception | None = None
) -> dict[str, ModuleType]:
    spatial = ModuleType("fake.Spatial")
    spectral = ModuleType("fake.Spectral")
    model = ModuleType("fake.model")
    spatial.SpatialVisionTransformer = FakeSpatial  # type: ignore[attr-defined]
    spectral.SpectralVisionTransformer = FakeSpectral  # type: ignore[attr-defined]

    def construct() -> object:
        FakeSpatial().init_weights("/private/spatial.pth")
        FakeSpectral().init_weights("/private/spectral.pth")
        if constructor_error is not None:
            raise constructor_error
        return {"official": "model"}

    model.spat_vit_b_rvsa = construct  # type: ignore[attr-defined]
    return {"Spatial": spatial, "Spectral": spectral, "model": model}


def canonical_hsi_input(
    signal: np.ndarray,
    *,
    axis_names: tuple[str, ...] = ("y", "x", "band"),
    axis_units: tuple[str, ...] = ("pixel", "pixel", "index"),
    axis_lengths: tuple[int, ...] | None = None,
    modality: str = "hyperspectral",
    representation: str = "dense",
    channel_labels: tuple[str, ...] = (),
    normalization_method: str = "per_spectrum_range",
    valid_mask: np.ndarray | None = None,
) -> CanonicalDenoisingInput:
    lengths = signal.shape if axis_lengths is None else axis_lengths
    return CanonicalDenoisingInput(
        sample_id="synthetic-cube",
        modality=modality,  # type: ignore[arg-type]
        representation=representation,  # type: ignore[arg-type]
        signal=signal,
        valid_mask=(
            np.ones(signal.shape, dtype=np.bool_)
            if valid_mask is None
            else valid_mask
        ),
        axis_values=tuple(np.arange(length, dtype=np.float64) for length in lengths),
        axis_names=axis_names,
        axis_units=axis_units,
        axis_directions=tuple("increasing" for _ in axis_names),  # type: ignore[arg-type]
        channel_labels=channel_labels,
        normalization_method=normalization_method,
        normalization_state_digest="a" * 64,
    )


def hsi_evaluation_pair(
    signal: np.ndarray,
    *,
    representation: str = "dense",
    channel_labels: tuple[str, ...] = (),
) -> DenoisingPair:
    axis_shape = signal.shape[1:] if channel_labels else signal.shape
    axes = tuple(
        SpectrumAxis(
            name=f"axis-{index}",
            unit="arb",
            direction="increasing",
            values=np.arange(size, dtype=np.float64),
        )
        for index, size in enumerate(axis_shape)
    )

    def sample() -> SpectrumSample:
        return SpectrumSample(
            sample_id="incompatible-hsi",
            group_id="incompatible-scene",
            modality="hyperspectral",
            representation=representation,  # type: ignore[arg-type]
            axes=axes,
            signal=signal,
            valid_mask=np.ones(signal.shape, dtype=np.bool_),
            signal_unit="relative_reflectance",
            channel_labels=channel_labels,
        )

    noisy = sample()
    clean = sample()
    manifest = SplitManifest(
        schema_version="hyperspectrum-split-manifest/v1",
        entries=(SplitEntry(noisy.sample_id, noisy.group_id, "test"),),
    )
    return DenoisingPair(
        noisy=noisy,
        clean=clean,
        assignment=manifest.assignment_for(noisy.sample_id, noisy.group_id),
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


def test_load_scopes_the_checkpoint_numpy_allowlist(tmp_path: Path) -> None:
    # Break caught: the pinned checkpoints contain a legacy NumPy scalar, which
    # safe torch loading rejects unless its narrow compatibility globals are
    # allowlisted for this one deserialization operation.
    payload = b"safe fake checkpoint with a NumPy scalar"
    path = tmp_path / "weight.pth"
    path.write_bytes(payload)
    fake_torch = FakeNumPyScalarTorch()

    state, _ = load_verified_state_dict(
        path, small_contract(payload), fake_torch
    )

    assert tuple(state) == ("layer.weight",)
    assert fake_torch.calls[0][2] is True
    assert fake_torch.serialization.active is False


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


def test_adapter_transposes_to_model_and_restores_canonical_layout() -> None:
    # Break caught: a valid y/x/band cube could be fed to the official model in
    # caller order or returned without restoring that exact order.
    signal = np.zeros((64, 64, 191), dtype=np.float64)
    model_input = canonical_hsi_input(signal)
    runtime = FakeRuntime()
    adapter = HyperSIGMADenoiseAdapter(
        Path("unused-source"),
        Path("unused-weight"),
        variant="gaussian",
        _runtime=runtime,
    )

    output = adapter.predict(model_input)

    assert runtime.seen_shapes == [(191, 64, 64)]
    assert output.signal.shape == signal.shape
    assert np.all(output.signal == 0.125)
    assert output.sample_id == model_input.sample_id
    assert output.normalization_state_digest == (
        model_input.normalization_state_digest
    )
    assert np.array_equal(output.valid_mask, model_input.valid_mask)


@pytest.mark.parametrize(
    ("model_input", "message"),
    [
        (
            canonical_hsi_input(
                np.zeros((64, 64, 191)),
                valid_mask=np.pad(
                    np.ones((64, 64, 190), dtype=np.bool_),
                    ((0, 0), (0, 0), (0, 1)),
                ),
            ),
            "fully valid",
        ),
        (canonical_hsi_input(np.zeros((64, 64, 190))), "band count"),
        (canonical_hsi_input(np.zeros((63, 64, 191))), "spatial shape"),
        (
            canonical_hsi_input(
                np.zeros((64, 64, 191), dtype=np.complex64),
                representation="complex",
            ),
            "real dense",
        ),
        (
            canonical_hsi_input(
                np.zeros((64, 64, 191)), representation="sparse_peaks"
            ),
            "real dense",
        ),
        (
            canonical_hsi_input(
                np.zeros((2, 64, 64, 191)),
                axis_lengths=(64, 64, 191),
                channel_labels=("one", "two"),
            ),
            "implicit channel",
        ),
        (
            canonical_hsi_input(np.zeros((64, 64, 191)), modality="raman"),
            "hyperspectral modality",
        ),
        (
            canonical_hsi_input(
                np.zeros((64, 64, 191)), normalization_method="none"
            ),
            "per_spectrum_range",
        ),
    ],
)
def test_adapter_rejects_incompatible_canonical_input_before_runtime(
    model_input: CanonicalDenoisingInput, message: str
) -> None:
    # Break caught: unsupported structure could be silently reshaped, flattened,
    # channel-coerced, or normalized inside the adapter.
    runtime = FakeRuntime()
    adapter = HyperSIGMADenoiseAdapter(
        Path("unused-source"),
        Path("unused-weight"),
        variant="gaussian",
        _runtime=runtime,
    )

    with pytest.raises(ValueError, match=message):
        adapter.predict(model_input)

    assert runtime.seen_shapes == []


def test_adapter_rejects_runtime_shape_drift() -> None:
    # Break caught: an upstream output with a changed shape could be reinterpreted
    # as the caller's declared HSI layout.
    adapter = HyperSIGMADenoiseAdapter(
        Path("unused-source"),
        Path("unused-weight"),
        variant="gaussian",
        _runtime=FakeRuntime(output_shape=(191, 64, 63)),
    )

    with pytest.raises(ValueError, match="model shape"):
        adapter.predict(canonical_hsi_input(np.zeros((64, 64, 191))))


@pytest.mark.parametrize("constructor_error", [None, RuntimeError("boom")])
def test_constructor_restores_private_pretrain_methods(
    constructor_error: Exception | None,
) -> None:
    # Break caught: process-global upstream class methods could remain patched
    # after successful construction or an exception.
    spatial_original = FakeSpatial.init_weights
    spectral_original = FakeSpectral.init_weights
    modules = fake_upstream_modules(constructor_error=constructor_error)

    if constructor_error is None:
        assert _construct_official_model(modules) == {"official": "model"}
    else:
        with pytest.raises(RuntimeError, match="boom"):
            _construct_official_model(modules)

    assert FakeSpatial.init_weights is spatial_original
    assert FakeSpectral.init_weights is spectral_original


def test_verified_module_loader_uses_an_isolated_temporary_namespace(
    tmp_path: Path,
) -> None:
    # Break caught: upstream modules could permanently modify sys.path or collide
    # with another package named model, Spatial, or Spectral.
    model_dir = tmp_path / "ImageDenoising/models/hypersigma"
    model_dir.mkdir(parents=True)
    (model_dir / "Spatial.py").write_text(
        "class SpatialVisionTransformer:\n"
        "    def init_weights(self, path):\n"
        "        raise AssertionError(path)\n",
        encoding="utf-8",
    )
    (model_dir / "Spectral.py").write_text(
        "class SpectralVisionTransformer:\n"
        "    def init_weights(self, path):\n"
        "        raise AssertionError(path)\n",
        encoding="utf-8",
    )
    (model_dir / "Spatial_route.py").write_text("ROUTE = 'spatial'\n", encoding="utf-8")
    (model_dir / "Spectral_route.py").write_text(
        "ROUTE = 'spectral'\n", encoding="utf-8"
    )
    (model_dir / "model.py").write_text(
        "from .Spatial import SpatialVisionTransformer\n"
        "from .Spectral import SpectralVisionTransformer\n"
        "def spat_vit_b_rvsa():\n"
        "    SpatialVisionTransformer().init_weights('/private/spatial.pth')\n"
        "    SpectralVisionTransformer().init_weights('/private/spectral.pth')\n"
        "    return {'official': 'model'}\n",
        encoding="utf-8",
    )
    verified = VerifiedSourceTree(
        root=tmp_path.resolve(),
        commit=HYPERSIGMA_SOURCE_COMMIT,
        file_sha256={
            f"ImageDenoising/models/hypersigma/{path.name}": hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in model_dir.glob("*.py")
        },
    )
    path_before = list(sys.path)
    modules_before = set(sys.modules)

    built = _build_official_model(verified)

    assert built == {"official": "model"}
    assert sys.path == path_before
    assert set(sys.modules) == modules_before
    assert not any(name.startswith("_hypersigma_verified_") for name in sys.modules)
    assert not (model_dir / "__pycache__").exists()


def test_verified_module_loader_rehashes_the_bytes_it_executes(tmp_path: Path) -> None:
    # Break caught: verified source files could be replaced after the initial
    # scan and the loader could execute different, unverified bytes by pathname.
    model_dir = tmp_path / "ImageDenoising/models/hypersigma"
    model_dir.mkdir(parents=True)
    sources = {
        "Spatial.py": (
            "class SpatialVisionTransformer:\n"
            "    def init_weights(self, path):\n"
            "        raise AssertionError(path)\n"
        ),
        "Spectral.py": (
            "class SpectralVisionTransformer:\n"
            "    def init_weights(self, path):\n"
            "        raise AssertionError(path)\n"
        ),
        "Spatial_route.py": "ROUTE = 'spatial'\n",
        "Spectral_route.py": "ROUTE = 'spectral'\n",
        "model.py": (
            "from .Spatial import SpatialVisionTransformer\n"
            "from .Spectral import SpectralVisionTransformer\n"
            "def spat_vit_b_rvsa():\n"
            "    SpatialVisionTransformer().init_weights('/private/spatial.pth')\n"
            "    SpectralVisionTransformer().init_weights('/private/spectral.pth')\n"
            "    return {'official': 'model'}\n"
        ),
    }
    expected: dict[str, str] = {}
    for filename, source in sources.items():
        relative = f"ImageDenoising/models/hypersigma/{filename}"
        (model_dir / filename).write_text(source, encoding="utf-8")
        expected[relative] = hashlib.sha256(source.encode()).hexdigest()
    verified = verify_source_tree(tmp_path, expected_files=expected)
    marker = tmp_path / "tampered-source-executed"
    (model_dir / "Spatial_route.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "ROUTE = 'tampered'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"source SHA-256 mismatch.*Spatial_route\.py"):
        _build_official_model(verified)

    assert not marker.exists()


def test_torch_runtime_loads_strict_eval_no_grad_and_predicts_inference_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Break caught: source/weight verification, strict loading, frozen parameters,
    # eval mode, CPU placement, or inference_mode could be skipped or reordered.
    events: list[object] = []
    payload = b"verified checkpoint"
    weight_path = tmp_path / "weight.pth"
    weight_path.write_bytes(payload)
    source_root = tmp_path / "source"
    source_identity = VerifiedSourceTree(
        root=source_root.resolve(),
        commit=HYPERSIGMA_SOURCE_COMMIT,
        file_sha256={"model.py": "1" * 64},
    )
    torch_module = FakeRuntimeTorch(events)
    model = FakeModel(events)

    def verify_source(path: Path) -> VerifiedSourceTree:
        events.append(("verify-source", path))
        return source_identity

    def build_model(verified: VerifiedSourceTree) -> FakeModel:
        events.append(("build-model", verified.commit))
        return model

    monkeypatch.setattr(adapter_module, "verify_source_tree", verify_source)
    monkeypatch.setattr(adapter_module, "_build_official_model", build_model)
    monkeypatch.setattr(
        adapter_module.importlib,
        "import_module",
        lambda name: torch_module if name == "torch" else _IMPORT_MODULE(name),
    )

    runtime = _TorchRuntime(
        source_root,
        weight_path,
        contract=small_contract(payload),
        device="cpu",
    )
    predicted = runtime.predict(np.zeros((191, 64, 64), dtype=np.float64))

    expected_state = {"layer.weight": torch_module.loaded["net"]["layer.weight"]}  # type: ignore[index]
    assert events[:8] == [
        ("verify-source", source_root),
        ("torch-load", "cpu", True),
        ("build-model", HYPERSIGMA_SOURCE_COMMIT),
        ("load-state", expected_state, True),
        ("requires-grad", False),
        "eval",
        ("device", "cpu"),
        ("model-to", "cpu"),
    ]
    assert events[8:] == [
        ("from-numpy", (191, 64, 64), np.dtype(np.float32)),
        ("unsqueeze", 0),
        ("tensor-to", "cpu"),
        "inference-enter",
        ("model-call", (1, 191, 64, 64), np.dtype(np.float32)),
        "inference-exit",
    ]
    assert predicted.dtype == np.dtype(np.float64)
    assert predicted.shape == (191, 64, 64)
    assert np.all(predicted == 0.25)
    assert runtime.runtime_identity == HyperSIGMARuntimeIdentity(
        source_commit=HYPERSIGMA_SOURCE_COMMIT,
        source_sha256={"model.py": "1" * 64},
        weight=VerifiedWeightIdentity(
            variant="gaussian",
            path=weight_path.resolve(),
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
        device="cpu",
    )


def test_torch_runtime_rejects_reported_incompatible_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Break caught: incompatible official checkpoint keys could be accepted even
    # when strict=True returns a non-empty incompatibility report.
    payload = b"verified checkpoint"
    weight_path = tmp_path / "weight.pth"
    weight_path.write_bytes(payload)
    source_identity = VerifiedSourceTree(
        root=tmp_path.resolve(),
        commit=HYPERSIGMA_SOURCE_COMMIT,
        file_sha256={},
    )
    torch_module = FakeRuntimeTorch([])
    model = FakeModel(
        [], incompatible=FakeIncompatible(missing_keys=("missing.weight",))
    )
    monkeypatch.setattr(
        adapter_module, "verify_source_tree", lambda path: source_identity
    )
    monkeypatch.setattr(adapter_module, "_build_official_model", lambda source: model)
    monkeypatch.setattr(
        adapter_module.importlib,
        "import_module",
        lambda name: torch_module if name == "torch" else _IMPORT_MODULE(name),
    )

    with pytest.raises(ValueError, match="incompatible keys"):
        _TorchRuntime(
            tmp_path,
            weight_path,
            contract=small_contract(payload),
            device="cpu",
        )


@pytest.mark.parametrize(
    ("pair", "normalization_method", "expected_code"),
    [
        (hsi_evaluation_pair(np.arange(6.0).reshape(2, 3)), "per_spectrum_range", "axis_rank"),
        (
            hsi_evaluation_pair(
                np.arange(8.0).reshape(2, 2, 2).astype(np.complex128) + 1j,
                representation="complex",
            ),
            "per_spectrum_range",
            "representation",
        ),
        (
            hsi_evaluation_pair(np.arange(4.0), representation="sparse_peaks"),
            "per_spectrum_range",
            "representation",
        ),
        (
            hsi_evaluation_pair(
                np.arange(16.0).reshape(2, 2, 2, 2),
                channel_labels=("one", "two"),
            ),
            "per_spectrum_range",
            "channel_count",
        ),
        (
            hsi_evaluation_pair(np.arange(8.0).reshape(2, 2, 2)),
            "identity_raw",
            "normalization",
        ),
    ],
)
def test_evaluator_structurally_skips_incompatible_hsi_without_runtime(
    pair: DenoisingPair,
    normalization_method: str,
    expected_code: str,
) -> None:
    # Break caught: unsupported HSI could execute the heavyweight runtime or be
    # counted as a zero-score success instead of a structured compatibility skip.
    runtime = FakeRuntime()
    adapter = HyperSIGMADenoiseAdapter(
        Path("unused-source"),
        Path("unused-weight"),
        variant="gaussian",
        _runtime=runtime,
    )

    report = evaluate_denoising_suite(
        adapter,
        (pair,),
        normalization_methods={"hyperspectral": normalization_method},
    )

    result = report.sample_results[0]
    assert result.status == "skipped"
    assert expected_code in result.issue_codes
    assert runtime.seen_shapes == []


def test_denoise_cubes_constructs_one_runtime_and_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Break caught: batched entrypoint could recreate the 2.2 GB model per cube,
    # reorder outputs, or return a partial result.
    runtime = FakeRuntime()
    construction_count = 0

    def construct_runtime(*args: object, **kwargs: object) -> FakeRuntime:
        nonlocal construction_count
        construction_count += 1
        return runtime

    monkeypatch.setattr(adapter_module, "_TorchRuntime", construct_runtime)
    first = canonical_hsi_input(np.zeros((191, 64, 64)))
    first = CanonicalDenoisingInput(
        sample_id="first",
        modality=first.modality,
        representation=first.representation,
        signal=first.signal,
        valid_mask=first.valid_mask,
        axis_values=first.axis_values,
        axis_names=("band", "y", "x"),
        axis_units=("index", "pixel", "pixel"),
        axis_directions=first.axis_directions,
        channel_labels=first.channel_labels,
        normalization_method=first.normalization_method,
        normalization_state_digest=first.normalization_state_digest,
    )
    second = CanonicalDenoisingInput(
        sample_id="second",
        modality=first.modality,
        representation=first.representation,
        signal=np.full((191, 64, 64), 0.5),
        valid_mask=first.valid_mask,
        axis_values=first.axis_values,
        axis_names=first.axis_names,
        axis_units=first.axis_units,
        axis_directions=first.axis_directions,
        channel_labels=first.channel_labels,
        normalization_method=first.normalization_method,
        normalization_state_digest="b" * 64,
    )

    outputs = denoise_cubes(
        (first, second),
        source_root=Path("source"),
        weight_path=Path("weight"),
        variant="gaussian",
    )

    assert construction_count == 1
    assert runtime.seen_shapes == [(191, 64, 64), (191, 64, 64)]
    assert tuple(output.sample_id for output in outputs) == ("first", "second")
    assert np.all(outputs[0].signal == 0.125)
    assert np.all(outputs[1].signal == 0.625)


def test_adapters_package_lazily_exposes_hypersigma_entrypoints() -> None:
    # Break caught: callers could be forced to know a private module path or the
    # adapters package could eagerly import the optional PyTorch runtime.
    assert adapters.HyperSIGMADenoiseAdapter is HyperSIGMADenoiseAdapter
    assert adapters.denoise_cubes is denoise_cubes
