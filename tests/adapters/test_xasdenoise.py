from __future__ import annotations

import io
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from hyperspectrum.adapters import xasdenoise as adapter_module
from hyperspectrum.adapters.xasdenoise import (
    XASDENOISE_CODE_LICENSE,
    XASDENOISE_REQUIRED_INPUT_ARTIFACT,
    XASDENOISE_SOURCE_COMMIT,
    XASDENOISE_STEP_INPUT_SPACE,
    XASDENOISE_WEIGHT,
    VerifiedWeightAsset,
    WeightAssetContract,
    XASDenoiseAdapter,
    XASDenoiseRuntimeIdentity,
    _TorchRuntime,
    denoise_spectra,
    load_state_dict_safely,
    preprocess_step_baseline,
    restore_step_baseline,
    verify_weight_asset,
)
from hyperspectrum.denoising import (
    CanonicalDenoisingInput,
    SpectrumAxis,
    SpectrumSample,
    normalize,
)
from hyperspectrum.plugins.xas.arrays import XASSpectrum


def test_official_source_and_weight_contract_is_exact() -> None:
    # Break caught: a similarly named checkpoint or mutable upstream revision could
    # run under the official model identity.
    assert XASDENOISE_SOURCE_COMMIT == "bda749ee956f9e02acc6995f238d759682ee2ca8"
    assert XASDENOISE_CODE_LICENSE == "MIT"
    assert XASDENOISE_WEIGHT == WeightAssetContract(
        asset_id="zenodo-17434349",
        source_url=(
            "https://zenodo.org/api/records/17434349/files/"
            "xas_denoiser_model_noise2noise_nonuniformly_sampled_notnormalized.pth/content"
        ),
        filename=(
            "xas_denoiser_model_noise2noise_nonuniformly_sampled_notnormalized.pth"
        ),
        size_bytes=780409,
        sha256="09620ee9ea0c96585f534d76ce42aa72edf2cf71e481f5737e43e93116e24160",
        license="CC-BY-4.0",
    )


def test_weight_verification_checks_size_and_digest_before_loading(
    tmp_path: Path,
) -> None:
    # Break caught: torch deserialization could occur before the mounted asset is
    # proven to be the exact declared bytes.
    payload = b"deterministic-test-state-dict"
    contract = WeightAssetContract(
        asset_id="test-asset",
        source_url="https://example.invalid/test.pth",
        filename="test.pth",
        size_bytes=len(payload),
        sha256=sha256(payload).hexdigest(),
        license="test-only",
    )
    weight_path = tmp_path / "test.pth"
    weight_path.write_bytes(payload)

    verified = verify_weight_asset(weight_path, contract=contract)

    assert verified.path == weight_path
    assert verified.size_bytes == len(payload)
    assert verified.sha256 == sha256(payload).hexdigest()
    weight_path.write_bytes(payload + b"changed")
    with pytest.raises(ValueError, match="size"):
        verify_weight_asset(weight_path, contract=contract)


def test_safe_loader_requires_weights_only_and_cpu_map_location(tmp_path: Path) -> None:
    # Break caught: a checkpoint could execute pickle payloads or allocate onto a GPU
    # before its state dictionary has been validated.
    payload = b"test"
    path = tmp_path / "state.pth"
    path.write_bytes(payload)
    contract = WeightAssetContract(
        asset_id="test",
        source_url="https://example.invalid/state.pth",
        filename="state.pth",
        size_bytes=len(payload),
        sha256=sha256(payload).hexdigest(),
        license="test-only",
    )
    verified = verify_weight_asset(path, contract=contract)
    calls: list[tuple[bytes, str, bool]] = []

    class FakeTorch:
        @staticmethod
        def load(
            supplied: io.BytesIO, *, map_location: str, weights_only: bool
        ) -> dict[str, object]:
            calls.append((supplied.getvalue(), map_location, weights_only))
            return {"encoder.0.weight": object()}

    state = load_state_dict_safely(verified, torch_module=FakeTorch())

    assert tuple(state) == ("encoder.0.weight",)
    assert calls == [(payload, "cpu", True)]


def test_safe_loader_uses_exact_verified_bytes_after_path_swap(tmp_path: Path) -> None:
    # Break caught: replacing a verified pathname between hashing and torch.load
    # could deserialize different bytes under the verified checkpoint identity.
    original = b"verified-checkpoint"
    replacement = b"swapped-checkpoint!"
    assert len(original) == len(replacement)
    path = tmp_path / "model.pth"
    path.write_bytes(original)
    contract = WeightAssetContract(
        asset_id="test",
        source_url="https://example.invalid/model.pth",
        filename="model.pth",
        size_bytes=len(original),
        sha256=sha256(original).hexdigest(),
        license="test-only",
    )
    verified = verify_weight_asset(path, contract=contract)
    path.write_bytes(replacement)
    loaded_bytes: list[bytes] = []

    class FakeTorch:
        @staticmethod
        def load(
            supplied: io.BytesIO, *, map_location: str, weights_only: bool
        ) -> dict[str, object]:
            assert map_location == "cpu"
            assert weights_only is True
            loaded_bytes.append(supplied.getvalue())
            return {"encoder.0.weight": object()}

    load_state_dict_safely(verified, torch_module=FakeTorch())

    assert verified.payload == original
    assert verified.sha256 == sha256(original).hexdigest()
    assert loaded_bytes == [original]


def test_safe_loader_rechecks_loaded_byte_digest_before_torch() -> None:
    # Break caught: a forged or incorrectly assembled verified-byte object could
    # record one digest while torch deserializes another immutable payload.
    payload = b"expected"
    contract = WeightAssetContract(
        asset_id="test",
        source_url="https://example.invalid/model.pth",
        filename="model.pth",
        size_bytes=len(payload),
        sha256=sha256(payload).hexdigest(),
        license="test-only",
    )
    forged = VerifiedWeightAsset(
        path=None,
        payload=b"tampered",
        size_bytes=len(payload),
        sha256=contract.sha256,
        contract=contract,
    )

    class FakeTorch:
        @staticmethod
        def load(*args: object, **kwargs: object) -> object:
            raise AssertionError("torch.load must not see mismatched loaded bytes")

    with pytest.raises(ValueError, match="loaded weight bytes"):
        load_state_dict_safely(forged, torch_module=FakeTorch())


def test_torch_runtime_is_strict_eval_inference_only_and_immutable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[object] = []

    class Device:
        type = "cpu"

        def __str__(self) -> str:
            return "cpu"

    class Tensor:
        def __init__(self, values: np.ndarray) -> None:
            self.values = np.array(values, copy=True)

        def __getitem__(self, index: object) -> Tensor:
            return Tensor(self.values[index])

        def to(self, device: object) -> Tensor:
            events.append(("tensor-to", str(device)))
            return self

        def detach(self) -> Tensor:
            return self

        def cpu(self) -> Tensor:
            return self

        def numpy(self) -> np.ndarray:
            return np.array(self.values, copy=True)

    class Model:
        training = True

        def __init__(self) -> None:
            self.parameter = Tensor(np.array([1.0], dtype=np.float32))
            self.output_mode = "identity"

        def load_state_dict(self, state: object, *, strict: bool) -> None:
            events.append(("load-state", state, strict))

        def to(self, device: object) -> Model:
            events.append(("model-to", str(device)))
            return self

        def eval(self) -> None:
            self.training = False
            events.append("eval")

        def state_dict(self) -> dict[str, Tensor]:
            return {"weight": self.parameter}

        def __call__(self, values: Tensor) -> Tensor:
            events.append("model-call")
            if self.output_mode == "mutate":
                self.parameter.values[0] += 1.0
            if self.output_mode == "short":
                return Tensor(values.values[..., :-1])
            if self.output_mode == "nonfinite":
                output = np.array(values.values, copy=True)
                output[..., 0] = np.nan
                return Tensor(output)
            return Tensor(values.values)

    class InferenceMode:
        def __enter__(self) -> None:
            events.append("inference-enter")

        def __exit__(self, *args: object) -> None:
            events.append("inference-exit")

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class Torch:
        __version__ = "test-torch"
        cuda = Cuda()

        @staticmethod
        def load(
            stream: io.BytesIO, *, map_location: str, weights_only: bool
        ) -> object:
            events.append(
                ("torch-load", stream.getvalue(), map_location, weights_only)
            )
            return {"weight": object()}

        @staticmethod
        def device(name: str) -> Device:
            assert name == "cpu"
            return Device()

        @staticmethod
        def from_numpy(values: np.ndarray) -> Tensor:
            return Tensor(values)

        @staticmethod
        def inference_mode() -> InferenceMode:
            return InferenceMode()

    model = Model()
    monkeypatch.setattr(adapter_module.importlib, "import_module", lambda name: Torch())
    monkeypatch.setattr(adapter_module, "_build_official_model", lambda torch: model)
    path = tmp_path / "verified.pth"
    path.write_bytes(b"verified")
    contract = WeightAssetContract(
        asset_id="test",
        source_url="https://example.invalid/test.pth",
        filename="test.pth",
        size_bytes=8,
        sha256=sha256(b"verified").hexdigest(),
        license="test-only",
    )
    verified = VerifiedWeightAsset(path, b"verified", 8, contract.sha256, contract)

    runtime = _TorchRuntime(verified, device="auto")
    values = np.linspace(0.0, 1.0, 33)
    predicted = runtime.predict(values)

    np.testing.assert_allclose(predicted, values.astype(np.float32))
    assert ("torch-load", b"verified", "cpu", True) in events
    assert any(
        isinstance(event, tuple)
        and len(event) == 3
        and event[0] == "load-state"
        and event[2] is True
        for event in events
    )
    assert any(event == "eval" for event in events)
    assert events.count("model-call") == 1
    assert events.count("inference-enter") == events.count("inference-exit") == 1
    assert runtime.device_name == "cpu"
    assert runtime.version == "test-torch"
    assert runtime.runtime_identity == XASDenoiseRuntimeIdentity(
        schema_version="hyperspectrum-xasdenoise-runtime/v1",
        requested_device="auto",
        resolved_device="cpu",
        backend="cpu",
        torch_version="test-torch",
        cuda_version=None,
        cudnn_version=None,
        loaded_weight_sha256=contract.sha256,
    )

    model.output_mode = "mutate"
    with pytest.raises(RuntimeError, match="parameters changed"):
        runtime.predict(values)
    model.parameter = Tensor(np.array([1.0], dtype=np.float32))
    model.output_mode = "short"
    with pytest.raises(RuntimeError, match="output shape"):
        runtime.predict(values)
    model.output_mode = "nonfinite"
    with pytest.raises(RuntimeError, match="non-finite"):
        runtime.predict(values)


def test_verify_entrypoint_reports_pinned_identity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter_module.main(["--verify"])

    report = json.loads(capsys.readouterr().out)
    assert report["source_commit"] == XASDENOISE_SOURCE_COMMIT
    assert report["code_license"] == "MIT"
    assert report["required_input_artifact"] == XASDENOISE_REQUIRED_INPUT_ARTIFACT
    assert report["raw_input_contract_status"] == "unverified"
    assert "required_input_normalization" not in report
    assert report["weight"]["sha256"] == XASDENOISE_WEIGHT.sha256


def test_step_baseline_state_is_versioned_recorded_and_exactly_reversible() -> None:
    # Break caught: model-native baseline removal could change output units or become
    # impossible to audit and invert.
    energy = np.linspace(5693.0, 5801.4, 135)
    step = 0.5 * (1.0 + np.tanh((energy - 5745.0) / 3.5))
    residual = 0.02 * np.sin(np.linspace(0.0, 8.0, len(energy)))
    signal = step + residual

    transformed, baseline, state = preprocess_step_baseline(energy, signal)

    np.testing.assert_allclose(transformed, signal - baseline, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        restore_step_baseline(transformed, baseline, state),
        signal,
        rtol=0.0,
        atol=1e-15,
    )
    assert state.schema_version == "hyperspectrum-xasdenoise-step-baseline/v1"
    assert state.method == "symmetric_tanh_step"
    assert state.edge_strategy == "maximum_first_derivative"
    assert state.inverse == "add_same_fitted_baseline"
    assert state.model_normalization_method is None
    assert state.normalization_method == XASDENOISE_STEP_INPUT_SPACE
    assert state.native_output_semantics == (
        "model residual plus the exact fitted baseline in upstream-normalized absorption units"
    )
    assert (
        state.baseline_sha256
        == sha256(np.ascontiguousarray(baseline, dtype="<f8").tobytes()).hexdigest()
    )
    assert state.to_dict()["model_normalization_method"] is None
    assert len(state.digest) == 64


def canonical_input(*, increasing: bool = True) -> CanonicalDenoisingInput:
    energy = np.linspace(5693.0, 5801.4, 135)
    if not increasing:
        energy = energy[::-1]
    sample = SpectrumSample(
        sample_id="xas-1",
        group_id="La08",
        modality="xas",
        representation="dense",
        axes=(
            SpectrumAxis(
                name="energy",
                unit="eV",
                direction="increasing" if increasing else "decreasing",
                values=energy,
            ),
        ),
        signal=np.linspace(0.0, 1.0, len(energy)),
        valid_mask=np.ones(len(energy), dtype=bool),
        signal_unit="ketek/i0 ratio",
        metadata={},
        provenance={},
    )
    return CanonicalDenoisingInput.from_normalized(normalize(sample, "identity_raw"))


def test_adapter_capabilities_require_structured_normalization_artifact() -> None:
    assert (
        XASDenoiseAdapter.capabilities.required_normalization
        == XASDENOISE_REQUIRED_INPUT_ARTIFACT
    )
    assert XASDenoiseAdapter.capabilities.native_unit_recovery is False
    assert (
        XASDenoiseAdapter.capabilities.check_sample(
            SpectrumSample(
                sample_id="xas-1",
                group_id="La08",
                modality="xas",
                representation="dense",
                axes=(
                    SpectrumAxis(
                        name="energy",
                        unit="eV",
                        direction="increasing",
                        values=np.linspace(5693.0, 5801.4, 135),
                    ),
                ),
                signal=np.linspace(0.0, 1.0, 135),
                valid_mask=np.ones(135, dtype=bool),
                signal_unit="ketek/i0 ratio",
                metadata={},
                provenance={},
            ),
            normalization_method="identity_raw",
        )[0].code
        == "normalization"
    )


def test_scientific_gate_rejects_raw_relabel_without_structured_artifact() -> None:
    # Break caught: replacing only a mutable method label leaves the identity_raw
    # state digest untouched and cannot prove upstream preprocessing occurred.
    raw = canonical_input()
    relabelled = replace(raw, normalization_method=XASDENOISE_STEP_INPUT_SPACE)
    assert relabelled.normalization_state_digest == raw.normalization_state_digest

    for candidate in (raw, relabelled):
        with pytest.raises(ValueError, match="input_contract_unverified"):
            XASDenoiseAdapter.validate_input(candidate)


def test_adapter_input_contract_gate_precedes_shape_validation() -> None:
    with pytest.raises(ValueError, match="input_contract_unverified"):
        XASDenoiseAdapter.validate_input(canonical_input(increasing=False))


def test_production_batch_entrypoint_fails_before_adapter_or_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Break caught: the public wrapper's private adapter hook could bypass the
    # scientific gate and publish predictions for relabelled raw arrays.
    constructed = False

    def forbidden_adapter(*args: object, **kwargs: object) -> object:
        nonlocal constructed
        constructed = True
        raise AssertionError("adapter construction and inference must remain unreachable")

    monkeypatch.setattr(adapter_module, "XASDenoiseAdapter", forbidden_adapter)
    energy = np.linspace(5693.0, 5801.4, 135)
    spectra = (
        XASSpectrum(
            sample_id="sample-a",
            group_id="La08",
            energy=energy,
            intensity=np.linspace(0.0, 1.0, len(energy)),
            energy_unit="eV",
        ),
    )

    with pytest.raises(ValueError, match="input_contract_unverified"):
        denoise_spectra(
            spectra,
            weights_path=tmp_path / "must-not-be-read.pth",
        )

    assert constructed is False
