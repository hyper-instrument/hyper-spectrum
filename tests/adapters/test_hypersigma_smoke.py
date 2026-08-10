from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from hyperspectrum.adapters import hypersigma_smoke as smoke_module
from hyperspectrum.adapters.hypersigma_smoke import load_mat_pair, main, run_smoke
from hyperspectrum.denoising import (
    CanonicalDenoisingInput,
    CanonicalDenoisingOutput,
    ModelCapabilities,
)


class IdentityModel:
    capabilities = ModelCapabilities(
        schema_version="hyperspectrum-denoising-capabilities/v1",
        axis_ranks=(3,),
        representations=("dense",),
        channel_counts=(1,),
        required_normalization="per_spectrum_range",
        native_unit_recovery=True,
    )

    def predict(
        self,
        model_input: CanonicalDenoisingInput,
    ) -> CanonicalDenoisingOutput:
        return CanonicalDenoisingOutput(
            sample_id=model_input.sample_id,
            signal=model_input.signal,
            valid_mask=model_input.valid_mask,
            normalization_state_digest=model_input.normalization_state_digest,
        )


def test_load_mat_pair_uses_only_explicit_keys_axes_and_units(tmp_path: Path) -> None:
    # Break caught: the loader could guess a MAT key or transpose a cube from its
    # shape instead of preserving the caller's explicit declarations.
    path = tmp_path / "synthetic.mat"
    noisy = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    clean = noisy - np.float32(0.25)
    savemat(path, {"input": noisy, "gt": clean, "ignored": np.ones((9, 9))})

    pair = load_mat_pair(
        path,
        input_key="input",
        target_key="gt",
        axis_names=("band", "y", "x"),
        axis_units=("index", "pixel", "pixel"),
        signal_unit="relative_reflectance",
        fixture_kind="synthetic",
        sample_id="synthetic-pair",
        group_id="synthetic-scene",
    )

    assert pair.noisy.signal.shape == (2, 3, 4)
    assert np.array_equal(pair.noisy.signal, noisy)
    assert np.array_equal(pair.clean.signal, clean)
    assert tuple(axis.name for axis in pair.noisy.axes) == ("band", "y", "x")
    assert tuple(axis.unit for axis in pair.noisy.axes) == (
        "index",
        "pixel",
        "pixel",
    )
    assert pair.assignment.split == "test"
    assert pair.noisy.metadata["fixture_kind"] == "synthetic"
    assert pair.noisy.metadata["source_filename"] == "synthetic.mat"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"input": np.ones((2, 2, 2))}, "gt"),
        (
            {"input": np.ones((2, 2)), "gt": np.ones((2, 2))},
            "input.*three-dimensional",
        ),
        (
            {
                "input": np.ones((2, 2, 2), dtype=np.complex64),
                "gt": np.ones((2, 2, 2)),
            },
            "input.*real",
        ),
        (
            {
                "input": np.full((2, 2, 2), np.nan),
                "gt": np.ones((2, 2, 2)),
            },
            "input.*finite",
        ),
        (
            {
                "input": np.ones((2, 2, 2)),
                "gt": np.ones((2, 2, 3)),
            },
            "shapes must match",
        ),
    ],
)
def test_load_mat_pair_rejects_invalid_arrays(
    tmp_path: Path,
    payload: dict[str, np.ndarray],
    message: str,
) -> None:
    # Break caught: missing, malformed, complex, non-finite, or mismatched MAT
    # arrays could reach normalization and model execution.
    path = tmp_path / "invalid.mat"
    savemat(path, payload)

    with pytest.raises(ValueError, match=message):
        load_mat_pair(
            path,
            input_key="input",
            target_key="gt",
            axis_names=("band", "y", "x"),
            axis_units=("index", "pixel", "pixel"),
            signal_unit="relative_reflectance",
            fixture_kind="synthetic",
            sample_id="synthetic-invalid",
            group_id="synthetic-scene",
        )


def test_run_smoke_uses_unified_evaluator_without_ranking(tmp_path: Path) -> None:
    # Break caught: smoke could bypass canonical normalization/evaluation, invent
    # its own metric formula, or turn a diagnostic into a ranking result.
    path = tmp_path / "synthetic.mat"
    noisy = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    savemat(path, {"input": noisy, "gt": noisy - 0.5})

    result = run_smoke(
        source_root=Path("unused-source"),
        weight_path=Path("unused-weight"),
        variant="gaussian",
        mat_path=path,
        input_key="input",
        target_key="gt",
        axis_names=("band", "y", "x"),
        axis_units=("index", "pixel", "pixel"),
        signal_unit="relative_reflectance",
        fixture_kind="synthetic",
        sample_id="synthetic-pair",
        group_id="synthetic-scene",
        device="cpu",
        _model=IdentityModel(),
    )

    assert result["schema_version"] == "hyperspectrum-hypersigma-smoke/v1"
    assert result["diagnostic_only"] is True
    assert result["fixture_kind"] == "synthetic"
    evaluation = result["evaluation"]
    assert isinstance(evaluation, dict)
    assert evaluation["coverage"]["value"] == 1.0
    modality = evaluation["per_modality"][0]
    assert modality["normalized_rmse"] is not None
    assert modality["normalized_mae"] is not None
    assert modality["native_rmse"] is not None
    assert modality["native_mae"] is not None
    assert "ranking" not in result


def complete_cli_arguments() -> list[str]:
    return [
        "--source-root",
        "/verified/source",
        "--weight-path",
        "/verified/weight.pth",
        "--variant",
        "gaussian",
        "--mat-path",
        "/research/patch.mat",
        "--input-key",
        "input",
        "--target-key",
        "gt",
        "--axis-order",
        "band,y,x",
        "--axis-units",
        "index,pixel,pixel",
        "--signal-unit",
        "relative_reflectance",
        "--sample-id",
        "case1-patch",
        "--group-id",
        "case1",
        "--device",
        "cpu",
    ]


@pytest.mark.parametrize(
    "missing", ["--axis-order", "--axis-units", "--signal-unit"]
)
def test_cli_requires_explicit_axis_and_signal_units(missing: str) -> None:
    # Break caught: a real-data invocation could omit physical declarations and
    # fall back to shape-based inference or an unlabeled signal unit.
    arguments = complete_cli_arguments()
    index = arguments.index(missing)
    del arguments[index : index + 2]

    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--axis-order", "band,y"),
        ("--axis-order", "band,,x"),
        ("--axis-order", "unknown,y,x"),
        ("--axis-units", "index,pixel"),
        ("--axis-units", "index,,pixel"),
    ],
)
def test_cli_rejects_invalid_axis_declarations_before_reading_mat(
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    value: str,
) -> None:
    # Break caught: malformed axes could reach MAT I/O or heavyweight model
    # construction before argparse reports the declaration error.
    monkeypatch.setattr(
        smoke_module,
        "run_smoke",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("run_smoke must not be called")
        ),
    )
    arguments = complete_cli_arguments()
    arguments[arguments.index(option) + 1] = value

    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 2


def test_cli_emits_one_json_document_and_marks_research_use_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Break caught: CLI arguments could be altered, synthetic provenance could be
    # attached to real data, or progress text could corrupt machine-readable JSON.
    received: dict[str, object] = {}
    expected = {
        "schema_version": "hyperspectrum-hypersigma-smoke/v1",
        "diagnostic_only": True,
        "evaluation": {
            "coverage": {
                "value": 1.0,
                "evaluated_count": 1,
                "skipped_count": 0,
                "failed_count": 0,
                "total_count": 1,
            }
        },
    }

    def fake_run_smoke(**kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        return expected

    monkeypatch.setattr(smoke_module, "run_smoke", fake_run_smoke)

    exit_code = main(complete_cli_arguments())

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == expected
    assert captured.err == ""
    assert received["axis_names"] == ("band", "y", "x")
    assert received["axis_units"] == ("index", "pixel", "pixel")
    assert received["fixture_kind"] == "research-use-only"


def test_cli_returns_nonzero_when_diagnostic_sample_is_not_evaluated(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Break caught: a failed model invocation could produce JSON but still be
    # reported to automation as a successful smoke run.
    failed = {
        "schema_version": "hyperspectrum-hypersigma-smoke/v1",
        "diagnostic_only": True,
        "evaluation": {
            "coverage": {
                "value": 0.0,
                "evaluated_count": 0,
                "skipped_count": 0,
                "failed_count": 1,
                "total_count": 1,
            }
        },
    }
    monkeypatch.setattr(smoke_module, "run_smoke", lambda **kwargs: failed)

    exit_code = main(complete_cli_arguments())

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out) == failed
