"""ACE-facing integration tests for canonical HyperSpectrum XAS execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hyperspectrum.datasets import materialize_xanes_spec
from hyperspectrum.execution import execute_ace_xas_denoising

ROOT = Path(__file__).resolve().parents[2]
REAL_HEADER = (
    ROOT / "tests/fixtures/xas/spec-real-header.txt"
).read_text(encoding="utf-8")


def _spec_bytes(*, ketek: float) -> bytes:
    energy = 5692.994 + (5801.399 - 5692.994) * np.linspace(0.0, 1.0, 135) ** 1.4
    lines = REAL_HEADER.rstrip().splitlines()
    for index, value in enumerate(energy):
        row = (
            value,
            float(index),
            1.0,
            100_000.0 + index,
            99_999.9 + index,
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


def _materialized_benchmark(
    tmp_path: Path, *, include_second_test_sample: bool = False
) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    files = {
        "Exp2_La05_heating_500_000_000.dat": _spec_bytes(ketek=1_000.0),
        "Exp4_La08_heating_800_000_000.dat": _spec_bytes(ketek=1_500.0),
    }
    if include_second_test_sample:
        files["Exp9_La08_heating_900_000_000.dat"] = _spec_bytes(ketek=1_700.0)
    declaration_files: list[dict[str, object]] = []
    for name, content in files.items():
        (source / name).write_bytes(content)
        declaration_files.append(
            {
                "path": name,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    declaration = tmp_path / "source-declaration.json"
    declaration.write_text(
        json.dumps(
            {
                "schema_version": "hyperspectrum-xanes-source-declaration/v1",
                "dataset": {
                    "id": "zenodo-10606662",
                    "code": "zenodo-10606662",
                    "version": "1.0.0",
                    "upstream_revision": None,
                    "declared_manifest_digest": "a" * 64,
                },
                "expected_file_count": len(declaration_files),
                "files": declaration_files,
            }
        ),
        encoding="utf-8",
    )
    result = materialize_xanes_spec(
        source_root=source,
        source_declaration_file=declaration,
        output_directory=tmp_path / "benchmark",
    )
    return result.output_directory


def test_ace_facade_executes_savgol_for_complete_test_split(
    tmp_path: Path,
) -> None:
    # Break caught: ACE could need to reconstruct split/readiness/plan contracts.
    benchmark = _materialized_benchmark(tmp_path)
    output = tmp_path / "run"

    result = execute_ace_xas_denoising(
        benchmark_directory=benchmark,
        output_directory=output,
        task_id="xas-denoising",
        tool_id="savgol",
        parameters={"window_length": 5, "polyorder": 2},
        expected_dose_fraction=0.25,
    )

    assert result.plan.selected_sample_ids == (
        "Exp4_La08_heating_800_000_000",
    )
    assert result.bundle.schema_version == "hyperspectrum-prediction/v3"
    assert result.bundle.failures == ()
    assert result.n_samples == 1
    assert result.artifact_paths[:2] == (
        output / "predictions.json",
        output / "run.json",
    )
    assert result.artifact_paths[2:] == tuple(
        output / prediction.uri for prediction in result.bundle.predictions
    )
    assert all(path.is_file() for path in result.artifact_paths)
    assert not (output / "metrics.json").exists()


def test_ace_facade_caps_only_the_canonical_test_split(tmp_path: Path) -> None:
    # Break caught: a smoke cap could select train rows or use unstable ordering.
    result = execute_ace_xas_denoising(
        benchmark_directory=_materialized_benchmark(
            tmp_path, include_second_test_sample=True
        ),
        output_directory=tmp_path / "run",
        task_id="xas-denoising",
        tool_id="savgol",
        parameters={"window_length": 5, "polyorder": 2},
        expected_dose_fraction=0.25,
        max_samples=1,
    )

    assert result.plan.max_samples == 1
    assert result.plan.selected_sample_ids == (
        "Exp4_La08_heating_800_000_000",
    )
    assert result.n_samples == 1


def test_ace_facade_rejects_any_sample_failure_without_metrics(
    tmp_path: Path,
) -> None:
    # Break caught: a formally incomplete run could look successful to ACE.
    output = tmp_path / "run"

    with pytest.raises(RuntimeError, match="incomplete"):
        execute_ace_xas_denoising(
            benchmark_directory=_materialized_benchmark(tmp_path),
            output_directory=output,
            task_id="xas-denoising",
            tool_id="savgol",
            parameters={"window_length": 4, "polyorder": 2},
            expected_dose_fraction=0.25,
        )

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed_with_failures"
    assert run["prediction_count"] == 0
    assert run["failure_count"] == 1
    assert not (output / "metrics.json").exists()


def test_ace_facade_rejects_a_different_track_dose_before_execution(
    tmp_path: Path,
) -> None:
    # Break caught: a model could run against a benchmark mounted for another track.
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="dose fraction"):
        execute_ace_xas_denoising(
            benchmark_directory=_materialized_benchmark(tmp_path),
            output_directory=output,
            task_id="xas-denoising",
            tool_id="savgol",
            parameters={"window_length": 5, "polyorder": 2},
            expected_dose_fraction=0.1,
        )

    assert not output.exists()


def test_ace_facade_rejects_an_unsupported_task_before_execution(
    tmp_path: Path,
) -> None:
    # Break caught: an ACE variant could relabel denoising output as another task.
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="xas-denoising"):
        execute_ace_xas_denoising(
            benchmark_directory=_materialized_benchmark(tmp_path),
            output_directory=output,
            task_id="xas-classification",
            tool_id="savgol",
            parameters={"window_length": 5, "polyorder": 2},
            expected_dose_fraction=0.25,
        )

    assert not output.exists()


@pytest.mark.parametrize("invalid_limit", [True, 0, -1, 1.5])
def test_ace_facade_rejects_invalid_sample_limits_before_execution(
    tmp_path: Path, invalid_limit: object
) -> None:
    # Break caught: bool, zero, negative, or fractional caps could select wrong rows.
    output = tmp_path / "run"

    with pytest.raises((TypeError, ValueError), match="max_samples"):
        execute_ace_xas_denoising(
            benchmark_directory=_materialized_benchmark(
                tmp_path, include_second_test_sample=True
            ),
            output_directory=output,
            task_id="xas-denoising",
            tool_id="savgol",
            parameters={"window_length": 5, "polyorder": 2},
            expected_dose_fraction=0.25,
            max_samples=invalid_limit,  # type: ignore[arg-type]
        )

    assert not output.exists()


def test_ace_facade_preserves_the_xasdenoise_scientific_blocker(
    tmp_path: Path,
) -> None:
    # Break caught: the ACE boundary could bypass HyperSpectrum's scientific gate.
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="input_contract_unverified"):
        execute_ace_xas_denoising(
            benchmark_directory=_materialized_benchmark(tmp_path),
            output_directory=output,
            task_id="xas-denoising",
            tool_id="xasdenoise",
            parameters={
                "input_contract_status": "unverified",
                "required_input_artifact": (
                    "structured_sample_bound_upstream_normalization_artifact"
                ),
                "model_normalization_method": None,
                "preprocessing": {
                    "schema_version": (
                        "hyperspectrum-xasdenoise-step-baseline/v1"
                    ),
                    "method": "symmetric_tanh_step",
                    "inverse": "add_same_fitted_baseline",
                },
                "device": "auto",
            },
            expected_dose_fraction=0.25,
        )

    assert not output.exists()
