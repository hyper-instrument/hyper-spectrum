"""Tests for reusable agent services behind the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperspectrum import agent
from hyperspectrum.execution.plan import canonical_digest
from hyperspectrum.hyperdata import (
    HydAuthenticationError,
    HydClientNotFoundError,
    HydCommandError,
    HydIncompleteSearchError,
    HydTransportError,
    HydUnsupportedClientError,
    HydUnsupportedJsonError,
)
from hyperspectrum.hyperdata.models import DatasetCandidate, HydCommandResult

ROOT = Path(__file__).resolve().parents[2]


def candidate() -> DatasetCandidate:
    return DatasetCandidate(
        dataset_code="XAS-1",
        dataset_version="v1",
        content_digest="a" * 64,
        title="XAS candidate",
        description="",
        file_count=2,
        parsed_file_count=2,
        formats=("xdi",),
        license="CC-BY-4.0",
        evidence={
            "access_status": "admitted_catalog",
            "axis_evidence": {"energy_axis": {"valid": True, "unit": "eV"}},
            "label_evidence": {
                "verified": True,
                "ground_truth_roles": ["clean_spectrum"],
            },
            "pairing_evidence": {
                "verified": True,
                "roles": ["noisy_spectrum", "clean_spectrum"],
            },
            "observations": [
                {
                    "label_evidence": {
                        "verified": True,
                        "ground_truth_roles": ["clean_spectrum"],
                    },
                    "pairing_evidence": {
                        "verified": True,
                        "roles": ["noisy_spectrum", "clean_spectrum"],
                    },
                }
            ],
        },
    )


def benchmark_manifest_payload(
    *, dataset_code: str, dataset_version: str, declared_manifest_digest: str
) -> dict[str, object]:
    dataset_identity = {
        "id": dataset_code,
        "code": dataset_code,
        "version": dataset_version,
        "upstream_revision": None,
        "declared_manifest_digest": declared_manifest_digest,
    }
    source_content_manifest = {
        "schema_version": "hyperspectrum-source-content-manifest/v1",
        "files": [{"path": "sample.dat", "actual_size": 1, "sha256": "b" * 64}],
    }
    return {
        "schema_version": "hyperspectrum-xanes-benchmark/v1",
        "dataset": dataset_identity,
        "source_dataset_digest": canonical_digest(
            {
                "schema_version": "hyperspectrum-source-dataset-identity/v1",
                **dataset_identity,
            }
        ),
        "source_content_manifest_digest": canonical_digest(source_content_manifest),
        "benchmark_asset_digest": "3" * 64,
        "source_content_manifest": source_content_manifest,
        "reference_grid": {
            "source_path": "sample.dat",
            "source_sha256": "b" * 64,
            "point_count": 135,
            "energy_unit": "eV",
        },
        "preprocessing": {
            "schema_version": "hyperspectrum-xanes-preprocessing/v1",
            "signal": "ketek/i0",
            "interpolation": "linear",
            "endpoint_behavior": "nearest_measured_value_for_reference_endpoint_jitter",
            "smoothing": "none",
            "normalization": "none",
        },
        "noise": {
            "schema_version": "hyperspectrum-xanes-binomial-thinning/v1",
            "algorithm": "Binomial(round(count), dose_fraction) / dose_fraction",
            "dose_fraction": 0.25,
            "global_seed": 0,
            "channels": ["i0", "ketek"],
        },
        "target": {
            "semantics": "pseudo-clean frozen measurement",
            "is_physical_noiseless_ground_truth": False,
        },
        "split": {
            "schema_version": "hyperspectrum-split-manifest/v1",
            "digest": "d" * 64,
            "group_key": "composition",
            "policy_version": "xanes-zenodo-10606662-fixed/v1",
        },
        "counts": {
            "source_file_count": 1,
            "spectrum_candidate_count": 1,
            "admitted_spectrum_count": 1,
            "rejected_spectrum_count": 0,
            "excluded_file_count": 0,
            "sidecar_file_count": 0,
        },
    }


def test_doctor_can_be_ready_without_reading_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent.shutil, "which", lambda binary: "/usr/local/bin/hyd")

    response = agent.doctor()

    assert response.result["ready"] is True  # type: ignore[index]
    assert response.warnings == ()


def test_doctor_maps_failed_checks_to_readiness_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent.shutil, "which", lambda binary: None)

    with pytest.raises(agent.AgentRequestError) as captured:
        agent.doctor()

    assert captured.value.result["ready"] is False  # type: ignore[index]
    assert captured.value.exit_code == 2


def test_discovery_uses_only_the_public_gateway_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[str] = []

    class Gateway:
        def __init__(self, *, profile: str) -> None:
            constructed.append(profile)

        def search(
            self, query: str, *, page: int = 1, limit: int = 20
        ) -> HydCommandResult:
            assert (page, limit) == (1, 20)
            return HydCommandResult(
                argv=("hyd", "search", query, "--json"),
                returncode=0,
                stdout="{}",
                stderr="",
                payload={
                    "mode": "ilike",
                    "query": query,
                    "data": {
                        "items": [],
                        "total": 0,
                        "page": page,
                        "limit": limit,
                        "next_cursor": None,
                    },
                    "pagination": {
                        "page": page,
                        "limit": limit,
                        "total": 0,
                        "has_next": False,
                    },
                },
            )

    monkeypatch.setattr(agent, "HydGateway", Gateway)

    response = agent.discover_data("xas", "volcano")

    assert response.result == {
        "modality": "xas",
        "profile": "volcano",
        "queries": ["XAS", "XANES", "EXAFS", "absorption edge"],
        "completion": {
            "complete": True,
            "queries": [
                {"query": query, "pages": 1, "total": 0, "records": 0}
                for query in ("XAS", "XANES", "EXAFS", "absorption edge")
            ],
        },
        "candidates": [],
    }
    assert constructed == ["volcano"]


def test_discovery_maps_authentication_failure_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Gateway:
        def __init__(self, *, profile: str) -> None:
            _ = profile

        def search(
            self, query: str, *, page: int = 1, limit: int = 20
        ) -> HydCommandResult:
            _ = page, limit
            raise HydAuthenticationError(f"{query}: login required")

    monkeypatch.setattr(agent, "HydGateway", Gateway)

    with pytest.raises(agent.AgentAuthError, match="login required"):
        agent.discover_data("xas", "volcano")


@pytest.mark.parametrize(
    ("gateway_error", "service_error", "exit_code"),
    [
        (HydAuthenticationError("login"), agent.AgentAuthError, 3),
        (HydTransportError("timeout"), agent.AgentAuthError, 3),
        (HydClientNotFoundError("missing"), agent.AgentMissingAssetError, 4),
        (HydUnsupportedClientError("unsupported"), agent.AgentMissingAssetError, 4),
        (HydUnsupportedJsonError("bad json"), agent.AgentMissingAssetError, 4),
        (HydIncompleteSearchError("partial"), agent.AgentIncompleteSearchError, 4),
        (HydCommandError("command failed"), agent.AgentExecutionError, 5),
    ],
)
def test_discovery_maps_each_gateway_failure_category(
    monkeypatch: pytest.MonkeyPatch,
    gateway_error: Exception,
    service_error: type[agent.AgentServiceError],
    exit_code: int,
) -> None:
    class Gateway:
        def __init__(self, *, profile: str) -> None:
            _ = profile

        def search(
            self, query: str, *, page: int = 1, limit: int = 20
        ) -> HydCommandResult:
            _ = query, page, limit
            raise gateway_error

    monkeypatch.setattr(agent, "HydGateway", Gateway)

    with pytest.raises(service_error) as captured:
        agent.discover_data("xas", "volcano")

    assert captured.value.exit_code == exit_code
    if isinstance(gateway_error, HydIncompleteSearchError):
        assert captured.value.error_code == "incomplete_search"
        assert captured.value.result == {"complete": False, "candidates": []}


def test_materialize_xanes_service_uses_public_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    declaration = tmp_path / "source-declaration.json"
    declaration.write_text("{}", encoding="utf-8")
    output = tmp_path / "bundle"
    observed: dict[str, object] = {}

    class Result:
        def to_dict(self) -> dict[str, object]:
            return {
                "schema_version": "hyperspectrum-xanes-materialization-result/v1",
                "source_dataset_digest": "a" * 64,
                "source_content_manifest_digest": "b" * 64,
                "benchmark_asset_digest": "c" * 64,
            }

    def fake_materialize(**kwargs: object) -> Result:
        observed.update(kwargs)
        return Result()

    monkeypatch.setattr(agent, "materialize_xanes_spec", fake_materialize)

    response = agent.materialize_xanes(
        source_root=source_root,
        source_declaration_file=declaration,
        output_directory=output,
        dose_fraction=0.5,
        global_seed=17,
    )

    assert response.result == {
        "schema_version": "hyperspectrum-xanes-materialization-result/v1",
        "source_dataset_digest": "a" * 64,
        "source_content_manifest_digest": "b" * 64,
        "benchmark_asset_digest": "c" * 64,
    }
    assert observed == {
        "source_root": source_root,
        "source_declaration_file": declaration,
        "output_directory": output,
        "dose_fraction": 0.5,
        "global_seed": 17,
    }


def test_materialize_xanes_service_maps_invalid_source_to_request_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def reject(**kwargs: object) -> None:
        _ = kwargs
        raise agent.SourceIntegrityError("source file SHA-256 mismatch")

    monkeypatch.setattr(agent, "materialize_xanes_spec", reject)

    with pytest.raises(agent.AgentRequestError, match="SHA-256 mismatch"):
        agent.materialize_xanes(
            source_root=tmp_path / "source",
            source_declaration_file=tmp_path / "declaration.json",
            output_directory=tmp_path / "bundle",
            dose_fraction=0.25,
            global_seed=0,
        )


def test_recommendation_reads_one_declared_candidate_file(tmp_path: Path) -> None:
    source = tmp_path / "candidate.json"
    source.write_text(candidate().model_dump_json(), encoding="utf-8")

    response = agent.recommend_task(source)

    result = response.result
    assert isinstance(result, dict)
    assert result["candidate"]["dataset_code"] == "XAS-1"  # type: ignore[index]
    assert result["verdicts"][0]["status"] == "scoreable"  # type: ignore[index]
    assert result["verdicts"][0]["candidate_tasks"] == ["denoising"]  # type: ignore[index]


def test_tool_matching_reports_selected_and_blocked_evidence() -> None:
    response = agent.match_tools("xas-denoising")

    result = response.result
    assert isinstance(result, dict)
    assert [match["id"] for match in result["matches"]] == ["savgol"]  # type: ignore[index]
    blocked = {item["id"]: item for item in result["blocked"]}  # type: ignore[index]
    xasdenoise = blocked["xasdenoise"]
    assert "weights-unverified" in xasdenoise["reasons"]
    assert xasdenoise["manifest"]["source"]["commit"] == (
        "bda749ee956f9e02acc6995f238d759682ee2ca8"
    )
    assert xasdenoise["manifest"]["weights"]["asset_id"] == "zenodo-17434349"
    assert xasdenoise["manifest"]["weights"]["size_bytes"] == 780409
    assert xasdenoise["manifest"]["weights"]["digest"] == (
        "09620ee9ea0c96585f534d76ce42aa72edf2cf71e481f5737e43e93116e24160"
    )


def test_tool_matching_delegates_positive_selection_to_registry_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = agent._load_tool("savgol")
    observed: dict[str, object] = {}

    class Registry:
        def __init__(self, tools: tuple[object, ...]) -> None:
            observed["tools"] = tools

        def match(self, request: object) -> tuple[object, ...]:
            observed["request"] = request
            return (selected,)

        def rejections(self, request: object) -> tuple[object, ...]:
            observed["rejection_request"] = request
            return ()

    monkeypatch.setattr(agent, "ToolRegistry", Registry)

    response = agent.match_tools("xas-denoising")

    request = observed["request"]
    assert request == observed["rejection_request"]
    assert request.license_policy == agent.LicensePolicy.private_validation()  # type: ignore[attr-defined]
    assert request.weights_state == "not-required"  # type: ignore[attr-defined]
    assert request.resources == agent.ResourceBudget(  # type: ignore[attr-defined]
        cpu=1, memory_gb=1, gpu_available=False
    )
    assert response.result["matches"][0]["id"] == "savgol"  # type: ignore[index]


def test_mounted_xas_weight_is_verified_before_availability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from hyperspectrum.adapters import xasdenoise as adapter_module

    weight = tmp_path / "mounted.pth"
    weight.write_bytes(b"test-only")
    observed: list[Path] = []

    def verified(path: Path) -> object:
        observed.append(path)
        return object()

    monkeypatch.setattr(adapter_module, "verify_weight_asset", verified)

    availability = agent._availability_with_mounted_weights(
        agent._load_tool("xasdenoise"), (weight,)
    )

    assert availability.available is True
    assert observed == [weight]


def test_xas_availability_requires_exactly_one_mounted_weight(tmp_path: Path) -> None:
    with pytest.raises(agent.AgentMissingAssetError, match="exactly one"):
        agent._availability_with_mounted_weights(agent._load_tool("xasdenoise"), ())


def test_missing_json_input_is_a_stable_missing_asset_error(tmp_path: Path) -> None:
    with pytest.raises(agent.AgentMissingAssetError, match="candidate file"):
        agent.recommend_task(tmp_path / "absent.json")


def test_model_validator_error_remains_a_useful_request_error(tmp_path: Path) -> None:
    verdict_file = tmp_path / "invalid-verdict.json"
    verdict_file.write_text(
        json.dumps(
            {
                "dataset_code": "XAS-1",
                "dataset_version": "v1",
                "content_digest": "a" * 64,
                "status": "scoreable",
                "reasons": [],
                "candidate_tasks": [],
                "ground_truth_roles": [],
                "split_group_keys": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        agent.AgentRequestError, match="exactly one candidate task"
    ) as captured:
        agent._read_model(verdict_file, agent.ReadinessVerdict, "verdict file")

    assert captured.value.exit_code == 2


def test_plan_service_uses_public_planner_and_returns_serializable_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_file = tmp_path / "task.json"
    candidate_file = tmp_path / "candidate.json"
    verdict_file = tmp_path / "verdict.json"
    benchmark_manifest_file = tmp_path / "benchmark-manifest.json"
    task_file.write_text(
        json.dumps(
            {
                "schema_version": "hyperspectrum-task/v1",
                "id": "xas-denoising",
                "modality": "xas",
                "task_type": "denoising",
                "input_roles": ["raw_signal"],
                "output_kind": "dense_array",
                "ground_truth_roles": ["clean_spectrum"],
                "split_group_keys": ["sample_id", "compound_id"],
                "metrics": [
                    {
                        "key": "nrmse",
                        "direction": "min",
                        "aggregation": "group_mean",
                        "primary": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate_file.write_text(candidate().model_dump_json(), encoding="utf-8")
    verdict_file.write_text(
        json.dumps(
            {
                "dataset_code": "XAS-1",
                "dataset_version": "v1",
                "content_digest": "a" * 64,
                "status": "scoreable",
                "reasons": ["xas_verified_noisy_clean_pair"],
                "candidate_tasks": ["denoising"],
                "ground_truth_roles": ["clean_spectrum"],
                "split_group_keys": ["sample_id", "compound_id"],
            }
        ),
        encoding="utf-8",
    )
    benchmark_manifest_file.write_text(
        json.dumps(
            benchmark_manifest_payload(
                dataset_code="XAS-1",
                dataset_version="v1",
                declared_manifest_digest="a" * 64,
            )
        ),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    class Planned:
        plan_digest = "f" * 64

        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"schema_version": "hyperspectrum-run-plan/v3", "dry_run": True}

    def fake_build(**kwargs: object) -> Planned:
        observed.update(kwargs)
        return Planned()

    monkeypatch.setattr(agent, "build_run_plan", fake_build)

    response = agent.plan_run(
        task_file=task_file,
        candidate_file=candidate_file,
        verdict_file=verdict_file,
        benchmark_manifest_file=benchmark_manifest_file,
        tool_id="savgol",
        output_directory=tmp_path / "run",
        max_samples=8,
        sample_ids=("sample-1", "sample-2"),
        dry_run=True,
    )

    assert response.result == {
        "plan": {"schema_version": "hyperspectrum-run-plan/v3", "dry_run": True},
        "plan_digest": "f" * 64,
    }
    assert observed["dry_run"] is True
    assert observed["selected_sample_ids"] == ("sample-1", "sample-2")
    assert observed["data_origin"] == "real"
    assert observed["parameters"] == {"window_length": 5, "polyorder": 2}
    assert observed["benchmark_asset"].benchmark_asset_digest == "3" * 64  # type: ignore[attr-defined]


def test_local_service_uses_public_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from hyperspectrum.contracts import PredictionBundle
    from hyperspectrum.execution.plan import RunPlan

    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "schema_version": "hyperspectrum-run-plan/v2",
                "task": {
                    "schema_version": "hyperspectrum-task/v1",
                    "id": "xas-denoising",
                    "modality": "xas",
                    "task_type": "denoising",
                    "input_roles": ["raw_signal"],
                    "output_kind": "dense_array",
                    "ground_truth_roles": ["clean_spectrum"],
                    "split_group_keys": ["sample_id", "compound_id"],
                    "metrics": [],
                },
                "dataset_code": "XAS-1",
                "dataset_version": "v1",
                "data_digest": "a" * 64,
                "tool_id": "savgol",
                "tool_digest": "b" * 64,
                "implementation_digest": "c" * 64,
                "weight_digest": "none",
                "model_digest": "d" * 64,
                "environment_digest": "e" * 64,
                "backend": "local",
                "resources": {"cpu": 1, "memory_gb": 1, "gpu_available": False},
                "max_samples": 2,
                "selection_policy": "explicit_order",
                "selection_policy_version": "1",
                "selected_sample_ids": ["sample-1", "sample-2"],
                "output_directory": str(tmp_path / "run"),
                "dry_run": False,
                "parameters": {"window_length": 5, "polyorder": 2},
                "data_origin": "real",
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.npz"
    source.write_bytes(b"fake-npz-for-mocked-executor")
    observed: dict[str, object] = {}

    class Bundle:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"schema_version": "hyperspectrum-predictions/v1", "predictions": []}

    def fake_execute(
        plan: RunPlan,
        *,
        tool: object,
        selected_sample_ids: tuple[str, ...],
        source_npz: Path,
        weight_files: tuple[Path, ...],
    ) -> PredictionBundle:
        observed.update(
            plan=plan,
            tool=tool,
            selected_sample_ids=selected_sample_ids,
            source_npz=source_npz,
            weight_files=weight_files,
        )
        return Bundle()  # type: ignore[return-value]

    monkeypatch.setattr(agent, "execute_local_run", fake_execute)

    response = agent.run_local(
        plan_file=plan_file,
        source_npz=source,
        sample_ids=("sample-1", "sample-2"),
    )

    assert response.result == {
        "prediction_bundle": {
            "schema_version": "hyperspectrum-predictions/v1",
            "predictions": [],
        }
    }
    assert observed["selected_sample_ids"] == ("sample-1", "sample-2")
    assert observed["source_npz"] == source
    assert observed["weight_files"] == ()


def test_local_service_clearly_refuses_unbound_legacy_v1_plan(tmp_path: Path) -> None:
    plan_file = tmp_path / "legacy-plan.json"
    plan_file.write_text(
        json.dumps({"schema_version": "hyperspectrum-run-plan/v1"}),
        encoding="utf-8",
    )

    with pytest.raises(
        agent.AgentRequestError, match="v1 lacks a bound sample selection"
    ):
        agent.run_local(
            plan_file=plan_file,
            source_npz=tmp_path / "unused.npz",
            sample_ids=("sample-1",),
        )
