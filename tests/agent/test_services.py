"""Tests for reusable agent services behind the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperspectrum import agent
from hyperspectrum.hyperdata import HydAuthenticationError
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

        def search(self, query: str) -> HydCommandResult:
            return HydCommandResult(
                argv=("hyd", "search", query, "--json"),
                returncode=0,
                stdout="{}",
                stderr="",
                payload={"records": []},
            )

    monkeypatch.setattr(agent, "HydGateway", Gateway)

    response = agent.discover_data("xas", "volcano")

    assert response.result == {
        "modality": "xas",
        "profile": "volcano",
        "queries": ["XAS", "XANES", "EXAFS", "absorption edge"],
        "candidates": [],
    }
    assert constructed == ["volcano"]


def test_discovery_maps_authentication_failure_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Gateway:
        def __init__(self, *, profile: str) -> None:
            _ = profile

        def search(self, query: str) -> HydCommandResult:
            raise HydAuthenticationError(f"{query}: login required")

    monkeypatch.setattr(agent, "HydGateway", Gateway)

    with pytest.raises(agent.AgentAuthError, match="login required"):
        agent.discover_data("xas", "volcano")


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
    assert "weights-required-missing" in blocked["xasdenoise"]["reasons"]


def test_missing_json_input_is_a_stable_missing_asset_error(tmp_path: Path) -> None:
    with pytest.raises(agent.AgentMissingAssetError, match="candidate file"):
        agent.recommend_task(tmp_path / "absent.json")


def test_plan_service_uses_public_planner_and_returns_serializable_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_file = tmp_path / "task.json"
    candidate_file = tmp_path / "candidate.json"
    verdict_file = tmp_path / "verdict.json"
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
    observed: dict[str, object] = {}

    class Planned:
        plan_digest = "f" * 64

        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"schema_version": "hyperspectrum-run-plan/v1", "dry_run": True}

    def fake_build(**kwargs: object) -> Planned:
        observed.update(kwargs)
        return Planned()

    monkeypatch.setattr(agent, "build_run_plan", fake_build)

    response = agent.plan_run(
        task_file=task_file,
        candidate_file=candidate_file,
        verdict_file=verdict_file,
        tool_id="savgol",
        output_directory=tmp_path / "run",
        max_samples=8,
        dry_run=True,
    )

    assert response.result == {
        "plan": {"schema_version": "hyperspectrum-run-plan/v1", "dry_run": True},
        "plan_digest": "f" * 64,
    }
    assert observed["dry_run"] is True
    assert observed["data_origin"] == "real"
    assert observed["parameters"] == {"window_length": 5, "polyorder": 2}


def test_local_service_uses_public_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from hyperspectrum.contracts import PredictionBundle
    from hyperspectrum.execution.plan import RunPlan

    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "schema_version": "hyperspectrum-run-plan/v1",
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
    ) -> PredictionBundle:
        observed.update(
            plan=plan,
            tool=tool,
            selected_sample_ids=selected_sample_ids,
            source_npz=source_npz,
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
