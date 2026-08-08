"""Agent-facing CLI contracts and thin-wrapper tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import tomllib
from typer.testing import CliRunner

from hyperspectrum import cli
from hyperspectrum.agent import (
    AgentAuthError,
    AgentExecutionError,
    AgentMissingAssetError,
    AgentRequestError,
    ServiceResponse,
)

runner = CliRunner()


def test_project_installs_the_hyperspectrum_console_script() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["hyperspectrum"] == "hyperspectrum.cli:main"
    assert project["tool"]["uv"]["package"] is True


def payload(result: Any = None, *, warnings: tuple[str, ...] = ()) -> ServiceResponse:
    return ServiceResponse(result=result, warnings=warnings)


def assert_envelope(
    result: Any, *, ok: bool, expected_result: object = None
) -> dict[str, object]:
    assert result.stdout.count("\n") == 1
    parsed = json.loads(result.stdout)
    assert set(parsed) == {
        "schema_version",
        "ok",
        "result",
        "warnings",
        "error",
    }
    assert parsed["schema_version"] == "hyperspectrum-cli/v1"
    assert parsed["ok"] is ok
    assert parsed["result"] == expected_result
    return parsed


def test_doctor_json_is_a_thin_service_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[None] = []

    def fake_doctor() -> ServiceResponse:
        calls.append(None)
        return payload({"ready": True}, warnings=("fixture warning",))

    monkeypatch.setattr(cli.services, "doctor", fake_doctor)

    result = runner.invoke(cli.app, ["doctor", "--json"])

    assert result.exit_code == 0
    parsed = assert_envelope(result, ok=True, expected_result={"ready": True})
    assert parsed["warnings"] == ["fixture warning"]
    assert result.stderr == "fixture warning\n"
    assert calls == [None]


def test_data_discover_forwards_modality_and_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_discover(modality: str, profile: str) -> ServiceResponse:
        calls.append((modality, profile))
        return payload({"candidates": [{"dataset_code": "XAS-1"}]})

    monkeypatch.setattr(cli.services, "discover_data", fake_discover)

    result = runner.invoke(
        cli.app,
        ["data", "discover", "--modality", "xas", "--profile", "volcano", "--json"],
    )

    assert result.exit_code == 0
    assert_envelope(
        result,
        ok=True,
        expected_result={"candidates": [{"dataset_code": "XAS-1"}]},
    )
    assert calls == [("xas", "volcano")]


def test_task_recommend_forwards_candidate_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = tmp_path / "candidate.json"
    calls: list[Path] = []

    def fake_recommend(candidate_file: Path) -> ServiceResponse:
        calls.append(candidate_file)
        return payload({"verdicts": [{"status": "scoreable"}]})

    monkeypatch.setattr(cli.services, "recommend_task", fake_recommend)

    result = runner.invoke(
        cli.app, ["task", "recommend", "--candidate-file", str(candidate), "--json"]
    )

    assert result.exit_code == 0
    assert_envelope(
        result,
        ok=True,
        expected_result={"verdicts": [{"status": "scoreable"}]},
    )
    assert calls == [candidate]


def test_tools_match_forwards_task(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_match(task: str) -> ServiceResponse:
        calls.append(task)
        return payload({"matches": [{"id": "savgol"}]})

    monkeypatch.setattr(cli.services, "match_tools", fake_match)

    result = runner.invoke(
        cli.app, ["tools", "match", "--task", "xas-denoising", "--json"]
    )

    assert result.exit_code == 0
    assert_envelope(result, ok=True, expected_result={"matches": [{"id": "savgol"}]})
    assert calls == ["xas-denoising"]


def test_run_plan_forwards_only_declared_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task = tmp_path / "task.json"
    candidate = tmp_path / "candidate.json"
    verdict = tmp_path / "verdict.json"
    output = tmp_path / "run"
    calls: list[dict[str, object]] = []

    def fake_plan(**kwargs: object) -> ServiceResponse:
        calls.append(kwargs)
        return payload({"plan_digest": "a" * 64, "dry_run": True})

    monkeypatch.setattr(cli.services, "plan_run", fake_plan)

    result = runner.invoke(
        cli.app,
        [
            "run",
            "plan",
            "--task-file",
            str(task),
            "--candidate-file",
            str(candidate),
            "--verdict-file",
            str(verdict),
            "--tool-id",
            "savgol",
            "--output-directory",
            str(output),
            "--max-samples",
            "8",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert_envelope(
        result,
        ok=True,
        expected_result={"plan_digest": "a" * 64, "dry_run": True},
    )
    assert calls == [
        {
            "task_file": task,
            "candidate_file": candidate,
            "verdict_file": verdict,
            "tool_id": "savgol",
            "output_directory": output,
            "max_samples": 8,
            "dry_run": True,
        }
    ]


def test_run_local_forwards_plan_source_and_sample_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = tmp_path / "plan.json"
    source = tmp_path / "source.npz"
    calls: list[dict[str, object]] = []

    def fake_local(**kwargs: object) -> ServiceResponse:
        calls.append(kwargs)
        return payload({"prediction_count": 2})

    monkeypatch.setattr(cli.services, "run_local", fake_local)

    result = runner.invoke(
        cli.app,
        [
            "run",
            "local",
            "--plan-file",
            str(plan),
            "--source-npz",
            str(source),
            "--sample-id",
            "sample-1",
            "--sample-id",
            "sample-2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert_envelope(result, ok=True, expected_result={"prediction_count": 2})
    assert calls == [
        {
            "plan_file": plan,
            "source_npz": source,
            "sample_ids": ("sample-1", "sample-2"),
        }
    ]


@pytest.mark.parametrize(
    ("error", "exit_code", "error_code"),
    [
        (AgentRequestError("not ready"), 2, "invalid_or_not_ready"),
        (AgentAuthError("login required"), 3, "auth_or_connection"),
        (AgentMissingAssetError("tool missing"), 4, "missing_asset_or_tool"),
        (AgentExecutionError("run failed"), 5, "execution_failure"),
    ],
)
def test_agent_errors_map_to_stable_exit_codes_and_one_json_envelope(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    exit_code: int,
    error_code: str,
) -> None:
    def fail() -> ServiceResponse:
        raise error

    monkeypatch.setattr(cli.services, "doctor", fail)

    result = runner.invoke(cli.app, ["doctor", "--json"])

    assert result.exit_code == exit_code
    parsed = assert_envelope(result, ok=False)
    assert parsed["error"] == {"code": error_code, "message": str(error)}
    assert result.stderr == f"{error}\n"


def test_json_mode_never_prints_service_logs_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def noisy_doctor() -> ServiceResponse:
        print("diagnostic", file=__import__("sys").stderr)
        return payload({"ready": True})

    monkeypatch.setattr(cli.services, "doctor", noisy_doctor)

    result = runner.invoke(cli.app, ["doctor", "--json"])

    assert result.exit_code == 0
    assert_envelope(result, ok=True, expected_result={"ready": True})
    assert "diagnostic" in result.stderr
    assert capsys.readouterr().out == ""
