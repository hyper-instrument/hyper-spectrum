"""Real process-boundary tests for JSON parsing failures and secret redaction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hyperspectrum import agent
from hyperspectrum.hyperdata.models import DatasetCandidate

ROOT = Path(__file__).resolve().parents[2]
SECRET_LITERALS = (
    "BEARER_LITERAL_123",
    "TOKEN_LITERAL_456",
    "APIKEY_LITERAL_789",
    "PASSWORD_LITERAL_012",
    "QUERY_LITERAL_345",
    "CAMEL_ACCESS_LITERAL_678",
    "CAMEL_REFRESH_LITERAL_901",
    "CLIENT_SECRET_LITERAL_234",
    "USERINFO_PASSWORD_LITERAL_567",
)


def run_module(*args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        (sys.executable, "-m", "hyperspectrum.cli", *args),
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )


def assert_one_error_envelope(completed: subprocess.CompletedProcess[str]) -> None:
    assert completed.returncode == 2
    assert completed.stdout.count("\n") == 1
    envelope = json.loads(completed.stdout)
    assert set(envelope) == {
        "schema_version",
        "ok",
        "result",
        "warnings",
        "error",
    }
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "invalid_or_not_ready"
    assert "Usage:" in completed.stderr
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize(
    "args",
    [
        ("doctor", "--json", "--bogus"),
        ("data", "discover", "--json"),
        (
            "run",
            "plan",
            "--task-file",
            "task.json",
            "--candidate-file",
            "candidate.json",
            "--verdict-file",
            "verdict.json",
            "--tool-id",
            "savgol",
            "--output-directory",
            "run",
            "--max-samples",
            "0",
            "--json",
        ),
    ],
)
def test_real_module_wraps_json_mode_click_errors(args: tuple[str, ...]) -> None:
    assert_one_error_envelope(run_module(*args))


def secret_candidate() -> DatasetCandidate:
    return DatasetCandidate(
        dataset_code="XAS-SECRET",
        dataset_version="v1",
        content_digest="a" * 64,
        title="Safe title",
        description="Safe description",
        file_count=1,
        parsed_file_count=1,
        formats=("xdi",),
        license="CC-BY-4.0",
        evidence={
            "safe_context": "keep-me",
            "authorization": "Bearer BEARER_LITERAL_123",
            "token": "TOKEN_LITERAL_456",
            "nested": {
                "api-key": "APIKEY_LITERAL_789",
                "password": "PASSWORD_LITERAL_012",
                "accessToken": "CAMEL_ACCESS_LITERAL_678",
                "refreshToken": "CAMEL_REFRESH_LITERAL_901",
                "landing_page": "https://example.invalid/data?signature=QUERY_LITERAL_345&public=yes",
                "oauth_url": (
                    "https://example.invalid/data?client_secret=CLIENT_SECRET_LITERAL_234"
                ),
                "userinfo_url": (
                    "https://safe-user:USERINFO_PASSWORD_LITERAL_567@example.invalid/data"
                ),
            },
        },
    )


def test_valid_candidate_result_recursively_redacts_secrets(tmp_path: Path) -> None:
    source = tmp_path / "candidate.json"
    source.write_text(secret_candidate().model_dump_json(), encoding="utf-8")

    response = agent.recommend_task(source)
    rendered = json.dumps(response.result)

    assert all(secret not in rendered for secret in SECRET_LITERALS)
    assert "keep-me" in rendered
    assert "[REDACTED]" in rendered


def test_invalid_candidate_validation_never_includes_rejected_input(
    tmp_path: Path,
) -> None:
    raw = secret_candidate().model_dump(mode="json")
    raw["extra_secret"] = "Bearer BEARER_LITERAL_123"
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(agent.AgentRequestError) as captured:
        agent.recommend_task(source)

    rendered = f"{captured.value} {captured.value.result}"
    assert "BEARER_LITERAL_123" not in rendered
    assert "extra_secret" in rendered


def test_real_json_process_removes_secret_from_both_streams(tmp_path: Path) -> None:
    raw = secret_candidate().model_dump(mode="json")
    raw["extra_secret"] = "api_key=APIKEY_LITERAL_789"
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(raw), encoding="utf-8")

    completed = run_module(
        "task", "recommend", "--candidate-file", str(source), "--json"
    )

    assert completed.returncode == 2
    assert completed.stdout.count("\n") == 1
    assert all(
        secret not in completed.stdout + completed.stderr for secret in SECRET_LITERALS
    )


def test_real_json_parse_error_redacts_invalid_option_value() -> None:
    completed = run_module(
        "run",
        "plan",
        "--task-file",
        "task.json",
        "--candidate-file",
        "candidate.json",
        "--verdict-file",
        "verdict.json",
        "--tool-id",
        "savgol",
        "--output-directory",
        "run",
        "--max-samples",
        "api_key=PARSE_SECRET_LITERAL",
        "--json",
    )

    assert_one_error_envelope(completed)
    assert "PARSE_SECRET_LITERAL" not in completed.stdout + completed.stderr
