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
from hyperspectrum.process_boundary import redact_text

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


def test_real_doctor_starts_with_the_supported_locked_typer(
    tmp_path: Path,
) -> None:
    """Catches CLI startup imports that fail with the permitted dependency set."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    hyd = fake_bin / "hyd"
    hyd.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hyd.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        (sys.executable, "-m", "hyperspectrum.cli", "doctor", "--json"),
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.count("\n") == 1
    envelope = json.loads(completed.stdout)
    assert envelope["schema_version"] == "hyperspectrum-cli/v1"
    assert envelope["ok"] is True
    assert envelope["result"]["ready"] is True


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


@pytest.mark.parametrize(
    ("diagnostic", "secrets", "safe_context"),
    [
        (
            (
                '{"clientSecret":"QUOTED_CAMEL_LITERAL",'
                '"nested":{"refresh_token":"NESTED_OAUTH_LITERAL"},'
                '"authorization":"Bearer QUOTED_BEARER_LITERAL",'
                '"context":"keep-json"}'
            ),
            (
                "QUOTED_CAMEL_LITERAL",
                "NESTED_OAUTH_LITERAL",
                "QUOTED_BEARER_LITERAL",
            ),
            "keep-json",
        ),
        (
            (
                '{"message":"{\\"accessToken\\":'
                '\\"ESCAPED_JSON_LITERAL\\"}",'
                '"context":"keep-escaped"}'
            ),
            ("ESCAPED_JSON_LITERAL",),
            "keep-escaped",
        ),
        (
            (
                "{'client_secret': 'PY_REPR_LITERAL', "
                "'nested': {'accessToken': 'ESCAPED_LITERAL'}, "
                "'context': 'keep-repr'}"
            ),
            ("PY_REPR_LITERAL", "ESCAPED_LITERAL"),
            "keep-repr",
        ),
        (
            (
                "failure keep-url "
                "https://USERINFO_TOKEN_LITERAL@example.invalid/data "
                "https://safe-user:USERINFO_PASSWORD_LITERAL@example.invalid/data "
                "https://example.invalid/?client_secret=QUERY_OAUTH_LITERAL"
            ),
            (
                "USERINFO_TOKEN_LITERAL",
                "USERINFO_PASSWORD_LITERAL",
                "QUERY_OAUTH_LITERAL",
            ),
            "keep-url",
        ),
        (
            'malformed keep-malformed {"clientSecret":"MALFORMED_LITERAL"',
            ("MALFORMED_LITERAL",),
            "keep-malformed",
        ),
        (
            (
                'corrupted keep-nested {"authorization":"Bearer [REDACTED],'
                '"nested":{"refresh_token":"CORRUPTED_NESTED_LITERAL"}}'
            ),
            ("CORRUPTED_NESTED_LITERAL",),
            "keep-nested",
        ),
    ],
)
def test_text_redactor_handles_quoted_nested_and_malformed_diagnostics(
    diagnostic: str, secrets: tuple[str, ...], safe_context: str
) -> None:
    rendered = redact_text(diagnostic)

    assert all(secret not in rendered for secret in secrets)
    assert safe_context in rendered


def test_service_exception_redacts_quoted_diagnostic_and_result() -> None:
    error = agent.AgentExecutionError(
        ('{"clientSecret":"EXCEPTION_MESSAGE_LITERAL","context":"keep-exception"}'),
        result={
            "client_secret": "EXCEPTION_RESULT_LITERAL",
            "context": "keep-result",
        },
    )
    rendered = f"{error} {json.dumps(error.result)}"

    assert "EXCEPTION_MESSAGE_LITERAL" not in rendered
    assert "EXCEPTION_RESULT_LITERAL" not in rendered
    assert "keep-exception" in rendered
    assert "keep-result" in rendered


def test_text_redactor_does_not_raise_on_deeply_malformed_diagnostic() -> None:
    diagnostic = "keep-deep " + "[" * 2_000 + '{"clientSecret":"DEEP_LITERAL"'

    rendered = redact_text(diagnostic)

    assert "DEEP_LITERAL" not in rendered
    assert "keep-deep" in rendered


def test_structured_suffix_redacts_the_unstructured_prefix_and_mapping_keys() -> None:
    placeholder_endpoint = "http://203.0.113.10:8443"
    diagnostic = (
        f"connection to {placeholder_endpoint} failed "
        + json.dumps(
            {
                placeholder_endpoint: "endpoint-key-context",
                "token=placeholder-key-secret": "credential-key-context",
                "ok": False,
            }
        )
    )

    rendered = redact_text(diagnostic)

    assert placeholder_endpoint not in rendered
    assert "203.0.113.10:8443" not in rendered
    assert "placeholder-key-secret" not in rendered
    assert "http://[REDACTED]" in rendered
    assert "endpoint-key-context" in rendered
    assert "credential-key-context" not in rendered


def test_process_boundary_preserves_public_url_without_credentials() -> None:
    public_url = "https://catalog.example.org/public/xas-dataset"

    assert redact_text(public_url) == public_url
    assert agent.ServiceResponse(result={"catalog_url": public_url}).result == {
        "catalog_url": public_url
    }


@pytest.mark.parametrize(
    "public_locator",
    (
        "s3://public-bucket/xas/input.npz",
        "gs://public-bucket/xas/input.npz",
        "hyperdata://dataset/v1/spectrum",
        "https://catalog.example.org/public/xas-dataset",
        "https://8.8.8.8/public/xas-dataset",
    ),
)
def test_process_boundary_preserves_credential_free_public_locators(
    public_locator: str,
) -> None:
    assert redact_text(public_locator) == public_locator
    assert agent.ServiceResponse(result={"locator": public_locator}).result == {
        "locator": public_locator
    }


@pytest.mark.parametrize(
    "endpoint_url",
    (
        "http://catalog.example.org/public/xas-dataset",
        "http://8.8.8.8:8443/public/xas-dataset",
    ),
)
def test_process_boundary_redacts_every_plain_http_authority(
    endpoint_url: str,
) -> None:
    rendered = redact_text(endpoint_url)

    assert rendered != endpoint_url
    assert endpoint_url.split("/", 3)[2] not in rendered
    assert rendered.startswith("http://[REDACTED]")


@pytest.mark.parametrize(
    "endpoint_url",
    (
        "ftp://catalog.example.org/public/xas-dataset",
        "ssh://catalog.example.org/public/xas-dataset",
    ),
)
def test_process_boundary_preserves_only_approved_public_locator_schemes(
    endpoint_url: str,
) -> None:
    rendered = redact_text(endpoint_url)

    assert rendered != endpoint_url
    assert endpoint_url.split("/", 3)[2] not in rendered
    assert "://[REDACTED]" in rendered


@pytest.mark.parametrize(
    "endpoint_url",
    (
        "https://compute-node:8080/private/input",
        "https://1node:8080/private/input",
        "https://db_service:5432/private/input",
        "https://database.corp/private/input",
        "https://10.0.0.8/private/input",
        "https://203.0.113.10/private/input",
        "https://[fd00::1]:8443/private/input",
        "https://[fe80::1%25en0]:8443/private/input",
        "https://[2606:4700:4700::1111%25en0]:8443/private/input",
    ),
)
def test_process_boundary_redacts_service_private_and_reserved_url_authorities(
    endpoint_url: str,
) -> None:
    rendered = redact_text(endpoint_url)

    assert rendered != endpoint_url
    assert "https://" in rendered
    assert "[REDACTED]" in rendered


def test_real_cli_refuses_legacy_v1_plan_with_migration_message(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "legacy-plan.json"
    plan.write_text(
        json.dumps({"schema_version": "hyperspectrum-run-plan/v1"}),
        encoding="utf-8",
    )

    completed = run_module(
        "run",
        "local",
        "--plan-file",
        str(plan),
        "--source-npz",
        str(tmp_path / "unused.npz"),
        "--sample-id",
        "sample-1",
        "--json",
    )

    assert completed.returncode == 2
    envelope = json.loads(completed.stdout)
    assert envelope["error"]["code"] == "invalid_or_not_ready"
    assert "v1 lacks a bound sample selection" in envelope["error"]["message"]
    assert completed.stderr == ""
    assert "Traceback" not in completed.stderr
