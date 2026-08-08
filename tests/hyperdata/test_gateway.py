from __future__ import annotations

import builtins
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from hyperspectrum.hyperdata.gateway import (
    HydAuthenticationError,
    HydClientNotFoundError,
    HydGateway,
    HydTransportError,
    HydUnsupportedClientError,
    HydUnsupportedJsonError,
)


class RecordingRunner:
    """A controlled stand-in for the single external process boundary."""

    def __init__(self, response: subprocess.CompletedProcess[str] | BaseException) -> None:
        self.response = response
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def __call__(self, args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((tuple(args), kwargs))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def completed(
    *, returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=("hyd",), returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_search_uses_the_json_contract_and_freezes_last_json_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a profile lookup, shell invocation, or mutable payload regression."""

    def fail_if_credential_file_is_read(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        raise AssertionError("the gateway must not read credential files")

    monkeypatch.setattr(builtins, "open", fail_if_credential_file_is_read)
    monkeypatch.setattr(Path, "open", fail_if_credential_file_is_read)
    monkeypatch.setattr(Path, "read_text", fail_if_credential_file_is_read)
    runner = RecordingRunner(
        completed(
            stdout=(
                "search progress\n\n"
                '{"records": [{"id": "xas-001", "tags": ["xas"]}]}\n\n'
            )
        )
    )

    result = HydGateway(profile="volcano", timeout=12.5, runner=runner).search("iron edge")

    assert result.argv == (
        "hyd",
        "--profile",
        "volcano",
        "search",
        "iron edge",
        "--ilike",
        "--json",
    )
    assert result.payload["records"][0]["id"] == "xas-001"  # type: ignore[index]
    with pytest.raises(AttributeError):
        result.payload["records"][0]["tags"].append("mutable")  # type: ignore[index,union-attr]
    assert runner.calls == [
        (
            result.argv,
            {
                "shell": False,
                "text": True,
                "capture_output": True,
                "timeout": 12.5,
            },
        )
    ]


def test_missing_hyd_binary_has_a_stable_absent_cli_code() -> None:
    """Catches collapsing a missing executable into a generic transport failure."""
    runner = RecordingRunner(FileNotFoundError("hyd was not found"))

    with pytest.raises(HydClientNotFoundError) as error:
        HydGateway(runner=runner).search("iron")

    assert error.value.code == "absent_cli"


def test_unauthenticated_profile_redacts_bearer_tokens() -> None:
    """Catches leaking an authentication secret through a nonzero CLI error."""
    runner = RecordingRunner(
        completed(
            returncode=1,
            stderr=(
                "profile volcano is not authenticated; "
                "Bearer ultra-secret-token; token another-secret-token"
            ),
        )
    )

    with pytest.raises(HydAuthenticationError) as error:
        HydGateway(profile="volcano", runner=runner).search("iron")

    assert error.value.code == "unauthenticated_profile"
    assert "ultra-secret-token" not in str(error.value)
    assert "another-secret-token" not in str(error.value)
    assert "Bearer [REDACTED]" in str(error.value)


def test_unknown_search_command_is_an_unsupported_client_without_legacy_fallback() -> None:
    """Catches a fallback to the legacy human-readable dataset-list table."""
    runner = RecordingRunner(completed(returncode=2, stderr='unknown command "search"'))

    with pytest.raises(HydUnsupportedClientError) as error:
        HydGateway(runner=runner).search("iron")

    assert error.value.code == "unsupported_client"
    assert [call[0] for call in runner.calls] == [
        ("hyd", "search", "iron", "--ilike", "--json")
    ]


def test_system_hd_binary_is_never_treated_as_hyperdata() -> None:
    """Catches confusing the platform's /usr/bin/hd utility for HyperData."""
    runner = RecordingRunner(completed(stdout='{"records": []}\n'))

    with pytest.raises(HydUnsupportedClientError) as error:
        HydGateway(binary="/usr/bin/hd", runner=runner)

    assert error.value.code == "unsupported_client"
    assert runner.calls == []


@pytest.mark.parametrize(
    "stdout",
    [
        "not json\n",
        "NAME                 ID\nIron K edge          xas-001\n",
    ],
)
def test_non_json_output_fails_closed(stdout: str) -> None:
    """Catches accepting malformed JSON or scraping columns from a human table."""
    runner = RecordingRunner(completed(stdout=stdout))

    with pytest.raises(HydUnsupportedJsonError) as error:
        HydGateway(runner=runner).search("iron")

    assert error.value.code == "unsupported_json"


def test_timeout_is_a_transport_failure() -> None:
    """Catches reporting a timed-out process as an unsupported CLI or JSON response."""
    runner = RecordingRunner(subprocess.TimeoutExpired(cmd=("hyd",), timeout=3))

    with pytest.raises(HydTransportError) as error:
        HydGateway(runner=runner).search("iron")

    assert error.value.code == "transport_failure"
