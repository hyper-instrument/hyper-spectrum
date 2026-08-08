from __future__ import annotations

import builtins
import json
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


def test_search_parses_the_pinned_pretty_json_contract_and_freezes_payload(
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
            stdout=json.dumps(
                {
                    "mode": "ilike",
                    "query": "iron edge",
                    "data": {
                        "items": [{"id": "xas-001", "tags": ["xas"]}],
                        "total": 1,
                        "page": 3,
                        "limit": 17,
                        "next_cursor": None,
                    },
                    "pagination": {
                        "page": 3,
                        "limit": 17,
                        "total": 1,
                        "has_next": False,
                    },
                },
                indent=2,
            )
            + "\n"
        )
    )

    result = HydGateway(profile="volcano", timeout=12.5, runner=runner).search(
        "iron edge", page=3, limit=17
    )

    assert result.argv == (
        "hyd",
        "--profile",
        "volcano",
        "search",
        "iron edge",
        "--limit",
        "17",
        "--page",
        "3",
        "--json",
        "--ilike",
    )
    assert result.payload["data"]["items"][0]["id"] == "xas-001"  # type: ignore[index]
    with pytest.raises(AttributeError):
        result.payload["data"]["items"][0]["tags"].append("mutable")  # type: ignore[index,union-attr]
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
        HydGateway(runner=runner).search("iron", page=1, limit=20)

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
        HydGateway(profile="volcano", runner=runner).search(
            "iron", page=1, limit=20
        )

    assert error.value.code == "unauthenticated_profile"
    assert "ultra-secret-token" not in str(error.value)
    assert "another-secret-token" not in str(error.value)
    assert "Bearer [REDACTED]" in str(error.value)


def test_unknown_search_command_is_an_unsupported_client_without_legacy_fallback() -> None:
    """Catches a fallback to the legacy human-readable dataset-list table."""
    runner = RecordingRunner(completed(returncode=2, stderr='unknown command "search"'))

    with pytest.raises(HydUnsupportedClientError) as error:
        HydGateway(runner=runner).search("iron", page=1, limit=20)

    assert error.value.code == "unsupported_client"
    assert [call[0] for call in runner.calls] == [
        (
            "hyd",
            "search",
            "iron",
            "--limit",
            "20",
            "--page",
            "1",
            "--json",
            "--ilike",
        )
    ]


def test_system_hd_binary_is_never_treated_as_hyperdata() -> None:
    """Catches confusing the platform's /usr/bin/hd utility for HyperData."""
    runner = RecordingRunner(completed(stdout='{"records": []}\n'))

    with pytest.raises(HydUnsupportedClientError) as error:
        HydGateway(binary="/usr/bin/hd", runner=runner)

    assert error.value.code == "unsupported_client"
    assert runner.calls == []


@pytest.mark.parametrize("binary", ("/bin/hd", "/usr/bin/../bin/hd"))
def test_equivalent_system_hd_paths_are_never_treated_as_hyperdata(binary: str) -> None:
    """Catches bypassing the system hd rejection through a path alias."""
    runner = RecordingRunner(completed(stdout='{"records": []}\n'))

    def resolve(candidate: str) -> str:
        _ = candidate
        return "/usr/bin/hd"

    with pytest.raises(HydUnsupportedClientError) as error:
        HydGateway(binary=binary, runner=runner, executable_resolver=resolve)

    assert error.value.code == "unsupported_client"
    assert runner.calls == []


def test_bare_hd_resolved_through_path_is_never_treated_as_hyperdata() -> None:
    """Catches running a PATH-resolved system hd utility as a HyperData client."""
    resolved: list[str] = []
    runner = RecordingRunner(completed(stdout='{"records": []}\n'))

    def resolve_from_path(candidate: str) -> str:
        resolved.append(candidate)
        return "/usr/bin/hd"

    with pytest.raises(HydUnsupportedClientError) as error:
        HydGateway(binary="hd", runner=runner, executable_resolver=resolve_from_path)

    assert error.value.code == "unsupported_client"
    assert resolved == ["hd"]
    assert runner.calls == []


def test_unresolved_bare_hd_fails_closed_without_running_a_candidate() -> None:
    """Catches allowing PATH to resolve an unverified bare hd executable later."""
    runner = RecordingRunner(completed(stdout='{"records": []}\n'))

    with pytest.raises(HydUnsupportedClientError) as error:
        HydGateway(
            binary="hd",
            runner=runner,
            executable_resolver=lambda _candidate: None,
        )

    assert error.value.code == "unsupported_client"
    assert runner.calls == []


def test_explicit_system_hd_case_alias_is_never_treated_as_hyperdata() -> None:
    """Catches a case-insensitive alias of the system utility bypassing the block."""
    runner = RecordingRunner(completed(stdout='{"records": []}\n'))

    with pytest.raises(HydUnsupportedClientError) as error:
        HydGateway(
            binary="/usr/bin/HD",
            runner=runner,
            executable_resolver=lambda _candidate: None,
        )

    assert error.value.code == "unsupported_client"
    assert runner.calls == []


def test_bare_system_hd_case_alias_resolved_through_path_is_rejected() -> None:
    """Catches PATH resolving HD to a case alias of the platform hd utility."""
    runner = RecordingRunner(completed(stdout='{"records": []}\n'))

    with pytest.raises(HydUnsupportedClientError) as error:
        HydGateway(
            binary="HD",
            runner=runner,
            executable_resolver=lambda _candidate: "/usr/bin/HD",
        )

    assert error.value.code == "unsupported_client"
    assert runner.calls == []


def test_unresolved_bare_system_hd_case_alias_fails_closed() -> None:
    """Catches case aliases reopening the unresolved bare hd PATH bypass."""
    runner = RecordingRunner(completed(stdout='{"records": []}\n'))

    with pytest.raises(HydUnsupportedClientError) as error:
        HydGateway(
            binary="HD",
            runner=runner,
            executable_resolver=lambda _candidate: None,
        )

    assert error.value.code == "unsupported_client"
    assert runner.calls == []


def test_explicit_system_hd_alias_is_rejected_even_if_a_resolver_cannot_find_it() -> None:
    """Catches letting a custom resolver bypass canonical explicit-path checks."""
    runner = RecordingRunner(completed(stdout='{"records": []}\n'))

    with pytest.raises(HydUnsupportedClientError) as error:
        HydGateway(
            binary="/usr/bin/../bin/hd",
            runner=runner,
            executable_resolver=lambda _candidate: None,
        )

    assert error.value.code == "unsupported_client"
    assert runner.calls == []


def test_resolved_custom_client_path_remains_usable() -> None:
    """Catches rejecting a legitimate custom client merely because it was resolved."""
    runner = RecordingRunner(completed(stdout='{"records": []}\n'))

    result = HydGateway(
        binary="custom-hyd",
        runner=runner,
        executable_resolver=lambda _candidate: "/opt/tools/custom-hyd",
    ).search("iron", page=1, limit=20)

    assert result.argv == (
        "custom-hyd",
        "search",
        "iron",
        "--limit",
        "20",
        "--page",
        "1",
        "--json",
        "--ilike",
    )


@pytest.mark.parametrize(("page", "limit"), ((0, 20), (True, 20), (1, 0), (1, False)))
def test_search_rejects_invalid_paging_before_process_execution(
    page: object, limit: object
) -> None:
    runner = RecordingRunner(completed(stdout='{"records": []}\n'))

    with pytest.raises(ValueError):
        HydGateway(runner=runner).search(  # type: ignore[arg-type]
            "iron", page=page, limit=limit
        )

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
        HydGateway(runner=runner).search("iron", page=1, limit=20)

    assert error.value.code == "unsupported_json"


@pytest.mark.parametrize(
    "stdout",
    (
        'diagnostic prefix\n{\n  "mode": "ilike"\n}\n',
        '{\n  "mode": "ilike"\n}\ntrailing noise\n',
        '{"mode":"ilike"}\n{"mode":"ilike"}\n',
    ),
)
def test_json_output_with_diagnostic_or_trailing_noise_fails_closed(
    stdout: str,
) -> None:
    runner = RecordingRunner(completed(stdout=stdout))

    with pytest.raises(HydUnsupportedJsonError) as captured:
        HydGateway(runner=runner).search("iron", page=1, limit=20)

    assert captured.value.code == "unsupported_json"
    assert stdout.strip() not in str(captured.value)


def test_timeout_is_a_transport_failure() -> None:
    """Catches reporting a timed-out process as an unsupported CLI or JSON response."""
    runner = RecordingRunner(subprocess.TimeoutExpired(cmd=("hyd",), timeout=3))

    with pytest.raises(HydTransportError) as error:
        HydGateway(runner=runner).search("iron", page=1, limit=20)

    assert error.value.code == "transport_failure"


def test_nonzero_timeout_uses_both_channels_and_redacts_the_http_authority() -> None:
    """Catches dropping structured stdout when stderr has a connection diagnostic."""
    placeholder_endpoint = "http://203.0.113.10:8443"
    runner = RecordingRunner(
        completed(
            returncode=1,
            stderr=(
                f"connection diagnostic for {placeholder_endpoint} "
                '{"phase":"connect"}\n'
            ),
            stdout=json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "cli-error",
                        "message": (
                            f"请求失败 ({placeholder_endpoint}): timed out"
                        ),
                        "hint": None,
                    },
                }
            )
            + "\n",
        )
    )

    with pytest.raises(HydTransportError) as captured:
        HydGateway(profile="volcano", runner=runner).search(
            "iron", page=1, limit=20
        )

    rendered = str(captured.value)
    assert captured.value.code == "transport_failure"
    assert "connection diagnostic" in rendered
    assert "timed out" in rendered
    assert placeholder_endpoint not in rendered
    assert "203.0.113.10:8443" not in rendered
    assert "http://[REDACTED]" in rendered
