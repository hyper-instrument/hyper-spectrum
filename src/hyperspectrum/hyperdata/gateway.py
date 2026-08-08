"""A narrow, shell-free boundary around the HyperData command-line client."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from typing import Protocol

from hyperspectrum.process_boundary import redact_text

from .models import HydCommandResult

_AUTH_MARKERS = ("not authenticated", "unauthenticated", "authentication required")
_UNSUPPORTED_MARKERS = ("unknown command", "unrecognized command", "unsupported command")
_TRANSPORT_MARKERS = (
    "timed out",
    "timeout",
    "connection refused",
    "connection reset",
    "connection aborted",
    "connection error",
    "connection failed",
    "failed to connect",
    "could not connect",
    "unable to connect",
    "network is unreachable",
    "no route to host",
    "transport error",
)
_TRANSPORT_CODES = (
    "connection-error",
    "network-error",
    "timeout",
    "transport-error",
    "transport-failure",
)
_SYSTEM_HD_BINARY = os.path.realpath("/usr/bin/hd").casefold()


class CommandRunner(Protocol):
    """The callable shape used for the one external-process dependency."""

    def __call__(
        self,
        args: Sequence[str],
        *,
        shell: bool,
        text: bool,
        capture_output: bool,
        timeout: float | None,
    ) -> subprocess.CompletedProcess[str]: ...


class ExecutableResolver(Protocol):
    """Resolve an executable path without ever invoking that executable."""

    def __call__(self, binary: str) -> str | None: ...


def _resolve_executable(binary: str) -> str | None:
    """Resolve an explicit path or PATH lookup without starting a process."""
    if os.path.dirname(binary):
        return os.path.realpath(binary)
    resolved = shutil.which(binary)
    if resolved is None:
        return None
    return os.path.realpath(resolved)


def _is_system_hd(resolved_binary: str | None) -> bool:
    """Identify only the platform hd utility after canonical path resolution."""
    if resolved_binary is None:
        return False
    return os.path.realpath(resolved_binary).casefold() == _SYSTEM_HD_BINARY


def _is_explicit_system_hd(binary: str) -> bool:
    """Compare only path-like argv values directly; bare names use PATH resolution."""
    return bool(os.path.dirname(binary)) and _is_system_hd(binary)


class HydGatewayError(RuntimeError):
    """Base error for a safe HyperData boundary failure."""

    code = "gateway_failure"

    def __init__(self, detail: str) -> None:
        super().__init__(f"{self.code}: {redact_text(detail)}")


class HydClientNotFoundError(HydGatewayError):
    """Raised when the supported `hyd` client cannot be executed."""

    code = "absent_cli"


class HydAuthenticationError(HydGatewayError):
    """Raised when a selected profile is not authenticated."""

    code = "unauthenticated_profile"


class HydUnsupportedClientError(HydGatewayError):
    """Raised when the installed client cannot provide the JSON search contract."""

    code = "unsupported_client"


class HydUnsupportedJsonError(HydGatewayError):
    """Raised when a successful command does not emit machine-readable JSON."""

    code = "unsupported_json"


class HydIncompleteSearchError(HydGatewayError):
    """Raised when catalog paging cannot prove that the search is complete."""

    code = "incomplete_search"


class HydTransportError(HydGatewayError):
    """Raised when the command cannot complete as a process transport operation."""

    code = "transport_failure"


class HydCommandError(HydGatewayError):
    """Raised for an unclassified nonzero HyperData CLI result."""

    code = "command_failed"


class HydGateway:
    """Execute only HyperData's supported machine-readable CLI contract."""

    def __init__(
        self,
        binary: str = "hyd",
        profile: str | None = None,
        *,
        timeout: float | None = 30.0,
        runner: CommandRunner | None = None,
        executable_resolver: ExecutableResolver | None = None,
    ) -> None:
        if not binary:
            raise HydClientNotFoundError("the HyperData CLI binary is empty")
        if profile is not None and not profile.strip():
            raise ValueError("profile must be a non-blank string when provided")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive when provided")

        resolver = _resolve_executable if executable_resolver is None else executable_resolver
        resolved_binary = resolver(binary)
        if (
            _is_explicit_system_hd(binary)
            or _is_system_hd(resolved_binary)
            or (binary.casefold() == "hd" and resolved_binary is None)
        ):
            raise HydUnsupportedClientError("/usr/bin/hd is not the HyperData CLI")

        self._binary = binary
        self._profile = profile
        self._timeout = timeout
        self._runner: CommandRunner = subprocess.run if runner is None else runner

    def run(self, args: Sequence[str], *, expect_json: bool = True) -> HydCommandResult:
        """Run one argument-vector command and, by default, require JSON output."""
        if any(not isinstance(arg, str) for arg in args):
            raise TypeError("HyperData command arguments must be strings")

        argv = (self._binary, *self._profile_args(), *args)
        try:
            completed = self._runner(
                argv,
                shell=False,
                text=True,
                capture_output=True,
                timeout=self._timeout,
            )
        except FileNotFoundError as error:
            raise HydClientNotFoundError(str(error)) from error
        except subprocess.TimeoutExpired as error:
            raise HydTransportError(str(error)) from error
        except OSError as error:
            raise HydTransportError(str(error)) from error

        if completed.returncode != 0:
            self._raise_for_failure(completed)

        payload = self._parse_json(completed.stdout) if expect_json else None
        return HydCommandResult(
            argv=argv,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            payload=payload,
        )

    def search(self, query: str, *, page: int, limit: int) -> HydCommandResult:
        """Search using only the current JSON search contract, never table scraping."""
        if not isinstance(query, str):
            raise TypeError("search query must be a string")
        if type(page) is not int or page < 1:
            raise ValueError("search page must be a positive integer")
        if type(limit) is not int or limit < 1:
            raise ValueError("search limit must be a positive integer")
        return self.run(
            (
                "search",
                query,
                "--limit",
                str(limit),
                "--page",
                str(page),
                "--json",
                "--ilike",
            )
        )

    def _profile_args(self) -> tuple[str, ...]:
        if self._profile is None:
            return ()
        return ("--profile", self._profile)

    @staticmethod
    def _parse_json(stdout: str) -> object:
        if not stdout.strip():
            raise HydUnsupportedJsonError("HyperData CLI returned no JSON output")
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as error:
            raise HydUnsupportedJsonError("HyperData CLI returned unsupported JSON output") from error

    @staticmethod
    def _raise_for_failure(completed: subprocess.CompletedProcess[str]) -> None:
        structured_fields = _structured_error_fields(completed.stdout)
        classification_text = "\n".join(
            part
            for part in (
                completed.stderr,
                "\n".join(structured_fields),
                completed.stdout if not structured_fields else "",
            )
            if part
        )
        normalized = classification_text.casefold()
        detail = _safe_failure_detail(completed)
        if any(marker in normalized for marker in _AUTH_MARKERS):
            raise HydAuthenticationError(detail)
        if completed.returncode == 127 or any(marker in normalized for marker in _UNSUPPORTED_MARKERS):
            raise HydUnsupportedClientError(detail)
        if any(code in _TRANSPORT_CODES for code in structured_fields) or any(
            marker in normalized for marker in _TRANSPORT_MARKERS
        ):
            raise HydTransportError(detail)
        raise HydCommandError(detail)


def _structured_error_fields(stdout: str) -> tuple[str, ...]:
    """Extract the pinned machine-error fields without trusting display text."""
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(payload, Mapping) or payload.get("ok") is not False:
        return ()
    error = payload.get("error")
    if not isinstance(error, Mapping):
        return ()
    return tuple(
        value.casefold()
        for name in ("code", "message", "hint")
        if isinstance((value := error.get(name)), str) and value.strip()
    )


def _safe_failure_detail(completed: subprocess.CompletedProcess[str]) -> str:
    """Preserve both diagnostic channels after independently redacting each one."""
    parts = tuple(
        f"{name}: {redact_text(value.strip())}"
        for name, value in (
            ("stderr", completed.stderr),
            ("stdout", completed.stdout),
        )
        if value.strip()
    )
    if parts:
        return "\n".join(parts)
    return f"HyperData exited {completed.returncode}"
