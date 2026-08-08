"""A narrow, shell-free boundary around the HyperData command-line client."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Sequence
from typing import Protocol

from .models import HydCommandResult

_AUTH_MARKERS = ("not authenticated", "unauthenticated", "authentication required")
_UNSUPPORTED_MARKERS = ("unknown command", "unrecognized command", "unsupported command")
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_NAMED_SECRET_PATTERN = re.compile(
    r"(?i)\b(?:access[_ -]?token|token|api[_ -]?key|secret|password)\b"
    r"(?:\s*[=:]\s*|\s+)[^\s,;]+"
)


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


def _redact_secrets(detail: str) -> str:
    """Prevent command diagnostics from surfacing credentials to callers."""
    redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", detail)
    return _NAMED_SECRET_PATTERN.sub("[REDACTED]", redacted)


class HydGatewayError(RuntimeError):
    """Base error for a safe HyperData boundary failure."""

    code = "gateway_failure"

    def __init__(self, detail: str) -> None:
        super().__init__(f"{self.code}: {_redact_secrets(detail)}")


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
    ) -> None:
        if binary == "/usr/bin/hd":
            raise HydUnsupportedClientError("/usr/bin/hd is not the HyperData CLI")
        if not binary:
            raise HydClientNotFoundError("the HyperData CLI binary is empty")
        if profile is not None and not profile.strip():
            raise ValueError("profile must be a non-blank string when provided")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive when provided")

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

    def search(self, query: str) -> HydCommandResult:
        """Search using only the current JSON search contract, never table scraping."""
        if not isinstance(query, str):
            raise TypeError("search query must be a string")
        return self.run(("search", query, "--ilike", "--json"))

    def _profile_args(self) -> tuple[str, ...]:
        if self._profile is None:
            return ()
        return ("--profile", self._profile)

    @staticmethod
    def _parse_json(stdout: str) -> object:
        lines = [line for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise HydUnsupportedJsonError("HyperData CLI returned no JSON output")
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError as error:
            raise HydUnsupportedJsonError("HyperData CLI returned unsupported JSON output") from error

    @staticmethod
    def _raise_for_failure(completed: subprocess.CompletedProcess[str]) -> None:
        detail = completed.stderr or completed.stdout or f"HyperData exited {completed.returncode}"
        normalized = detail.lower()
        if any(marker in normalized for marker in _AUTH_MARKERS):
            raise HydAuthenticationError(detail)
        if completed.returncode == 127 or any(marker in normalized for marker in _UNSUPPORTED_MARKERS):
            raise HydUnsupportedClientError(detail)
        raise HydCommandError(detail)
