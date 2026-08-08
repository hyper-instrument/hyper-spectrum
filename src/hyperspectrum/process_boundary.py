"""Shared redaction for every value crossing the agent process boundary."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from urllib.parse import unquote_plus

_REDACTED = "[REDACTED]"
_SENSITIVE_NAMES = (
    "authorization",
    "credential",
    "password",
    "secret",
    "signature",
    "token",
    "apikey",
    "accesskey",
    "privatekey",
)
_BEARER = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_NAMED_VALUE = re.compile(
    r"(?i)\b(?:access[_ -]?token|refresh[_ -]?token|token|api[_ -]?key|"
    r"access[_ -]?key|private[_ -]?key|client[_ -]?secret|secret|password|"
    r"signature)\b"
    r"(?:\s*[=:]\s*|\s+)[^\s,;&]+"
)
_ASSIGNED_VALUE = re.compile(r"(?i)\b([a-z][a-z0-9_-]*)(\s*[=:]\s*)([^\s,;&]+)")
_QUOTED_KEY_VALUE = re.compile(
    r"""(?ix)
    (?P<key_quote>["'])
    (?P<key>[a-z][a-z0-9_-]*)
    (?P=key_quote)
    (?P<separator>\s*:\s*)
    (?P<value_quote>["']?)
    (?P<value>(?:\\.|[^\s,}\]{"'\\])+)
    """
)
_QUERY_VALUE = re.compile(r"([?&])([^=&#\s]+)(=)([^&#\s]*)")
_URL_USERINFO = re.compile(r"(?i)(://)[^/@\s]+@")


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", unquote_plus(key).casefold())
    return any(
        normalized == name or normalized.endswith(name) for name in _SENSITIVE_NAMES
    )


def redact_text(value: str) -> str:
    """Remove credential-shaped substrings while preserving useful context."""

    structured = _redact_structured_diagnostic(value)
    if structured is not None:
        return structured
    redacted = "".join(
        _redact_structured_diagnostic(line) or line
        for line in value.splitlines(keepends=True)
    )
    redacted = _URL_USERINFO.sub(r"\1[REDACTED]@", redacted)
    redacted = _BEARER.sub("Bearer [REDACTED]", redacted)
    redacted = _NAMED_VALUE.sub(_REDACTED, redacted)
    redacted = _QUOTED_KEY_VALUE.sub(
        lambda match: (
            f"{match.group('key_quote')}{match.group('key')}"
            f"{match.group('key_quote')}{match.group('separator')}"
            f"{match.group('value_quote')}{_REDACTED}{match.group('value_quote')}"
            if _is_sensitive_key(match.group("key"))
            else match.group(0)
        ),
        redacted,
    )
    redacted = _ASSIGNED_VALUE.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2)}{_REDACTED}"
            if _is_sensitive_key(match.group(1))
            else match.group(0)
        ),
        redacted,
    )
    return _QUERY_VALUE.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2)}{match.group(3)}{_REDACTED}"
            if _is_sensitive_key(match.group(2))
            else match.group(0)
        ),
        redacted,
    )


def _redact_structured_diagnostic(value: str) -> str | None:
    """Safely redact a JSON/Python-literal diagnostic or structured suffix."""

    stripped = value.strip()
    if not stripped:
        return None
    candidates = [(stripped, "")]
    for index, character in enumerate(stripped):
        if character in "[{":
            candidates.append((stripped[index:], stripped[:index]))
            break
    for candidate, prefix in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
                if not isinstance(parsed, Mapping) and (
                    not isinstance(parsed, Sequence)
                    or isinstance(parsed, (str, bytes, bytearray))
                ):
                    continue
                rendered = json.dumps(
                    redact_value(parsed),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (
                MemoryError,
                OverflowError,
                RecursionError,
                SyntaxError,
                TypeError,
                ValueError,
            ):
                continue
            trailing_newline = "\n" if value.endswith("\n") else ""
            return f"{prefix}{rendered}{trailing_newline}"
    return None


def redact_value(value: object, *, key: str | None = None) -> object:
    """Recursively detach and redact JSON-like process-boundary values."""

    if key is not None and _is_sensitive_key(key):
        return _REDACTED
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_value(item) for item in value]
    return value
