"""Digest-bound split membership for leakage-safe spectral evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

SplitName: TypeAlias = Literal["train", "val", "test"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _nonblank(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class SplitEntry:
    """One sample/group assignment in a canonical split manifest."""

    sample_id: str
    group_id: str
    split: SplitName

    def __post_init__(self) -> None:
        _nonblank(self.sample_id, name="split sample_id")
        _nonblank(self.group_id, name="split group_id")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """One entry cryptographically bound to its complete manifest."""

    sample_id: str
    group_id: str
    split: SplitName
    manifest_digest: str

    def __post_init__(self) -> None:
        SplitEntry(self.sample_id, self.group_id, self.split)
        if _SHA256.fullmatch(self.manifest_digest) is None:
            raise ValueError("manifest_digest must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Complete deterministic assignments with sample/group leakage checks."""

    schema_version: Literal["hyperspectrum-split-manifest/v1"]
    entries: tuple[SplitEntry, ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        object.__setattr__(self, "entries", entries)
        if self.schema_version != "hyperspectrum-split-manifest/v1":
            raise ValueError("unsupported split manifest schema")
        if not entries:
            raise ValueError("split manifest entries must be non-empty")
        if any(not isinstance(entry, SplitEntry) for entry in entries):
            raise TypeError("split manifest entries must be SplitEntry values")
        sample_ids = tuple(entry.sample_id for entry in entries)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("split manifest sample IDs must be unique")
        group_splits: dict[str, str] = {}
        for entry in entries:
            previous = group_splits.setdefault(entry.group_id, entry.split)
            if previous != entry.split:
                raise ValueError(f"group {entry.group_id} crosses splits")
        payload = {
            "schema_version": self.schema_version,
            "entries": [
                {
                    "sample_id": entry.sample_id,
                    "group_id": entry.group_id,
                    "split": entry.split,
                }
                for entry in entries
            ],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "digest", digest)

    def assignment_for(self, sample_id: str, group_id: str) -> SplitAssignment:
        """Return membership only when both sample and group identity match."""
        for entry in self.entries:
            if entry.sample_id != sample_id:
                continue
            if entry.group_id != group_id:
                raise ValueError("split assignment group identity does not match")
            return SplitAssignment(
                sample_id=entry.sample_id,
                group_id=entry.group_id,
                split=entry.split,
                manifest_digest=self.digest,
            )
        raise KeyError(f"sample {sample_id} is missing from the split manifest")
