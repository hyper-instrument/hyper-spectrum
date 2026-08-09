from __future__ import annotations

import re

import pytest

from hyperspectrum.denoising.split import SplitEntry, SplitManifest


def test_split_manifest_binds_assignments_to_one_canonical_digest() -> None:
    # Break caught: split membership could be represented by an unbound free string.
    manifest = SplitManifest(
        schema_version="hyperspectrum-split-manifest/v1",
        entries=(
            SplitEntry("train-1", "compound-a", "train"),
            SplitEntry("val-1", "compound-b", "val"),
            SplitEntry("test-1", "compound-c", "test"),
        ),
    )

    assignment = manifest.assignment_for("test-1", "compound-c")
    assert assignment.split == "test"
    assert assignment.manifest_digest == manifest.digest
    assert re.fullmatch(r"[0-9a-f]{64}", manifest.digest)


def test_split_manifest_rejects_duplicate_samples_or_groups_crossing_splits() -> None:
    # Break caught: the same sample/group could leak into held-out evaluation.
    with pytest.raises(ValueError, match="sample IDs"):
        SplitManifest(
            schema_version="hyperspectrum-split-manifest/v1",
            entries=(
                SplitEntry("same", "compound-a", "train"),
                SplitEntry("same", "compound-a", "test"),
            ),
        )
    with pytest.raises(ValueError, match="group compound-a crosses splits"):
        SplitManifest(
            schema_version="hyperspectrum-split-manifest/v1",
            entries=(
                SplitEntry("train-1", "compound-a", "train"),
                SplitEntry("test-1", "compound-a", "test"),
            ),
        )


def test_assignment_lookup_checks_both_sample_and_group_identity() -> None:
    manifest = SplitManifest(
        schema_version="hyperspectrum-split-manifest/v1",
        entries=(SplitEntry("sample-1", "compound-a", "test"),),
    )

    with pytest.raises(ValueError, match="group identity"):
        manifest.assignment_for("sample-1", "wrong-group")
    with pytest.raises(KeyError, match="missing"):
        manifest.assignment_for("missing", "compound-a")


def test_manifest_detaches_runtime_lists_before_digesting() -> None:
    # Break caught: caller mutation could change assignments while retaining the old digest.
    entries = [SplitEntry("sample-1", "compound-a", "train")]
    manifest = SplitManifest(
        schema_version="hyperspectrum-split-manifest/v1",
        entries=entries,  # type: ignore[arg-type]
    )
    digest = manifest.digest

    entries[0] = SplitEntry("sample-1", "compound-a", "test")
    entries.append(SplitEntry("sample-2", "compound-b", "test"))

    assert manifest.assignment_for("sample-1", "compound-a").split == "train"
    assert manifest.digest == digest
    with pytest.raises(KeyError):
        manifest.assignment_for("sample-2", "compound-b")
