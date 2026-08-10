"""The seam where a hub-published content identity reaches a readiness verdict.

The hub is growing a dataset-detail wire that answers "*which bytes was this?*"
even for corpora no quality run ever touched, and it names where the answer came
from. HyperSpectrum reads those field names off the record when they are present.
The two repos share no code, so `hyd-dataset-detail-identity.json` is the
contract: if the hub renames a field, a test here fails instead of the identity
quietly reading as absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperspectrum.hyperdata.discovery import discover_xas
from hyperspectrum.hyperdata.models import DatasetCandidate, HydCommandResult
from hyperspectrum.tasks.recommend import profile_xas_candidate, recommend_xas_tasks

DETAIL_FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "hyperdata"
    / "hyd-dataset-detail-identity.json"
)


@pytest.fixture(scope="module")
def detail_wire() -> dict[str, object]:
    """The upstream `hd dataset detail --json` contract, verbatim."""
    return json.loads(DETAIL_FIXTURE_PATH.read_text(encoding="utf-8"))


def _identity(detail_wire: dict[str, object], case: str) -> dict[str, object]:
    entry = detail_wire[case]
    assert isinstance(entry, dict)
    identity = entry["identity"]
    assert isinstance(identity, dict)
    return identity


class _Gateway:
    """Serve one record through the pinned hyd search envelope."""

    def __init__(self, record: dict[str, object]) -> None:
        self.record = record

    def _record_for(self, query: str) -> dict[str, object] | None:
        return self.record if query == "XAS" else None

    def search(self, query: str, *, page: int = 1, limit: int = 20) -> HydCommandResult:
        served = self._record_for(query)
        items = [served] if served is not None else []
        return HydCommandResult(
            argv=("hyd", "search", query),
            returncode=0,
            stdout="",
            stderr="",
            payload={
                "mode": "ilike",
                "query": query,
                "data": {
                    "items": items,
                    "total": len(items),
                    "page": page,
                    "limit": limit,
                    "next_cursor": None,
                },
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": len(items),
                    "has_next": False,
                },
            },
        )


def _record(identity: dict[str, object], **extra: object) -> dict[str, object]:
    """A search record carrying the identity fields the detail wire publishes."""
    record: dict[str, object] = {
        "id": "00000000-0000-0000-0000-000000000000",
        "dataset_code": "zenodo-identified",
        "title": "XANES reference spectra",
        "visibility": "public",
        "status": "active",
        "origin": "hub",
        "primary_format": "dat",
        "file_count": 3,
        "parsed_file_count": 3,
        "dataset_version": identity["dataset_version_id"],
        "content_digest": identity["content_digest"],
        "content_digest_source": identity["content_digest_source"],
        "quality_status": identity["quality_status"],
    }
    record.update(extra)
    return record


def _candidate(identity: dict[str, object], **extra: object) -> DatasetCandidate:
    candidates = discover_xas(_Gateway(_record(identity, **extra)))
    assert len(candidates) == 1
    return candidates[0]


def test_the_fixture_pins_the_field_names_this_repo_reads(
    detail_wire: dict[str, object],
) -> None:
    """Catches drifting away from the hub's field names without noticing."""
    provenance = detail_wire["provenance"]
    assert isinstance(provenance, dict)

    assert set(provenance["identity_block_keys"]) >= {  # type: ignore[operator]
        "dataset_version_id",
        "content_digest",
        "content_digest_source",
        "quality_status",
    }
    for case in ("hub_ingested_unverified", "quality_verified", "no_identity"):
        assert set(_identity(detail_wire, case)) == set(
            provenance["identity_block_keys"]  # type: ignore[arg-type]
        )


def test_digest_provenance_and_quality_status_survive_to_the_candidate(
    detail_wire: dict[str, object],
) -> None:
    """Catches the identity arriving stripped of where it came from."""
    identity = _identity(detail_wire, "hub_ingested_unverified")

    candidate = _candidate(identity)

    assert candidate.content_digest == identity["content_digest"]
    assert candidate.content_digest_source == "files_fingerprint"
    assert candidate.quality_status == "unverified"


def test_a_wire_without_the_new_fields_still_reads_as_absent_not_as_wrong(
    detail_wire: dict[str, object],
) -> None:
    """Catches a missing field being invented into a provenance claim.

    Today's hub search wire carries neither field. Absent must stay absent.
    """
    identity = dict(_identity(detail_wire, "hub_ingested_unverified"))
    record = _record(identity)
    del record["content_digest_source"]
    del record["quality_status"]

    candidates = discover_xas(_Gateway(record))

    assert candidates[0].content_digest == identity["content_digest"]
    assert candidates[0].content_digest_source is None
    assert candidates[0].quality_status is None


@pytest.mark.parametrize(
    ("case", "source", "status"),
    [
        ("hub_ingested_unverified", "files_fingerprint", "unverified"),
        ("quality_verified", "quality_verified", "passed"),
    ],
)
def test_an_unverified_fingerprint_identity_is_still_an_identity(
    detail_wire: dict[str, object], case: str, source: str, status: str
) -> None:
    """Catches conflating "which bytes" with "how well checked".

    A digest derived from the ingest-time files fingerprint pins the bytes just
    as a quality-verified one does. Refusing it would put every hub-ingested
    corpus back behind the identity gate the hub just opened.
    """
    identity = _identity(detail_wire, case)

    verdict = recommend_xas_tasks(profile_xas_candidate(_candidate(identity)))[0]

    assert "xas_content_digest_unavailable" not in verdict.reasons
    assert "xas_dataset_version_unavailable" not in verdict.reasons
    assert verdict.content_digest_source == source
    assert verdict.quality_status == status


def test_a_verdict_carries_the_provenance_next_to_the_digest(
    detail_wire: dict[str, object],
) -> None:
    """Catches a consumer having to guess how well the identity was established."""
    identity = _identity(detail_wire, "hub_ingested_unverified")

    verdict = recommend_xas_tasks(profile_xas_candidate(_candidate(identity)))[0]

    assert verdict.content_digest == identity["content_digest"]
    assert verdict.content_digest_source == identity["content_digest_source"]
    assert verdict.quality_status == identity["quality_status"]


def test_no_identity_on_the_wire_degrades_to_inference_only(
    detail_wire: dict[str, object],
) -> None:
    """Catches the hub's honest null being read as anything but absence."""
    identity = _identity(detail_wire, "no_identity")
    assert identity["content_digest"] is None

    verdict = recommend_xas_tasks(profile_xas_candidate(_candidate(identity)))[0]

    assert verdict.status == "inference_only"
    assert "xas_content_digest_unavailable" in verdict.reasons
    assert verdict.content_digest is None
    assert verdict.content_digest_source is None
    # The quality axis is still reported: the hub said "unverified", not "unknown".
    assert verdict.quality_status == "unverified"


def test_conflicting_provenance_across_observations_collapses_to_absent() -> None:
    """Catches picking a winner when two catalog rows disagree about provenance.

    Same rule the digest itself already follows: disagreement is not evidence.
    """
    verified = {
        "dataset_version_id": "v1",
        "content_digest": "a" * 64,
        "content_digest_source": "quality_verified",
        "quality_status": "passed",
    }

    class _TwoQueryGateway(_Gateway):
        """Serve the same dataset twice, disagreeing only about provenance."""

        def _record_for(self, query: str) -> dict[str, object] | None:
            if query == "XAS":
                return self.record
            if query == "XANES":
                return {
                    **self.record,
                    "content_digest_source": "files_fingerprint",
                    "quality_status": "unverified",
                }
            return None

    candidates = discover_xas(_TwoQueryGateway(_record(verified)))

    assert len(candidates) == 1
    assert candidates[0].content_digest == "a" * 64
    assert candidates[0].content_digest_source is None
    assert candidates[0].quality_status is None
