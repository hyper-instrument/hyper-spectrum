"""Discovery behaviour against the record shape the live hub actually serves.

Every record here comes from `tests/fixtures/hyperdata/hyd-search-live-volcano-2026-08-10.json`,
a verbatim capture whose per-query SHA-256 digests equal the ones the 2026-08-10
volcano smoke recorded in `docs/evidence/xas-m0-discovery-smoke.json`.  The older
`xas-search.json` fixture describes a record shape the hub has never emitted, so
it can only prove that the extractor still reads a richer wire if one appears --
never that it reads today's.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from hyperspectrum.hyperdata.discovery import discover_xas
from hyperspectrum.hyperdata.models import DatasetCandidate, HydCommandResult
from hyperspectrum.tasks.recommend import profile_xas_candidate, recommend_xas_tasks

LIVE_FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "hyperdata"
    / "hyd-search-live-volcano-2026-08-10.json"
)

# The five hits whose title names an X-ray absorption technique as a whole word.
DOMAIN_TERM_CODES = (
    "zenodo-10606662",  # In-situ XANES, 495 .dat files
    "zenodo-15498570",  # HERFD-XAS, 109 .dat files
    "zenodo-18142209",  # EELS + XAS, .json
    "zenodo-16892323",  # XANES, .rar archive
    "zenodo-16610131",  # EXAFS, .mp4 video
)
# Ten hits where the query term is only a fragment of a longer word ("Texas",
# "hexasomes", "siloxanes") or names an adjacent technique ("NEXAFS").
COINCIDENCE_CODES = (
    "zenodo-3473148",
    "zenodo-3380560",
    "zenodo-17915983",
    "zenodo-17483574",
    "zenodo-10070373",
    "zenodo-14966054",
    "zenodo-154112",
    "zenodo-15706590",
    "zenodo-15706543",
    "zenodo-15706518",
)


class LiveWireGateway:
    """Replay the captured envelopes exactly as `hyd search --ilike --json` sent them."""

    def __init__(self, envelopes: Mapping[str, object]) -> None:
        self.envelopes = envelopes

    def search(self, query: str, *, page: int = 1, limit: int = 20) -> HydCommandResult:
        envelope = self.envelopes[query]
        assert isinstance(envelope, Mapping)
        assert page == 1, "the captured catalog fits on one page per query"
        data = envelope["data"]
        assert isinstance(data, Mapping)
        assert data["limit"] == limit
        return HydCommandResult(
            argv=("hyd", "search", query, "--ilike", "--json"),
            returncode=0,
            stdout="",
            stderr="",
            payload=envelope,
        )


@pytest.fixture(name="live_candidates")
def fixture_live_candidates() -> tuple[DatasetCandidate, ...]:
    capture = json.loads(LIVE_FIXTURE_PATH.read_text(encoding="utf-8"))
    return discover_xas(LiveWireGateway(capture["queries"]))


def _by_code(
    candidates: tuple[DatasetCandidate, ...],
) -> dict[str, DatasetCandidate]:
    return {candidate.dataset_code: candidate for candidate in candidates}


def test_live_capture_declares_the_smoke_run_it_was_recaptured_from() -> None:
    """Catches this fixture drifting away from the run whose evidence it explains."""

    capture = json.loads(LIVE_FIXTURE_PATH.read_text(encoding="utf-8"))
    smoke = json.loads(
        (
            LIVE_FIXTURE_PATH.parents[3] / "docs" / "evidence"
            / "xas-m0-discovery-smoke.json"
        ).read_text(encoding="utf-8")
    )

    assert (
        capture["response_sha256"]
        == smoke["artifact_digests"]["search_response_sha256"]
    )
    assert {
        candidate["dataset_code"] for candidate in smoke["candidates"]
    } == {
        item["dataset_code"]
        for envelope in capture["queries"].values()
        for item in envelope["data"]["items"]
    }


def test_live_records_produce_every_candidate_without_crashing(
    live_candidates: tuple[DatasetCandidate, ...],
) -> None:
    """Catches the extractor reading fixture-only names and dropping live evidence."""

    by_code = _by_code(live_candidates)

    assert set(by_code) == set(DOMAIN_TERM_CODES) | set(COINCIDENCE_CODES)

    xanes = by_code["zenodo-10606662"]
    assert xanes.formats == ("dat",)
    assert xanes.file_count == 495
    assert xanes.parsed_file_count == 491
    assert xanes.evidence["access_status"] == "admitted_catalog"
    assert xanes.evidence["source_kind"] == "zenodo"


def test_single_primary_format_is_never_reported_as_a_complete_format_list(
    live_candidates: tuple[DatasetCandidate, ...],
) -> None:
    """Catches one primary format silently standing in for the dataset's formats."""

    relevance = _by_code(live_candidates)["zenodo-10606662"].evidence["relevance"]
    assert isinstance(relevance, Mapping)

    assert relevance["primary_format"] == "dat"
    assert relevance["formats_complete"] is False
    assert relevance["format_families"] == ("spectrum_text",)


@pytest.mark.parametrize(
    ("dataset_code", "absent_field"),
    [
        (code, field)
        for code in DOMAIN_TERM_CODES
        for field in ("dataset_version", "content_digest")
    ],
)
def test_identity_fields_the_wire_omits_degrade_to_none_not_to_a_guess(
    live_candidates: tuple[DatasetCandidate, ...],
    dataset_code: str,
    absent_field: str,
) -> None:
    """Catches inventing a version or digest from a field that does not carry one."""

    assert getattr(_by_code(live_candidates)[dataset_code], absent_field) is None


def test_absent_evidence_is_declared_absent_rather_than_declared_negative(
    live_candidates: tuple[DatasetCandidate, ...],
) -> None:
    """Catches an undeclared axis or label reading as a published negative claim."""

    evidence = _by_code(live_candidates)["zenodo-10606662"].evidence

    assert evidence["axis_evidence"]["declared"] is False  # type: ignore[index]
    assert evidence["label_evidence"]["declared"] is False  # type: ignore[index]
    assert evidence["pairing_evidence"]["declared"] is False  # type: ignore[index]
    assert evidence["parser_status"]["declared"] is False  # type: ignore[index]
    assert evidence["readiness_score"] == 0


def test_live_verdicts_name_the_missing_identity_not_a_false_inaccessibility(
    live_candidates: tuple[DatasetCandidate, ...],
) -> None:
    """Catches reporting `xas_access_unavailable` for a readable public dataset."""

    candidate = _by_code(live_candidates)["zenodo-10606662"]

    verdicts = recommend_xas_tasks(profile_xas_candidate(candidate))

    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert verdict.status == "blocked"
    assert "xas_access_unavailable" not in verdict.reasons
    assert set(verdict.reasons) == {
        "xas_dataset_version_unavailable",
        "xas_content_digest_unavailable",
        "xas_energy_axis_evidence_unavailable",
    }


def test_every_live_candidate_reaches_a_verdict_without_raising(
    live_candidates: tuple[DatasetCandidate, ...],
) -> None:
    """Catches one unusual live record aborting the whole readiness pass."""

    for candidate in live_candidates:
        verdicts = recommend_xas_tasks(profile_xas_candidate(candidate))
        assert verdicts
        assert all(verdict.status == "blocked" for verdict in verdicts)
        assert all(verdict.reasons for verdict in verdicts)


def test_ranking_puts_whole_word_technique_hits_above_substring_coincidences(
    live_candidates: tuple[DatasetCandidate, ...],
) -> None:
    """Catches a degenerate all-zero ranking leaving a flooding study first."""

    ranked = [candidate.dataset_code for candidate in live_candidates]

    assert set(ranked[:5]) == set(DOMAIN_TERM_CODES)
    for coincidence in COINCIDENCE_CODES:
        assert ranked.index(coincidence) >= 5
    # The two large .dat spectroscopy corpora lead; the flooding study that the
    # degenerate all-zero ranking left first is now below every real hit.
    assert set(ranked[:2]) == {"zenodo-15498570", "zenodo-10606662"}
    assert ranked.index("zenodo-3473148") > ranked.index("zenodo-10606662")


def test_coincidence_hits_sink_on_lexical_and_format_evidence(
    live_candidates: tuple[DatasetCandidate, ...],
) -> None:
    """Catches promoting a substring hit that carries no spectrum-bearing format."""

    by_code = _by_code(live_candidates)
    flood_relevance = by_code["zenodo-3473148"].evidence["relevance"]
    xanes_relevance = by_code["zenodo-10606662"].evidence["relevance"]
    assert isinstance(flood_relevance, Mapping)
    assert isinstance(xanes_relevance, Mapping)

    # The separation is lexical and format-based, not nominal: 'XAS' sits inside
    # 'Texas', and a spreadsheet is not a spectrum carrier.
    assert flood_relevance["term_match"] == "affix_term"
    assert xanes_relevance["term_match"] == "exact_term"
    assert flood_relevance["format_families"] == ("structured_document",)
    assert xanes_relevance["format_families"] == ("spectrum_text",)
    assert flood_relevance["score"] < xanes_relevance["score"]


def test_ranking_follows_the_evidence_when_two_records_swap_identities(
    live_candidates: tuple[DatasetCandidate, ...],
) -> None:
    """Catches a hardcoded dataset-name or dataset-code blocklist doing the sinking."""

    capture = json.loads(LIVE_FIXTURE_PATH.read_text(encoding="utf-8"))
    envelopes = json.loads(json.dumps(capture["queries"]))
    items = envelopes["XAS"]["data"]["items"]
    flood = next(item for item in items if item["dataset_code"] == "zenodo-3473148")
    herfd = next(item for item in items if item["dataset_code"] == "zenodo-15498570")
    assert [candidate.dataset_code for candidate in live_candidates].index(
        "zenodo-15498570"
    ) < [candidate.dataset_code for candidate in live_candidates].index("zenodo-3473148")

    # Keep both dataset codes where they are; move every field the ranking is
    # allowed to read across to the other record.
    for field in ("title", "primary_format", "file_count", "parsed_file_count"):
        flood[field], herfd[field] = herfd[field], flood[field]

    ranked = [
        candidate.dataset_code for candidate in discover_xas(LiveWireGateway(envelopes))
    ]

    assert ranked.index("zenodo-3473148") < ranked.index("zenodo-15498570")


def test_term_evidence_can_never_outrank_declared_task_evidence(
    live_candidates: tuple[DatasetCandidate, ...],
) -> None:
    """Catches lexical relevance being promoted above verified ground truth."""

    capture = json.loads(LIVE_FIXTURE_PATH.read_text(encoding="utf-8"))
    envelopes = {query: json.loads(json.dumps(value)) for query, value in capture["queries"].items()}
    unnamed_but_evidenced = {
        "id": "00000000-0000-0000-0000-000000000000",
        "dataset_code": "plain-but-evidenced",
        "title": "Plain catalog entry",
        "visibility": "public",
        "status": "active",
        "origin": "upload",
        "primary_format": "xlsx",
        "file_count": 1,
        "parsed_file_count": 0,
        "parser_status": {"kind": "spectroscopy", "valid": True},
        "axis_evidence": {"energy_axis": {"valid": True, "unit": "eV"}},
        "label_evidence": {"verified": True, "ground_truth_roles": ["oxidation_state"]},
        "pairing_evidence": {"verified": True, "roles": ["structure", "spectrum"]},
    }
    envelopes["XAS"]["data"]["items"].append(unnamed_but_evidenced)
    envelopes["XAS"]["data"]["total"] += 1
    envelopes["XAS"]["pagination"]["total"] += 1

    ranked = [
        candidate.dataset_code for candidate in discover_xas(LiveWireGateway(envelopes))
    ]

    assert ranked[0] == "plain-but-evidenced"
