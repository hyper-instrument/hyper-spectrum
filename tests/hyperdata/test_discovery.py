from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from hyperspectrum.hyperdata.discovery import XAS_QUERIES, discover_xas
from hyperspectrum.hyperdata.gateway import HydUnsupportedJsonError
from hyperspectrum.hyperdata.models import HydCommandResult

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "hyperdata" / "xas-search.json"
DATA_ENVELOPE_FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "hyperdata"
    / "hyd-search-data-envelope.json"
)


class FixtureGateway:
    """Internal exact records-fixture compatibility, not a real hyd wire shape."""

    def __init__(self, records_by_query: Mapping[str, list[dict[str, object]]]) -> None:
        self.records_by_query = records_by_query
        self.queries: list[str] = []

    def search(self, query: str) -> HydCommandResult:
        self.queries.append(query)
        return HydCommandResult(
            argv=("hyd", "search", query, "--ilike", "--json"),
            returncode=0,
            stdout="",
            stderr="",
            payload={"records": self.records_by_query.get(query, [])},
        )


class EnvelopeGateway:
    """Return exact JSON envelopes at the immutable command-result boundary."""

    def __init__(self, payloads_by_query: Mapping[str, object]) -> None:
        self.payloads_by_query = payloads_by_query
        self.queries: list[str] = []

    def search(self, query: str) -> HydCommandResult:
        self.queries.append(query)
        return HydCommandResult(
            argv=("hyd", "search", query, "--ilike", "--json"),
            returncode=0,
            stdout="",
            stderr="",
            payload=self.payloads_by_query.get(
                query,
                {
                    "mode": "ilike",
                    "query": query,
                    "data": {"items": [], "total": 0, "page": 1, "limit": 20},
                    "pagination": {
                        "page": 1,
                        "limit": 20,
                        "total": 0,
                        "has_next": False,
                    },
                },
            ),
        )


def load_fixture() -> dict[str, dict[str, list[dict[str, object]]]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_discovery_reads_the_hyd_0_11_data_envelope() -> None:
    """Reading only the old records key would silently erase real catalog hits."""

    payload = json.loads(DATA_ENVELOPE_FIXTURE_PATH.read_text(encoding="utf-8"))
    gateway = EnvelopeGateway({"XAS": payload})

    candidates = discover_xas(gateway)

    assert gateway.queries == list(XAS_QUERIES)
    assert [candidate.dataset_code for candidate in candidates] == [
        "public-real-envelope-xas"
    ]
    assert candidates[0].dataset_version == "2026.08.1"
    assert candidates[0].content_digest == "a" * 64


def test_discovery_retains_records_envelope_compatibility() -> None:
    """Removing the tested records compatibility would break cached test contracts."""

    record = json.loads(DATA_ENVELOPE_FIXTURE_PATH.read_text(encoding="utf-8"))["data"][
        "items"
    ][0]
    candidates = discover_xas(FixtureGateway({"XAS": [record]}))

    assert [candidate.dataset_code for candidate in candidates] == [
        "public-real-envelope-xas"
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"mode": "ilike", "query": "XAS", "pagination": {}},
        {
            "mode": "dataset",
            "query": "XAS",
            "data": {"items": [], "total": 0, "page": 1, "limit": 20},
            "pagination": {"page": 1, "limit": 20, "total": 0, "has_next": False},
        },
        {
            "mode": "ilike",
            "query": "XANES",
            "data": {"items": [], "total": 0, "page": 1, "limit": 20},
            "pagination": {"page": 1, "limit": 20, "total": 0, "has_next": False},
        },
        {
            "mode": "ilike",
            "query": "XAS",
            "data": {"items": [42], "total": 1, "page": 1, "limit": 20},
            "pagination": {"page": 1, "limit": 20, "total": 1, "has_next": False},
        },
        {"records": {}},
        {"records": [], "extra": True},
        {
            "mode": "ilike",
            "query": "XAS",
            "data": {"items": [], "total": 0, "page": 1, "limit": 20},
            "pagination": {"page": 1, "limit": 20, "total": 0, "has_next": False},
            "records": [],
        },
    ],
)
def test_discovery_rejects_unknown_json_shapes(payload: object) -> None:
    """Returning zero candidates for an unknown shape would hide a client drift."""

    with pytest.raises(HydUnsupportedJsonError) as error:
        discover_xas(EnvelopeGateway({"XAS": payload}))

    assert error.value.code == "unsupported_json"


@pytest.mark.parametrize(
    "change",
    [
        {"data": {"items": [], "total": 0, "page": 1}},
        {
            "data": {
                "items": [],
                "total": 0,
                "page": 1,
                "limit": 20,
                "extra": True,
            }
        },
        {"data": {"items": [], "total": True, "page": 1, "limit": 20}},
        {"pagination": {"page": 1, "limit": 20, "total": 0}},
        {
            "pagination": {
                "page": 1,
                "limit": 20,
                "total": 0,
                "has_next": 0,
            }
        },
        {
            "pagination": {
                "page": 2,
                "limit": 20,
                "total": 0,
                "has_next": False,
            }
        },
        {
            "pagination": {
                "page": 1,
                "limit": 10,
                "total": 0,
                "has_next": False,
            }
        },
        {
            "pagination": {
                "page": 1,
                "limit": 20,
                "total": 3,
                "has_next": False,
            }
        },
    ],
)
def test_discovery_rejects_invalid_real_hyd_pagination_contract(
    change: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "mode": "ilike",
        "query": "XAS",
        "data": {"items": [], "total": 0, "page": 1, "limit": 20},
        "pagination": {"page": 1, "limit": 20, "total": 0, "has_next": False},
    }
    payload.update(change)

    with pytest.raises(HydUnsupportedJsonError):
        discover_xas(EnvelopeGateway({"XAS": payload}))


def test_discovery_allows_empty_data_only_in_the_exact_real_envelope() -> None:
    candidates = discover_xas(EnvelopeGateway({}))

    assert candidates == ()


def test_discovery_allows_documented_null_pagination_total() -> None:
    payload = {
        "mode": "ilike",
        "query": "XAS",
        "data": {"items": [], "total": 0, "page": 1, "limit": 20},
        "pagination": {"page": 1, "limit": 20, "total": None, "has_next": False},
    }

    assert discover_xas(EnvelopeGateway({"XAS": payload})) == ()


def test_discovery_deduplicates_catalog_hits_and_retains_search_evidence() -> None:
    """Catches dropping duplicate-query provenance or external access distinction."""
    fixture = load_fixture()
    gateway = FixtureGateway(fixture["queries"])

    candidates = discover_xas(gateway)

    assert XAS_QUERIES == ("XAS", "XANES", "EXAFS", "absorption edge")
    assert gateway.queries == list(XAS_QUERIES)
    assert {candidate.dataset_code for candidate in candidates} == {
        "zenodo-10606662",
        "zenodo-16892323",
        "zenodo-154112",
        "zenodo-16610131",
        "zenodo-19681499",
        "zenodo-7096004",
        "zenodo-7868979",
        "zenodo-14917057",
        "zenodo-10044134",
    }

    archive = next(candidate for candidate in candidates if candidate.dataset_code == "zenodo-16892323")
    assert archive.evidence["source_queries"] == ("XAS", "XANES", "EXAFS")
    assert archive.evidence["parser_status"]["kind"] == "archive_only"  # type: ignore[index]
    assert archive.evidence["axis_evidence"]["energy_axis"]["valid"] is False  # type: ignore[index]
    assert archive.evidence["ground_truth_roles"] == ()
    assert archive.evidence["license"] is None

    proposal = next(candidate for candidate in candidates if candidate.dataset_code == "zenodo-19681499")
    assert proposal.evidence["source_kind"] == "external_discovery"
    assert proposal.evidence["access_status"] == "external_discovery_proposal"
    assert "locator" not in proposal.evidence
    with pytest.raises(TypeError):
        proposal.evidence["source_kind"] = "admitted_catalog"  # type: ignore[index]


def test_discovery_ranks_explicit_evidence_above_xas_words_in_a_title() -> None:
    """Catches using dataset names instead of labels, pairings, axes, and parser evidence."""
    plain_but_evidenced = {
        "dataset_code": "plain-evidenced",
        "dataset_version": None,
        "content_digest": None,
        "title": "Plain catalog entry",
        "description": "",
        "file_count": 2,
        "parsed_file_count": 2,
        "formats": ["DAT"],
        "license": "CC-BY-4.0",
        "parser_status": {"kind": "spectroscopy", "valid": True},
        "axis_evidence": {"energy_axis": {"valid": True, "unit": "eV"}},
        "label_evidence": {"verified": True, "ground_truth_roles": ["oxidation_state"]},
        "pairing_evidence": {"verified": True, "roles": ["structure", "spectrum"]},
        "access_status": "admitted_catalog",
        "source_kind": "volcano_catalog",
    }
    xas_named_but_unevidenced = {
        **plain_but_evidenced,
        "dataset_code": "xas-named-only",
        "title": "XAS XANES EXAFS absorption edge",
        "parser_status": {"kind": "generic_text", "valid": False},
        "axis_evidence": {"energy_axis": {"valid": False, "unit": None}},
        "label_evidence": {"verified": False, "ground_truth_roles": []},
        "pairing_evidence": {"verified": False, "roles": []},
    }
    gateway = FixtureGateway(
        {query: [plain_but_evidenced, xas_named_but_unevidenced] for query in XAS_QUERIES}
    )

    candidates = discover_xas(gateway)

    assert [candidate.dataset_code for candidate in candidates] == [
        "plain-evidenced",
        "xas-named-only",
    ]
    assert candidates[0].evidence["readiness_score"] > candidates[1].evidence["readiness_score"]


def test_equal_evidence_keeps_the_documented_search_observation_order() -> None:
    """Catches using a dataset code or title as an undocumented score tiebreaker."""
    observation = {
        "dataset_version": None,
        "content_digest": None,
        "title": "Same evidence",
        "description": "",
        "file_count": 1,
        "parsed_file_count": 0,
        "formats": ["DAT"],
        "license": None,
        "parser_status": {"kind": "generic_text", "valid": False},
        "axis_evidence": {"energy_axis": {"valid": False, "unit": None}},
        "label_evidence": {"verified": False, "ground_truth_roles": []},
        "pairing_evidence": {"verified": False, "roles": []},
        "access_status": "admitted_catalog",
        "source_kind": "volcano_catalog",
    }
    gateway = FixtureGateway(
        {
            "XAS": [
                {**observation, "dataset_code": "z-seen-first"},
                {**observation, "dataset_code": "a-would-sort-first"},
            ]
        }
    )

    candidates = discover_xas(gateway)

    assert [candidate.dataset_code for candidate in candidates] == [
        "z-seen-first",
        "a-would-sort-first",
    ]


def test_duplicate_hits_merge_all_evidence_and_keep_misparse_conservative() -> None:
    """Catches query-order loss of duplicate parser, axis, role, or access evidence."""
    first_duplicate_hit = {
        "dataset_code": "duplicate",
        "dataset_version": None,
        "content_digest": None,
        "title": "Duplicate first observation",
        "description": "",
        "file_count": 2,
        "parsed_file_count": 2,
        "formats": ["DAT"],
        "license": "CC-BY-4.0",
        "parser_status": {"kind": "spectroscopy", "valid": True},
        "axis_evidence": {"energy_axis": {"valid": True, "unit": "eV"}},
        "label_evidence": {"verified": True, "ground_truth_roles": ["oxidation_state"]},
        "pairing_evidence": {"verified": True, "roles": ["structure"]},
        "access_status": "admitted_catalog",
        "source_kind": "volcano_catalog",
    }
    conflicting_duplicate_hit = {
        **first_duplicate_hit,
        "title": "Duplicate conflicting observation",
        "formats": ["NXS"],
        "license": "CC0-1.0",
        "parser_status": {"kind": "esri_grid_misparse", "valid": False},
        "axis_evidence": {"energy_axis": {"valid": True, "unit": "eV"}},
        "label_evidence": {"verified": False, "ground_truth_roles": ["coordination_motif"]},
        "pairing_evidence": {"verified": True, "roles": ["spectrum"]},
        "access_status": "external_discovery_proposal",
        "source_kind": "external_discovery",
    }
    clean_comparator = {
        **first_duplicate_hit,
        "dataset_code": "clean-comparator",
    }
    gateway = FixtureGateway(
        {
            "XAS": [first_duplicate_hit, clean_comparator],
            "XANES": [conflicting_duplicate_hit],
        }
    )

    candidates = discover_xas(gateway)
    by_code = {candidate.dataset_code: candidate for candidate in candidates}
    duplicate = by_code["duplicate"]

    assert duplicate.formats == ("DAT", "NXS")
    assert duplicate.license is None
    assert duplicate.evidence["source_queries"] == ("XAS", "XANES")
    assert duplicate.evidence["ground_truth_roles"] == (
        "coordination_motif",
        "oxidation_state",
    )
    assert duplicate.evidence["parser_status"]["misparsed"] is True  # type: ignore[index]
    assert duplicate.evidence["axis_evidence"]["energy_axis"]["valid"] is False  # type: ignore[index]
    assert duplicate.evidence["license_observations"] == ("CC-BY-4.0", "CC0-1.0")
    assert [observation["source_query"] for observation in duplicate.evidence["observations"]] == [  # type: ignore[index]
        "XAS",
        "XANES",
    ]
    assert duplicate.evidence["readiness_score"] < by_code["clean-comparator"].evidence["readiness_score"]


def test_unverified_label_roles_cannot_borrow_verification_from_another_hit() -> None:
    """Catches promoting an unverified label claim through a separate empty verification."""
    verified_but_empty = {
        "dataset_code": "split-label-evidence",
        "dataset_version": None,
        "content_digest": None,
        "title": "Split label evidence",
        "description": "",
        "file_count": 1,
        "parsed_file_count": 0,
        "formats": ["DAT"],
        "license": None,
        "parser_status": {"kind": "generic_text", "valid": False},
        "axis_evidence": {"energy_axis": {"valid": False, "unit": None}},
        "label_evidence": {"verified": True, "ground_truth_roles": []},
        "pairing_evidence": {"verified": False, "roles": []},
        "access_status": "admitted_catalog",
        "source_kind": "volcano_catalog",
    }
    claimed_but_unverified = {
        **verified_but_empty,
        "label_evidence": {"verified": False, "ground_truth_roles": ["oxidation_state"]},
    }
    gateway = FixtureGateway(
        {"XAS": [verified_but_empty], "XANES": [claimed_but_unverified]}
    )

    candidate = discover_xas(gateway)[0]

    assert candidate.evidence["ground_truth_roles"] == ("oxidation_state",)
    assert candidate.evidence["label_evidence"]["verified"] is False  # type: ignore[index]
    assert candidate.evidence["label_evidence"]["ground_truth_roles"] == ()  # type: ignore[index]
    assert candidate.evidence["readiness_score"] == 0


def test_unverified_pairing_roles_cannot_borrow_verification_from_another_hit() -> None:
    """Catches promoting an unverified pairing claim through a separate empty verification."""
    verified_but_empty = {
        "dataset_code": "split-pairing-evidence",
        "dataset_version": None,
        "content_digest": None,
        "title": "Split pairing evidence",
        "description": "",
        "file_count": 1,
        "parsed_file_count": 0,
        "formats": ["DAT"],
        "license": None,
        "parser_status": {"kind": "generic_text", "valid": False},
        "axis_evidence": {"energy_axis": {"valid": False, "unit": None}},
        "label_evidence": {"verified": False, "ground_truth_roles": []},
        "pairing_evidence": {"verified": True, "roles": []},
        "access_status": "admitted_catalog",
        "source_kind": "volcano_catalog",
    }
    claimed_but_unverified = {
        **verified_but_empty,
        "pairing_evidence": {"verified": False, "roles": ["structure", "spectrum"]},
    }
    gateway = FixtureGateway(
        {"XAS": [verified_but_empty], "XANES": [claimed_but_unverified]}
    )

    candidate = discover_xas(gateway)[0]

    assert candidate.evidence["pairing_evidence"]["verified"] is False  # type: ignore[index]
    assert candidate.evidence["pairing_evidence"]["roles"] == ()  # type: ignore[index]
    assert candidate.evidence["readiness_score"] == 0


def test_esri_misparsed_asc_never_becomes_a_valid_energy_axis() -> None:
    """Catches treating generic ASC parser output as XAS energy-axis evidence."""
    clean = {
        "dataset_code": "asc-clean",
        "dataset_version": None,
        "content_digest": None,
        "title": "Plain data",
        "description": "",
        "file_count": 1,
        "parsed_file_count": 1,
        "formats": ["ASC"],
        "license": None,
        "parser_status": {"kind": "spectroscopy", "valid": True},
        "axis_evidence": {"energy_axis": {"valid": True, "unit": "eV"}},
        "label_evidence": {"verified": False, "ground_truth_roles": []},
        "pairing_evidence": {"verified": False, "roles": []},
        "access_status": "admitted_catalog",
        "source_kind": "volcano_catalog",
    }
    misparsed = {
        **clean,
        "dataset_code": "asc-esri-misparsed",
        "parser_status": {"kind": "esri_grid_misparse", "valid": False},
        "axis_evidence": {"energy_axis": {"valid": True, "unit": "eV"}},
    }
    gateway = FixtureGateway({query: [clean, misparsed] for query in XAS_QUERIES})

    candidates = discover_xas(gateway)
    by_code = {candidate.dataset_code: candidate for candidate in candidates}

    assert by_code["asc-esri-misparsed"].evidence["axis_evidence"]["energy_axis"]["valid"] is False  # type: ignore[index]
    assert by_code["asc-esri-misparsed"].evidence["readiness_score"] < by_code["asc-clean"].evidence["readiness_score"]
