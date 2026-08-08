"""Evidence-led XAS candidate discovery over HyperData JSON search results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from hyperspectrum.contracts.json import freeze_json_mapping

from .gateway import HydIncompleteSearchError, HydUnsupportedJsonError
from .models import DatasetCandidate, HydCommandResult

XAS_QUERIES = ("XAS", "XANES", "EXAFS", "absorption edge")
XAS_PAGE_LIMIT = 20
XAS_MAX_PAGES = 50
XAS_MAX_RESULTS = 1_000


@dataclass(frozen=True, slots=True)
class QueryCompletion:
    """Proof that one fixed query was read through its final catalog page."""

    query: str
    pages: int
    total: int
    records: int


@dataclass(frozen=True, slots=True)
class XasDiscoveryResult:
    """Ranked candidates plus explicit, machine-readable search completion."""

    candidates: tuple[DatasetCandidate, ...]
    completion: tuple[QueryCompletion, ...]


@dataclass(frozen=True, slots=True)
class _SearchPage:
    records: tuple[Mapping[str, object], ...]
    page: int
    limit: int
    total: int
    has_next: bool
    legacy_complete: bool = False


class SearchGateway(Protocol):
    """The narrow result boundary needed for catalog-only candidate discovery."""

    def search(self, query: str, *, page: int, limit: int) -> HydCommandResult: ...


def discover_xas(
    gateway: SearchGateway,
    *,
    page_limit: int = XAS_PAGE_LIMIT,
    max_pages: int = XAS_MAX_PAGES,
    max_results: int = XAS_MAX_RESULTS,
) -> tuple[DatasetCandidate, ...]:
    """Search fixed XAS terms, merge catalog hits, and rank declared evidence.

    The gateway supplies only parsed JSON from ``hyd search --json``.  This
    function deliberately reads catalog fields and small header evidence only;
    it never opens dataset files or archive members.
    """
    return discover_xas_with_trace(
        gateway,
        page_limit=page_limit,
        max_pages=max_pages,
        max_results=max_results,
    ).candidates


def discover_xas_with_trace(
    gateway: SearchGateway,
    *,
    page_limit: int = XAS_PAGE_LIMIT,
    max_pages: int = XAS_MAX_PAGES,
    max_results: int = XAS_MAX_RESULTS,
) -> XasDiscoveryResult:
    """Read every fixed-query page before exposing any ranked candidates."""

    if type(page_limit) is not int or page_limit < 1:
        raise ValueError("page_limit must be a positive integer")
    if type(max_pages) is not int or max_pages < 1:
        raise ValueError("max_pages must be a positive integer")
    if type(max_results) is not int or max_results < 1:
        raise ValueError("max_results must be a positive integer")

    completed_records: list[tuple[str, tuple[Mapping[str, object], ...]]] = []
    completion: list[QueryCompletion] = []
    for query in XAS_QUERIES:
        records, trace = _read_complete_query(
            gateway,
            query,
            page_limit=page_limit,
            max_pages=max_pages,
            max_results=max_results,
        )
        completed_records.append((query, records))
        completion.append(trace)

    observations: dict[str, list[tuple[str, Mapping[str, object]]]] = {}
    for query, records in completed_records:
        for record in records:
            dataset_code = _optional_string(record.get("dataset_code"))
            if dataset_code is None:
                continue
            observations.setdefault(dataset_code, []).append((query, record))

    candidates = tuple(
        _candidate_from_observations(dataset_code, candidate_observations)
        for dataset_code, candidate_observations in observations.items()
    )
    # Python's stable sort preserves first observed search order for exact
    # evidence ties. Dataset metadata is intentionally not a tie-breaker.
    ranked = tuple(sorted(candidates, key=lambda candidate: -_score(candidate)))
    return XasDiscoveryResult(candidates=ranked, completion=tuple(completion))


def _read_complete_query(
    gateway: SearchGateway,
    query: str,
    *,
    page_limit: int,
    max_pages: int,
    max_results: int,
) -> tuple[tuple[Mapping[str, object], ...], QueryCompletion]:
    records: list[Mapping[str, object]] = []
    expected_total: int | None = None
    for requested_page in range(1, max_pages + 1):
        parsed = _page_from_result(
            gateway.search(query, page=requested_page, limit=page_limit),
            query,
            requested_page=requested_page,
            requested_limit=page_limit,
        )
        if parsed.legacy_complete:
            if requested_page != 1:
                raise HydIncompleteSearchError(
                    "HyperData fixture paging changed before completion"
                )
            if len(parsed.records) > max_results:
                raise HydIncompleteSearchError(
                    "HyperData search exceeds the bounded result cap"
                )
            return parsed.records, QueryCompletion(
                query=query, pages=1, total=len(parsed.records), records=len(parsed.records)
            )
        if expected_total is None:
            expected_total = parsed.total
            if expected_total > max_results:
                raise HydIncompleteSearchError(
                    "HyperData search exceeds the bounded result cap"
                )
        elif parsed.total != expected_total:
            raise HydIncompleteSearchError(
                "HyperData search total changed between pages"
            )
        records.extend(parsed.records)
        if len(records) > max_results or len(records) > parsed.total:
            raise HydIncompleteSearchError(
                "HyperData search returned an inconsistent result count"
            )
        if not parsed.has_next:
            if len(records) != parsed.total:
                raise HydIncompleteSearchError(
                    "HyperData search ended before its declared total"
                )
            return tuple(records), QueryCompletion(
                query=query,
                pages=requested_page,
                total=parsed.total,
                records=len(records),
            )
    raise HydIncompleteSearchError(
        "HyperData search remains incomplete at the bounded page cap"
    )


def _page_from_result(
    result: HydCommandResult,
    requested_query: str,
    *,
    requested_page: int,
    requested_limit: int,
) -> _SearchPage:
    """Read only the pinned hyd wire envelope or exact internal fixture shape."""
    payload = result.payload
    if not isinstance(payload, Mapping):
        raise HydUnsupportedJsonError("HyperData search JSON must be an object")
    keys = set(payload)
    if keys == {"mode", "query", "data", "pagination"}:
        return _page_from_hyd_envelope(
            payload,
            requested_query,
            requested_page=requested_page,
            requested_limit=requested_limit,
        )
    if keys == {"records"}:
        records = payload["records"]
        parsed = _require_object_array(records)
        return _SearchPage(
            records=parsed,
            page=1,
            limit=max(len(parsed), 1),
            total=len(parsed),
            has_next=False,
            legacy_complete=True,
        )
    raise HydUnsupportedJsonError("HyperData search JSON has an unsupported shape")


def _page_from_hyd_envelope(
    payload: Mapping[str, object],
    requested_query: str,
    *,
    requested_page: int,
    requested_limit: int,
) -> _SearchPage:
    """Validate hyperdata-client 84404d53's `search --ilike --json` shape."""

    if payload["mode"] != "ilike" or payload["query"] != requested_query:
        raise HydUnsupportedJsonError("HyperData search mode or query does not match")
    data = payload["data"]
    pagination = payload["pagination"]
    if not isinstance(data, Mapping) or set(data) != {
        "items",
        "total",
        "page",
        "limit",
        "next_cursor",
    }:
        raise HydUnsupportedJsonError("HyperData search data has an unsupported shape")
    if not isinstance(pagination, Mapping) or set(pagination) != {
        "page",
        "limit",
        "total",
        "has_next",
    }:
        raise HydUnsupportedJsonError(
            "HyperData search pagination has an unsupported shape"
        )
    if (
        not _is_nonnegative_int(data["total"])
        or not _is_positive_int(data["page"])
        or not _is_positive_int(data["limit"])
        or not _is_positive_int(pagination["page"])
        or not _is_positive_int(pagination["limit"])
        or not _is_nonnegative_int(pagination["total"])
        or type(pagination["has_next"]) is not bool
        or (
            data["next_cursor"] is not None
            and not isinstance(data["next_cursor"], str)
        )
    ):
        raise HydUnsupportedJsonError(
            "HyperData search pagination fields have invalid types or ranges"
        )
    if (
        data["page"] != pagination["page"]
        or data["limit"] != pagination["limit"]
        or data["page"] != requested_page
        or data["limit"] != requested_limit
        or data["total"] != pagination["total"]
    ):
        raise HydIncompleteSearchError("HyperData search pagination is inconsistent")
    records = _require_object_array(data["items"])
    page = cast(int, data["page"])
    limit = cast(int, data["limit"])
    total = cast(int, data["total"])
    has_next = pagination["has_next"]
    if len(records) > limit or has_next != (page * limit < total):
        raise HydIncompleteSearchError(
            "HyperData search pagination cannot prove completeness"
        )
    return _SearchPage(
        records=records,
        page=page,
        limit=limit,
        total=total,
        has_next=has_next,
    )


def _require_object_array(records: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise HydUnsupportedJsonError(
            "HyperData search result collection must be a JSON array"
        )
    if any(not isinstance(record, Mapping) for record in records):
        raise HydUnsupportedJsonError(
            "HyperData search result collection entries must be JSON objects"
        )
    return tuple(cast(Mapping[str, object], record) for record in records)


def _is_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _is_positive_int(value: object) -> bool:
    return type(value) is int and value >= 1


def _candidate_from_observations(
    dataset_code: str, observations: Sequence[tuple[str, Mapping[str, object]]]
) -> DatasetCandidate:
    """Aggregate duplicate catalog/header observations without suppressing conflicts."""
    observed = tuple(_observation_evidence(query, record) for query, record in observations)
    parser_status = _aggregate_parser_status(observed)
    axis_evidence = _aggregate_axis_evidence(observed)
    label_evidence = _aggregate_role_evidence(observed, "label_evidence", "ground_truth_roles")
    pairing_evidence = _aggregate_role_evidence(observed, "pairing_evidence", "roles")
    roles = _sorted_unique_strings(
        role
        for observation in observed
        for role in cast(tuple[str, ...], observation["ground_truth_roles"])
    )
    readiness_score = _readiness_score(
        parser_status=parser_status,
        axis_evidence=axis_evidence,
        label_evidence=label_evidence,
        pairing_evidence=pairing_evidence,
    )
    licenses = _sorted_unique_strings(
        license_name
        for observation in observed
        if (license_name := _optional_string(observation["license"])) is not None
    )
    license_name = licenses[0] if len(licenses) == 1 else None
    source_queries = _ordered_unique_strings(
        cast(str, observation["source_query"]) for observation in observed
    )
    evidence: dict[str, object] = {
        "source_queries": source_queries,
        "source_kind": _aggregate_state(observed, "source_kind"),
        "source_kinds": _sorted_unique_strings(
            cast(str, observation["source_kind"]) for observation in observed
        ),
        "access_status": _aggregate_state(observed, "access_status"),
        "access_statuses": _sorted_unique_strings(
            cast(str, observation["access_status"]) for observation in observed
        ),
        "parser_status": parser_status,
        "axis_evidence": axis_evidence,
        "label_evidence": label_evidence,
        "pairing_evidence": pairing_evidence,
        "ground_truth_roles": roles,
        "license": license_name,
        "license_observations": licenses,
        "observations": observed,
        "readiness_score": readiness_score,
    }
    return DatasetCandidate(
        dataset_code=dataset_code,
        dataset_version=_uniform_optional_string(observed, "dataset_version"),
        content_digest=_uniform_optional_string(observed, "content_digest"),
        title=_canonical_string(observed, "title"),
        description=_canonical_string(observed, "description"),
        file_count=max(cast(int, observation["file_count"]) for observation in observed),
        parsed_file_count=min(
            cast(int, observation["parsed_file_count"]) for observation in observed
        ),
        formats=_sorted_unique_strings(
            file_format
            for observation in observed
            for file_format in cast(tuple[str, ...], observation["formats"])
        ),
        license=license_name,
        evidence=freeze_json_mapping(evidence),
    )


def _observation_evidence(query: str, record: Mapping[str, object]) -> dict[str, object]:
    """Keep selected public evidence from one result without retaining raw payloads."""
    parser_status = _object(record.get("parser_status"))
    return {
        "source_query": query,
        "source_kind": _optional_string(record.get("source_kind")) or "unknown",
        "access_status": _optional_string(record.get("access_status")) or "unknown",
        "parser_status": parser_status,
        "axis_evidence": _normalise_axis_evidence(
            _object(record.get("axis_evidence")), _optional_string(parser_status.get("kind"))
        ),
        "label_evidence": _object(record.get("label_evidence")),
        "pairing_evidence": _object(record.get("pairing_evidence")),
        "ground_truth_roles": _string_tuple(
            _object(record.get("label_evidence")).get("ground_truth_roles")
        ),
        "license": _optional_string(record.get("license")),
        "formats": _string_tuple(record.get("formats")),
        "dataset_version": _optional_string(record.get("dataset_version")),
        "content_digest": _optional_string(record.get("content_digest")),
        "title": _optional_string(record.get("title")) or "",
        "description": _optional_string(record.get("description")) or "",
        "file_count": _nonnegative_int(record.get("file_count")),
        "parsed_file_count": _nonnegative_int(record.get("parsed_file_count")),
    }


def _aggregate_parser_status(observations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    statuses = tuple(cast(Mapping[str, object], observation["parser_status"]) for observation in observations)
    kinds = _sorted_unique_strings(
        _optional_string(status.get("kind")) or "unknown" for status in statuses
    )
    misparsed = any((_optional_string(status.get("kind")) == "esri_grid_misparse") for status in statuses)
    return {
        "kind": kinds[0] if len(kinds) == 1 else "mixed",
        "valid": bool(statuses) and all(_truth(status.get("valid")) for status in statuses) and not misparsed,
        "misparsed": misparsed,
        "observed_kinds": kinds,
    }


def _aggregate_axis_evidence(observations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    axes = tuple(
        _object(cast(Mapping[str, object], observation["axis_evidence"]).get("energy_axis"))
        for observation in observations
    )
    units = _sorted_unique_strings(
        unit for axis in axes if (unit := _optional_string(axis.get("unit"))) is not None
    )
    return {
        "energy_axis": {
            "valid": bool(axes) and all(_truth(axis.get("valid")) for axis in axes),
            "unit": units[0] if len(units) == 1 else None,
            "observed_units": units,
        }
    }


def _aggregate_role_evidence(
    observations: Sequence[Mapping[str, object]], evidence_key: str, roles_key: str
) -> dict[str, object]:
    evidence = tuple(
        cast(Mapping[str, object], observation[evidence_key]) for observation in observations
    )
    verified_roles = _sorted_unique_strings(
        role
        for item in evidence
        if _truth(item.get("verified"))
        for role in _string_tuple(item.get(roles_key))
    )
    return {
        "verified": bool(verified_roles),
        roles_key: verified_roles,
    }


def _aggregate_state(observations: Sequence[Mapping[str, object]], key: str) -> str:
    values = _sorted_unique_strings(cast(str, observation[key]) for observation in observations)
    return values[0] if len(values) == 1 else "mixed"


def _uniform_optional_string(observations: Sequence[Mapping[str, object]], key: str) -> str | None:
    values = {_optional_string(observation[key]) for observation in observations}
    return next(iter(values)) if len(values) == 1 else None


def _canonical_string(observations: Sequence[Mapping[str, object]], key: str) -> str:
    values = _sorted_unique_strings(cast(str, observation[key]) for observation in observations)
    return values[0] if values else ""


def _score(candidate: DatasetCandidate) -> int:
    """Return the declared evidence score retained with a candidate."""
    return cast(int, candidate.evidence["readiness_score"])


def _readiness_score(
    *,
    parser_status: Mapping[str, object],
    axis_evidence: Mapping[str, object],
    label_evidence: Mapping[str, object],
    pairing_evidence: Mapping[str, object],
) -> int:
    """Score verifiable label, pairing, axis, and parser evidence only."""
    score = 0
    if _truth(label_evidence.get("verified")) and _string_tuple(
        label_evidence.get("ground_truth_roles")
    ):
        score += 40
    if _truth(pairing_evidence.get("verified")) and _string_tuple(pairing_evidence.get("roles")):
        score += 30

    energy_axis = _object(axis_evidence.get("energy_axis"))
    if _truth(energy_axis.get("valid")):
        score += 20

    parser_kind = _optional_string(parser_status.get("kind")) or ""
    if _truth(parser_status.get("misparsed")) or parser_kind == "esri_grid_misparse":
        score -= 20
    elif _truth(parser_status.get("valid")) and parser_kind == "spectroscopy":
        score += 10
    return score


def _normalise_axis_evidence(
    axis_evidence: Mapping[str, object], parser_kind: str | None
) -> dict[str, object]:
    """Invalidate claimed XAS axes from a known non-spectroscopy misparse."""
    normalised = dict(axis_evidence)
    energy_axis = _object(axis_evidence.get("energy_axis"))
    unit = _optional_string(energy_axis.get("unit"))
    valid = _truth(energy_axis.get("valid")) and unit is not None
    if parser_kind == "esri_grid_misparse":
        valid = False
    normalised["energy_axis"] = {"valid": valid, "unit": unit}
    return normalised


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _required_string(value: object, field_name: str) -> str:
    result = _optional_string(value)
    if result is None:
        raise ValueError(f"{field_name} must be a non-blank string")
    return result


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _ordered_unique_strings(values: Iterable[object]) -> tuple[str, ...]:
    """Return the first observed spelling of each non-blank string."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalised = _optional_string(value)
        if normalised is not None and normalised not in seen:
            seen.add(normalised)
            ordered.append(normalised)
    return tuple(ordered)


def _sorted_unique_strings(values: Iterable[object]) -> tuple[str, ...]:
    """Return canonical evidence values independently of discovery query order."""
    return tuple(sorted(set(_ordered_unique_strings(values))))


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _truth(value: object) -> bool:
    return value is True
