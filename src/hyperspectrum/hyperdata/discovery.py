"""Evidence-led XAS candidate discovery over HyperData JSON search results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from hyperspectrum.contracts.json import freeze_json_mapping

from .models import DatasetCandidate, HydCommandResult

XAS_QUERIES = ("XAS", "XANES", "EXAFS", "absorption edge")


class SearchGateway(Protocol):
    """The narrow result boundary needed for catalog-only candidate discovery."""

    def search(self, query: str) -> HydCommandResult: ...


def discover_xas(gateway: SearchGateway) -> tuple[DatasetCandidate, ...]:
    """Search fixed XAS terms, merge catalog hits, and rank declared evidence.

    The gateway supplies only parsed JSON from ``hyd search --json``.  This
    function deliberately reads catalog fields and small header evidence only;
    it never opens dataset files or archive members.
    """
    records: dict[str, Mapping[str, object]] = {}
    source_queries: dict[str, list[str]] = {}
    for query in XAS_QUERIES:
        for record in _records_from_result(gateway.search(query)):
            dataset_code = _optional_string(record.get("dataset_code"))
            if dataset_code is None:
                continue
            records.setdefault(dataset_code, record)
            source_queries.setdefault(dataset_code, []).append(query)

    candidates = tuple(
        _candidate_from_record(record, tuple(source_queries[dataset_code]))
        for dataset_code, record in records.items()
    )
    return tuple(sorted(candidates, key=lambda candidate: (-_score(candidate), candidate.dataset_code)))


def _records_from_result(result: HydCommandResult) -> tuple[Mapping[str, object], ...]:
    """Read the documented JSON payload envelope without table parsing."""
    payload = result.payload
    if not isinstance(payload, Mapping):
        return ()
    records = payload.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        return ()
    return tuple(cast(Mapping[str, object], record) for record in records if isinstance(record, Mapping))


def _candidate_from_record(
    record: Mapping[str, object], source_queries: tuple[str, ...]
) -> DatasetCandidate:
    """Select safe catalog/header fields and retain their provenance as evidence."""
    parser_status = _object(record.get("parser_status"))
    axis_evidence = _normalise_axis_evidence(
        _object(record.get("axis_evidence")), _optional_string(parser_status.get("kind"))
    )
    label_evidence = _object(record.get("label_evidence"))
    pairing_evidence = _object(record.get("pairing_evidence"))
    roles = _string_tuple(label_evidence.get("ground_truth_roles"))
    readiness_score = _readiness_score(
        parser_status=parser_status,
        axis_evidence=axis_evidence,
        label_evidence=label_evidence,
        pairing_evidence=pairing_evidence,
    )
    license_name = _optional_string(record.get("license"))
    evidence: dict[str, object] = {
        "source_queries": source_queries,
        "source_kind": _optional_string(record.get("source_kind")) or "unknown",
        "access_status": _optional_string(record.get("access_status")) or "unknown",
        "parser_status": parser_status,
        "axis_evidence": axis_evidence,
        "label_evidence": label_evidence,
        "pairing_evidence": pairing_evidence,
        "ground_truth_roles": roles,
        "license": license_name,
        "readiness_score": readiness_score,
    }
    return DatasetCandidate(
        dataset_code=_required_string(record.get("dataset_code"), "dataset_code"),
        dataset_version=_optional_string(record.get("dataset_version")),
        content_digest=_optional_string(record.get("content_digest")),
        title=_optional_string(record.get("title")) or "",
        description=_optional_string(record.get("description")) or "",
        file_count=_nonnegative_int(record.get("file_count")),
        parsed_file_count=_nonnegative_int(record.get("parsed_file_count")),
        formats=_string_tuple(record.get("formats")),
        license=license_name,
        evidence=freeze_json_mapping(evidence),
    )


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
    if parser_kind == "esri_grid_misparse":
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


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _truth(value: object) -> bool:
    return value is True
