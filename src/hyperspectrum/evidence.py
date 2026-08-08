"""Packaged validation for successful XAS M0 evidence egress."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any, cast

from jsonschema import Draft202012Validator

from hyperspectrum.process_boundary import redact_value

_URL_LOCATOR = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s<>{}\[\]\"']+")


class XasM0EvidenceError(ValueError):
    """One stable, non-leaking evidence refusal."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_xas_m0_evidence(value: object) -> dict[str, Any]:
    """Validate schema and structural redaction before evidence can leave a process."""

    try:
        detached = json.loads(
            json.dumps(value, ensure_ascii=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise XasM0EvidenceError(
            "schema_invalid", "XAS M0 evidence is not valid JSON"
        ) from error
    if not isinstance(detached, dict):
        raise XasM0EvidenceError(
            "schema_invalid", "XAS M0 evidence must be a JSON object"
        )

    schema = _load_schema()
    validator = Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    )
    if next(validator.iter_errors(detached), None) is not None:
        raise XasM0EvidenceError(
            "schema_invalid", "XAS M0 evidence violates the success schema"
        )
    if redact_value(detached) != detached:
        raise XasM0EvidenceError(
            "sensitive_or_private_value",
            "XAS M0 evidence contains a sensitive or private value",
        )
    if _contains_url_locator(detached):
        raise XasM0EvidenceError(
            "locator_forbidden", "XAS M0 evidence contains a URL locator"
        )
    _validate_provenance_chain(detached)
    _validate_count_chain(detached)
    return cast(dict[str, Any], detached)


def _load_schema() -> dict[str, Any]:
    resource = files("hyperspectrum.resources").joinpath(
        "schemas/xas-m0-selection.schema.json"
    )
    loaded = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):  # pragma: no cover - packaged invariant
        raise TypeError("packaged XAS M0 evidence schema must be an object")
    Draft202012Validator.check_schema(loaded)
    return cast(dict[str, Any], loaded)


def _contains_url_locator(value: object) -> bool:
    if isinstance(value, str):
        return _URL_LOCATOR.search(value) is not None
    if isinstance(value, Mapping):
        return any(
            _contains_url_locator(str(key)) or _contains_url_locator(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return any(_contains_url_locator(item) for item in value)
    return False


def _validate_provenance_chain(evidence: dict[str, Any]) -> None:
    dataset = cast(dict[str, Any], evidence["dataset"])
    smoke = cast(dict[str, Any], evidence["smoke_run"])
    provenance = cast(dict[str, Any], smoke["provenance"])
    handoff = cast(dict[str, Any], evidence["ace_handoff"])
    assets = cast(list[dict[str, Any]], evidence["assets"])
    matching_assets = [
        asset for asset in assets if asset["id"] == handoff["data_asset_id"]
    ]
    if len(matching_assets) != 1:
        raise XasM0EvidenceError(
            "provenance_mismatch", "XAS M0 evidence asset identity is inconsistent"
        )
    asset = matching_assets[0]
    byte_provenance = cast(dict[str, Any], asset["byte_provenance"])
    if len(
        {
            dataset["content_digest"],
            byte_provenance["source_dataset_digest"],
            provenance["dataset_digest"],
        }
    ) != 1 or len(
        {
            asset["sha256"],
            provenance["data_digest"],
            handoff["data_asset_sha256"],
        }
    ) != 1:
        raise XasM0EvidenceError(
            "provenance_mismatch", "XAS M0 evidence provenance is inconsistent"
        )


def _validate_count_chain(evidence: dict[str, Any]) -> None:
    dataset = cast(dict[str, Any], evidence["dataset"])
    counts = cast(dict[str, int], dataset["counts"])
    full_split = cast(dict[str, Any], dataset["full_split"])
    selection = cast(dict[str, Any], evidence["selection"])
    smoke = cast(dict[str, Any], evidence["smoke_run"])
    selected = cast(int, selection["selected_sample_count"])
    input_count = cast(int, smoke["input_sample_count"])
    prediction_count = cast(int, smoke["prediction_count"])
    if not (
        counts["sample_count"] >= full_split["sample_count"] >= selected
        and selected == input_count == prediction_count
        and selection["max_samples"] == smoke["max_samples"]
        and selected <= selection["max_samples"]
        and counts["file_count"]
        >= counts["parsed_file_count"]
        >= full_split["file_count"]
    ):
        raise XasM0EvidenceError(
            "count_mismatch", "XAS M0 evidence counts are inconsistent"
        )
