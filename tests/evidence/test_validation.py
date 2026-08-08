"""Semantic egress validation for a successful XAS M0 handoff."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_xas_m0_evidence import make_valid_evidence

from hyperspectrum.evidence import XasM0EvidenceError, validate_xas_m0_evidence

ROOT = Path(__file__).resolve().parents[2]


def test_semantic_validator_accepts_one_consistent_success_handoff() -> None:
    evidence = make_valid_evidence()

    validated = validate_xas_m0_evidence(evidence)

    assert validated == evidence
    assert validated is not evidence


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "PASSWORD=UPPER_ASSIGNMENT_LITERAL",
        "clientSecret: CAMEL_ASSIGNMENT_LITERAL",
        '{"ClientSecret":"QUOTED_NESTED_LITERAL"}',
        "https://safe-user:USERINFO_LITERAL@example.invalid/data",
        "endpoint 10.0.0.8:8443",
        "endpoint 169.254.10.2",
        "endpoint [::1]:8080",
        "endpoint database.internal:5432",
        "endpoint localhost:8000",
        "/srv/private/catalog/input.npz",
        r"C:\Users\operator\private\input.npz",
        r"\\fileserver\private\input.npz",
    ],
)
def test_semantic_validator_rejects_any_value_changed_by_shared_redaction(
    unsafe_value: str,
) -> None:
    evidence = make_valid_evidence()
    evidence["dataset"]["description"] = unsafe_value

    with pytest.raises(XasM0EvidenceError) as captured:
        validate_xas_m0_evidence(evidence)

    assert captured.value.code in {"schema_invalid", "sensitive_or_private_value"}
    assert unsafe_value not in str(captured.value)


def test_public_url_locator_reaches_the_structural_redaction_gate() -> None:
    evidence = make_valid_evidence()
    evidence["dataset"]["description"] = "https://public.example.invalid/item"

    with pytest.raises(XasM0EvidenceError) as captured:
        validate_xas_m0_evidence(evidence)

    assert captured.value.code == "locator_forbidden"


@pytest.mark.parametrize(
    "private_locator",
    (
        "compute-node:8080",
        "database.corp:5432",
        "8.8.8.8:53",
        "1.1.1.1:443",
        "fd00::1",
        "fc00::1234",
        "fe80::1",
        "fe80::1%en0",
        "::1",
        "::ffff:8.8.8.8",
        "2001:4860:4860::8888",
        "[2001:4860:4860::8888]:443",
        "[fe80::1%en0]:8080",
    ),
)
def test_semantic_validator_rejects_endpoint_locators_without_echoing_input(
    private_locator: str,
) -> None:
    evidence = make_valid_evidence()
    evidence["dataset"]["description"] = private_locator

    with pytest.raises(XasM0EvidenceError) as captured:
        validate_xas_m0_evidence(evidence)

    assert captured.value.code == "locator_forbidden"
    assert private_locator not in str(captured.value)


def test_semantic_validator_checks_schema_before_redaction() -> None:
    evidence = make_valid_evidence()
    evidence["unexpected"] = "PASSWORD=SCHEMA_FIRST_LITERAL"

    with pytest.raises(XasM0EvidenceError) as captured:
        validate_xas_m0_evidence(evidence)

    assert captured.value.code == "schema_invalid"
    assert "SCHEMA_FIRST_LITERAL" not in str(captured.value)


def test_semantic_validator_detaches_the_validated_payload() -> None:
    evidence = make_valid_evidence()
    validated = validate_xas_m0_evidence(evidence)

    evidence["dataset"]["title"] = "caller mutation"

    assert validated["dataset"]["title"] == "Public paired XAS spectra"


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("assets", 0, "byte_provenance", "source_dataset_digest"), "f" * 64),
        (("smoke_run", "provenance", "dataset_digest"), "f" * 64),
        (("smoke_run", "provenance", "data_digest"), "f" * 64),
        (("ace_handoff", "data_asset_sha256"), "f" * 64),
        (("ace_handoff", "data_asset_id"), "other-valid-asset-id"),
    ],
)
def test_semantic_validator_rejects_different_but_valid_provenance_identities(
    path: tuple[str | int, ...], replacement: str
) -> None:
    evidence = make_valid_evidence()
    target: object = evidence
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(XasM0EvidenceError) as captured:
        validate_xas_m0_evidence(evidence)

    assert captured.value.code == "provenance_mismatch"


def test_semantic_validator_accepts_dataset_100_full_split_80_smoke_8() -> None:
    evidence = make_valid_evidence()
    evidence["dataset"]["counts"].update(
        {"file_count": 120, "parsed_file_count": 100, "sample_count": 100}
    )
    evidence["dataset"]["full_split"].update(
        {"file_count": 100, "sample_count": 80}
    )

    validate_xas_m0_evidence(evidence)


@pytest.mark.parametrize(
    "mutation",
    [
        {"dataset_sample_count": 79, "full_sample_count": 80},
        {"dataset_file_count": 79, "full_file_count": 80},
        {"dataset_file_count": 80, "parsed_file_count": 81},
    ],
)
def test_semantic_validator_rejects_individually_valid_but_inconsistent_counts(
    mutation: dict[str, int],
) -> None:
    evidence = make_valid_evidence()
    counts = evidence["dataset"]["counts"]
    full_split = evidence["dataset"]["full_split"]
    counts["sample_count"] = mutation.get("dataset_sample_count", 100)
    full_split["sample_count"] = mutation.get("full_sample_count", 80)
    counts["file_count"] = mutation.get("dataset_file_count", 100)
    counts["parsed_file_count"] = mutation.get("parsed_file_count", 80)
    full_split["file_count"] = mutation.get("full_file_count", 80)

    with pytest.raises(XasM0EvidenceError) as captured:
        validate_xas_m0_evidence(evidence)

    assert captured.value.code == "count_mismatch"


def test_agent_service_validates_success_evidence_file(tmp_path: Path) -> None:
    from hyperspectrum import agent

    evidence_file = tmp_path / "selection.json"
    evidence_file.write_text(json.dumps(make_valid_evidence()), encoding="utf-8")

    response = agent.validate_evidence(evidence_file)

    assert response.result == {
        "schema_version": "hyperspectrum-xas-m0-selection/v1",
        "status": "valid",
    }


def test_real_cli_refuses_sensitive_evidence_in_one_stable_envelope(
    tmp_path: Path,
) -> None:
    evidence = make_valid_evidence()
    evidence["dataset"]["description"] = (
        "https://public.example.invalid/CLI_PRIVATE_LITERAL"
    )
    evidence_file = tmp_path / "selection.json"
    evidence_file.write_text(json.dumps(evidence), encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "hyperspectrum.cli",
            "evidence",
            "validate",
            "--evidence-file",
            str(evidence_file),
            "--json",
        ),
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 2
    assert completed.stdout.count("\n") == 1
    envelope = json.loads(completed.stdout)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "invalid_xas_m0_evidence"
    assert envelope["result"] == {"reason_code": "locator_forbidden"}
    assert "CLI_PRIVATE_LITERAL" not in completed.stdout + completed.stderr


@pytest.mark.parametrize(
    ("reason_code", "mutation"),
    [
        (
            "sensitive_or_private_value",
            {"description": "endpoint 10.0.0.8:8443 PRIVATE_REASON_LITERAL"},
        ),
        (
            "provenance_mismatch",
            {"data_asset_sha256": "f" * 64, "marker": "PROVENANCE_REASON_LITERAL"},
        ),
        (
            "count_mismatch",
            {
                "dataset_sample_count": 79,
                "full_sample_count": 80,
                "marker": "COUNT_REASON_LITERAL",
            },
        ),
    ],
)
def test_real_cli_reports_safe_semantic_reason_codes(
    tmp_path: Path, reason_code: str, mutation: dict[str, object]
) -> None:
    evidence = make_valid_evidence()
    marker = str(mutation.get("marker", "PRIVATE_REASON_LITERAL"))
    evidence["dataset"]["description"] = str(
        mutation.get("description", marker)
    )
    if "data_asset_sha256" in mutation:
        evidence["ace_handoff"]["data_asset_sha256"] = mutation[
            "data_asset_sha256"
        ]
    if "dataset_sample_count" in mutation:
        evidence["dataset"]["counts"]["sample_count"] = mutation[
            "dataset_sample_count"
        ]
        evidence["dataset"]["full_split"]["sample_count"] = mutation[
            "full_sample_count"
        ]
    evidence_file = tmp_path / f"{reason_code}.json"
    evidence_file.write_text(json.dumps(evidence), encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "hyperspectrum.cli",
            "evidence",
            "validate",
            "--evidence-file",
            str(evidence_file),
            "--json",
        ),
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 2
    envelope = json.loads(completed.stdout)
    assert envelope["error"]["code"] == "invalid_xas_m0_evidence"
    assert envelope["result"] == {"reason_code": reason_code}
    assert marker not in completed.stdout + completed.stderr
