"""Contract tests for the successful real XAS M0 evidence handoff."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

SCHEMA_PATH = (
    Path(__file__).parents[2] / "docs" / "evidence" / "xas-m0-selection.schema.json"
)


@pytest.fixture
def valid_evidence() -> dict[str, Any]:
    """Return a hand-checked successful handoff with no real remote identifiers."""

    return {
        "schema_version": "hyperspectrum-xas-m0-selection/v1",
        "query": {
            "timestamp": "2026-08-08T14:30:00Z",
            "profile": "volcano",
            "client_version": "0.11.2",
            "source_queries": ["XAS", "XANES", "EXAFS", "absorption edge"],
        },
        "runtime": {
            "host_class": "5090",
            "hyperspectrum_commit": "1" * 40,
            "wheel_sha256": "2" * 64,
            "hyperdata_client_commit": "3" * 40,
        },
        "dataset": {
            "code": "public-xas-pairs",
            "version": "2026.08.1",
            "content_digest": "4" * 64,
            "title": "Public paired XAS spectra",
            "description": "Documented repeated acquisitions with an average reference",
            "license_state": "declared",
            "license": "CC-BY-4.0",
            "access_state": "authorized",
            "source": {
                "kind": "hyperdata_catalog",
                "profile": "volcano",
                "evidence": "live_json_search",
            },
            "counts": {
                "file_count": 12,
                "parsed_file_count": 12,
                "sample_count": 8,
                "split_count": 3,
                "source": "catalog_metadata",
            },
            "full_split": {
                "id": "full",
                "file_count": 12,
                "sample_count": 8,
                "source": {
                    "kind": "catalog_split_contract",
                    "reference": "full",
                },
            },
            "formats": ["npz"],
        },
        "task": {
            "verdict": "scoreable",
            "task_id": "xas-denoising",
            "task_type": "denoising",
            "ground_truth_roles": ["clean_spectrum"],
            "split_group_keys": ["compound_id"],
            "reason_codes": ["documented_noisy_clean_pairing"],
        },
        "assets": [
            {
                "id": "xas-inference-input",
                "role": "data",
                "file_name": "xas-inference-input.npz",
                "size_bytes": 4096,
                "sha256": "5" * 64,
                "materialization": "mount-file",
                "byte_provenance": {
                    "identity_contract": "catalog_exact_bytes",
                    "source_dataset_digest": "4" * 64,
                    "source_asset_id": "catalog-asset-1",
                    "source_availability": "authorized",
                },
                "inner_path": None,
                "semantics": {
                    "modality": "xas",
                    "role": "inference_input",
                    "npz_keys": [
                        "energy",
                        "noisy",
                        "sample_ids",
                        "group_ids",
                        "energy_unit",
                    ],
                    "energy_unit": "eV",
                    "target_data_excluded": True,
                },
            }
        ],
        "selection": {
            "algorithm": "catalog_order_first_n",
            "version": "1",
            "ordered_sample_ids_sha256": "8" * 64,
            "selected_sample_count": 8,
            "max_samples": 8,
        },
        "smoke_run": {
            "status": "completed",
            "max_samples": 8,
            "input_sample_count": 8,
            "prediction_count": 8,
            "failure_count": 0,
            "provenance": {
                "plan_digest": "9" * 64,
                "model_digest": "a" * 64,
                "tool_digest": "b" * 64,
                "implementation_digest": "c" * 64,
                "weight_digest": "none",
                "data_digest": "5" * 64,
                "environment_digest": "d" * 64,
                "dataset_digest": "4" * 64,
            },
            "artifacts": {
                "prediction_bundle_sha256": "6" * 64,
                "run_manifest_sha256": "e" * 64,
                "artifact_names": ["predictions.json", "run.json"],
            },
        },
        "ace_handoff": {
            "dataset_id": "public-xas-pairs-2026-08-1",
            "data_asset_sha256": "5" * 64,
            "task_spec_sha256": "7" * 64,
            "primary_metric": {
                "key": "normalized_spectrum_rmse",
                "direction": "min",
                "aggregation": "grouped_bootstrap",
            },
            "runtime_owner": "HyperSpectrum",
            "formal_evaluator": "ACE Benchmark",
        },
        "blockers": [],
    }


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)


def test_schema_accepts_complete_scoreable_real_handoff(
    valid_evidence: dict[str, Any],
) -> None:
    """A missing required ACE field must make a complete handoff fail."""

    _validator().validate(valid_evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", None),
        ("content_digest", None),
        ("content_digest", "0" * 64),
    ],
)
def test_schema_rejects_unpinned_dataset_identity(
    valid_evidence: dict[str, Any], field: str, value: object
) -> None:
    """Permitting null or placeholder identity would admit unverifiable data."""

    invalid = copy.deepcopy(valid_evidence)
    invalid["dataset"][field] = value

    with pytest.raises(ValidationError):
        _validator().validate(invalid)


def test_schema_accepts_a_large_full_split_with_an_eight_sample_smoke(
    valid_evidence: dict[str, Any],
) -> None:
    """A bounded smoke selection must not be confused with total split size."""

    valid_evidence["dataset"]["counts"]["sample_count"] = 100
    valid_evidence["dataset"]["full_split"]["sample_count"] = 100

    _validator().validate(valid_evidence)


@pytest.mark.parametrize(
    "field", ["id", "file_count", "sample_count", "source"]
)
def test_schema_requires_full_benchmark_split_counts_and_source(
    valid_evidence: dict[str, Any], field: str
) -> None:
    """Dropping full-split identity would force ACE to infer benchmark scope."""

    invalid = copy.deepcopy(valid_evidence)
    del invalid["dataset"]["full_split"][field]

    with pytest.raises(ValidationError):
        _validator().validate(invalid)


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "Bearer abcdefghijklmnopqrstuvwxyz",
        "https://assets.example.test/input.npz?X-Amz-Signature=secret",
        "/home/operator/private/input.npz",
        "/data/project/raw/input.npz",
        "/Users/operator/runs/input.npz",
    ],
)
def test_schema_rejects_secret_signed_url_and_private_path_values(
    valid_evidence: dict[str, Any], unsafe_value: str
) -> None:
    """Relaxing safe text would leak private transport details into Git evidence."""

    invalid = copy.deepcopy(valid_evidence)
    invalid["dataset"]["description"] = unsafe_value

    with pytest.raises(ValidationError):
        _validator().validate(invalid)


def test_schema_rejects_secret_bearing_extra_fields(
    valid_evidence: dict[str, Any],
) -> None:
    """Allowing unknown keys would provide an unchecked channel for credentials."""

    invalid = copy.deepcopy(valid_evidence)
    invalid["query"]["access_token"] = "redacted-but-not-allowed"

    with pytest.raises(ValidationError):
        _validator().validate(invalid)


@pytest.mark.parametrize(
    "field",
    [
        "id",
        "role",
        "file_name",
        "size_bytes",
        "sha256",
        "materialization",
        "byte_provenance",
        "inner_path",
        "semantics",
    ],
)
def test_schema_requires_complete_asset_contract(
    valid_evidence: dict[str, Any], field: str
) -> None:
    """Dropping an asset identity or semantics field would make M1 infer storage."""

    invalid = copy.deepcopy(valid_evidence)
    del invalid["assets"][0][field]

    with pytest.raises(ValidationError):
        _validator().validate(invalid)


@pytest.mark.parametrize(
    ("materialization", "inner_path"),
    [
        ("unpack", None),
        ("mount-file", "archive/input.npz"),
        ("mount-dir", "archive/input.npz"),
        ("catalog_exact_bytes", None),
    ],
)
def test_schema_keeps_npz_materialization_directly_mappable_to_ace(
    valid_evidence: dict[str, Any], materialization: str, inner_path: str | None
) -> None:
    """Inferring unpack behavior from a suffix would break ACE asset staging."""

    invalid = copy.deepcopy(valid_evidence)
    invalid["assets"][0]["materialization"] = materialization
    invalid["assets"][0]["inner_path"] = inner_path

    with pytest.raises(ValidationError):
        _validator().validate(invalid)


@pytest.mark.parametrize(
    "field",
    ["algorithm", "version", "ordered_sample_ids_sha256", "selected_sample_count"],
)
def test_schema_requires_deterministic_selection_identity(
    valid_evidence: dict[str, Any], field: str
) -> None:
    """A max-samples cap alone must not identify which samples were executed."""

    invalid = copy.deepcopy(valid_evidence)
    del invalid["selection"][field]

    with pytest.raises(ValidationError):
        _validator().validate(invalid)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("selection", "selected_sample_count"), 7),
        (("selection", "max_samples"), 7),
        (("smoke_run", "max_samples"), 7),
        (("dataset", "counts", "sample_count"), 7),
        (("dataset", "full_split", "sample_count"), 7),
    ],
)
def test_schema_rejects_inconsistent_selection_and_observed_counts(
    valid_evidence: dict[str, Any], path: tuple[str, ...], value: int
) -> None:
    """Selection, dataset, and smoke counts must describe the same admitted run."""

    invalid = copy.deepcopy(valid_evidence)
    target: dict[str, Any] = invalid
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        _validator().validate(invalid)


@pytest.mark.parametrize(
    "field",
    [
        "plan_digest",
        "model_digest",
        "tool_digest",
        "implementation_digest",
        "weight_digest",
        "data_digest",
        "environment_digest",
        "dataset_digest",
    ],
)
def test_schema_requires_complete_smoke_runtime_provenance(
    valid_evidence: dict[str, Any], field: str
) -> None:
    """Missing execution digests would make the prediction handoff mutable."""

    invalid = copy.deepcopy(valid_evidence)
    del invalid["smoke_run"]["provenance"][field]

    with pytest.raises(ValidationError):
        _validator().validate(invalid)


@pytest.mark.parametrize("field", ["adapter", "image", "backendHandle"])
def test_schema_rejects_fabricated_m1_runtime_fields(
    valid_evidence: dict[str, Any], field: str
) -> None:
    """M0 must not fabricate adapter, image, or backend fields owned by ACE M1."""

    invalid = copy.deepcopy(valid_evidence)
    invalid["smoke_run"][field] = "not-owned-by-m0"

    with pytest.raises(ValidationError):
        _validator().validate(invalid)


@pytest.mark.parametrize("verdict", ["inference_only", "blocked"])
def test_schema_rejects_non_scoreable_selection(
    valid_evidence: dict[str, Any], verdict: str
) -> None:
    """Changing the success gate to admit a non-scoreable verdict must fail."""

    invalid = copy.deepcopy(valid_evidence)
    invalid["task"]["verdict"] = verdict

    with pytest.raises(ValidationError):
        _validator().validate(invalid)


def test_schema_rejects_partial_prediction_coverage(
    valid_evidence: dict[str, Any],
) -> None:
    """A dropped materialized sample must invalidate the smoke-run evidence."""

    invalid = copy.deepcopy(valid_evidence)
    invalid["smoke_run"]["prediction_count"] = 7

    with pytest.raises(ValidationError):
        _validator().validate(invalid)


def test_schema_rejects_nonempty_blockers(valid_evidence: dict[str, Any]) -> None:
    """A successful selection cannot coexist with an unresolved gate blocker."""

    invalid = copy.deepcopy(valid_evidence)
    invalid["blockers"] = ["catalog_unreachable"]

    with pytest.raises(ValidationError):
        _validator().validate(invalid)
