from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from hyperspectrum.contracts import (
    ArtifactRef,
    AxisSpec,
    MetricSpec,
    ObservationBundle,
    PredictionBundle,
    TaskSpec,
)


def axis() -> AxisSpec:
    return AxisSpec(name="energy", unit="eV", direction="increasing")


def artifact(*, role: str = "raw_signal") -> ArtifactRef:
    return ArtifactRef(
        role=role,
        kind="dense_array",
        uri="hyperdata://dataset/v1/spectrum",
        sha256="a" * 64,
        axes=(axis(),),
    )


def test_public_contracts_construct_with_explicit_metadata() -> None:
    raw_signal = artifact()
    observation = ObservationBundle(
        schema_version="hyperspectrum-observation/v1",
        sample_id="sample-001",
        modality="xas",
        artifacts=(raw_signal,),
        context={"instrument": {"beamline": "BL-1"}},
        labels={"oxidation_state": 2},
        provenance={"data_digest": "b" * 64},
    )
    metric = MetricSpec(
        key="normalized_spectrum_rmse",
        direction="min",
        aggregation="sample_mean",
        primary=True,
    )
    task = TaskSpec(
        schema_version="hyperspectrum-task/v1",
        id="xas-denoising",
        modality="xas",
        task_type="denoising",
        input_roles=("raw_signal",),
        output_kind="dense_array",
        ground_truth_roles=("ground_truth",),
        split_group_keys=("compound_id",),
        metrics=(metric,),
    )
    prediction = PredictionBundle(
        schema_version="hyperspectrum-prediction/v1",
        run_id="run-001",
        task_id=task.id,
        predictions=(artifact(role="prediction"),),
        failures=(),
        provenance={
            "model_digest": "c" * 64,
            "tool_digest": "d" * 64,
            "data_digest": "e" * 64,
            "environment_digest": "f" * 64,
        },
    )

    assert observation.artifacts == (raw_signal,)
    assert task.metrics == (metric,)
    assert prediction.predictions[0].role == "prediction"


def test_dense_array_requires_explicit_axes() -> None:
    with pytest.raises(ValidationError, match="at least one axis"):
        ArtifactRef(
            role="raw_signal",
            kind="dense_array",
            uri="hyperdata://dataset/v1/spectrum",
        )


def test_axis_unit_cannot_be_blank() -> None:
    with pytest.raises(ValidationError, match="unit"):
        AxisSpec(name="energy", unit="   ", direction="increasing")


def test_artifact_sha256_must_be_lowercase_hex() -> None:
    with pytest.raises(ValidationError, match="sha256"):
        ArtifactRef(
            role="metadata",
            kind="metadata",
            uri="hyperdata://dataset/v1/metadata",
            sha256="A" * 64,
        )


def test_dense_array_axis_names_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="unique"):
        ArtifactRef(
            role="raw_signal",
            kind="dense_array",
            uri="hyperdata://dataset/v1/spectrum",
            axes=(axis(), axis()),
        )


def test_prediction_requires_all_nonempty_provenance_digests() -> None:
    with pytest.raises(ValidationError, match="environment_digest"):
        PredictionBundle(
            schema_version="hyperspectrum-prediction/v1",
            run_id="run-001",
            task_id="xas-denoising",
            predictions=(artifact(role="prediction"),),
            failures=(),
            provenance={
                "model_digest": "c" * 64,
                "tool_digest": "d" * 64,
                "data_digest": "e" * 64,
            },
        )


def test_contract_instances_are_frozen() -> None:
    instance = axis()

    with pytest.raises(ValidationError):
        instance.unit = "keV"  # type: ignore[misc]


def test_observation_json_metadata_is_recursively_immutable_and_detached() -> None:
    context = {"instrument": {"scan_modes": ["transmission"]}}
    labels = {"assignments": ["oxide"]}
    provenance = {"sources": {"files": ["scan-001"]}}
    observation = ObservationBundle(
        schema_version="hyperspectrum-observation/v1",
        sample_id="sample-001",
        modality="xas",
        artifacts=(artifact(),),
        context=context,
        labels=labels,
        provenance=provenance,
    )

    context["instrument"]["scan_modes"].append("fluorescence")
    labels["assignments"].append("metal")
    provenance["sources"]["files"].append("scan-002")

    assert observation.model_dump()["context"] == {
        "instrument": {"scan_modes": ["transmission"]}
    }
    assert observation.model_dump()["labels"] == {"assignments": ["oxide"]}
    assert observation.model_dump()["provenance"] == {
        "sources": {"files": ["scan-001"]}
    }
    with pytest.raises(TypeError):
        observation.context["new_context"] = "forbidden"  # type: ignore[index]
    with pytest.raises(TypeError):
        observation.context["instrument"]["beamline"] = "BL-1"  # type: ignore[index]
    with pytest.raises(AttributeError):
        observation.labels["assignments"].append("forbidden")  # type: ignore[union-attr]


def test_prediction_json_metadata_and_failures_are_recursively_immutable() -> None:
    provenance = {
        "model_digest": "c" * 64,
        "tool_digest": "d" * 64,
        "data_digest": "e" * 64,
        "environment_digest": "f" * 64,
        "inputs": {"shards": ["shard-001"]},
    }
    failures = [{"sample_id": "sample-002", "details": {"reasons": ["out_of_range"]}}]
    prediction = PredictionBundle(
        schema_version="hyperspectrum-prediction/v1",
        run_id="run-001",
        task_id="xas-denoising",
        predictions=(artifact(role="prediction"),),
        failures=failures,
        provenance=provenance,
    )

    provenance["inputs"]["shards"].append("shard-002")
    failures[0]["details"]["reasons"].append("missing_axis")

    assert prediction.model_dump()["provenance"]["inputs"] == {"shards": ["shard-001"]}
    assert prediction.model_dump()["failures"] == [
        {"sample_id": "sample-002", "details": {"reasons": ["out_of_range"]}}
    ]
    with pytest.raises(TypeError):
        prediction.provenance["new_digest"] = "forbidden"  # type: ignore[index]
    with pytest.raises(TypeError):
        prediction.failures[0]["details"]["error"] = "forbidden"  # type: ignore[index]
    with pytest.raises(AttributeError):
        prediction.failures[0]["details"]["reasons"].append("forbidden")  # type: ignore[union-attr]


def test_immutable_json_metadata_serializes_with_pydantic() -> None:
    observation = ObservationBundle(
        schema_version="hyperspectrum-observation/v1",
        sample_id="sample-001",
        modality="xas",
        artifacts=(artifact(),),
        context={"instrument": {"scan_modes": ["transmission"]}},
        labels={},
        provenance={},
    )
    prediction = PredictionBundle(
        schema_version="hyperspectrum-prediction/v1",
        run_id="run-001",
        task_id="xas-denoising",
        predictions=(artifact(role="prediction"),),
        failures=({"sample_id": "sample-002", "details": {"reasons": ["out_of_range"]}},),
        provenance={
            "model_digest": "c" * 64,
            "tool_digest": "d" * 64,
            "data_digest": "e" * 64,
            "environment_digest": "f" * 64,
        },
    )

    assert observation.model_dump()["context"] == {
        "instrument": {"scan_modes": ["transmission"]}
    }
    assert json.loads(observation.model_dump_json())["context"] == {
        "instrument": {"scan_modes": ["transmission"]}
    }
    assert prediction.model_dump()["failures"] == [
        {"sample_id": "sample-002", "details": {"reasons": ["out_of_range"]}}
    ]
    assert json.loads(prediction.model_dump_json())["failures"] == [
        {"sample_id": "sample-002", "details": {"reasons": ["out_of_range"]}}
    ]


def test_non_json_mutable_metadata_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ObservationBundle(
            schema_version="hyperspectrum-observation/v1",
            sample_id="sample-001",
            modality="xas",
            artifacts=(artifact(),),
            context={"unsafe": bytearray(b"mutable")},
            labels={},
            provenance={},
        )
