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
    PredictionBundleV2,
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


def test_prediction_v1_remains_compatible_with_nonempty_legacy_provenance() -> None:
    prediction = PredictionBundle(
        schema_version="hyperspectrum-prediction/v1",
        run_id="run-legacy",
        task_id="xas-denoising",
        predictions=(artifact(role="prediction"),),
        failures=(),
        provenance={
            "model_digest": "legacy-model",
            "tool_digest": "legacy-tool",
            "data_digest": "legacy-data",
            "environment_digest": "legacy-environment",
        },
    )

    assert prediction.schema_version == "hyperspectrum-prediction/v1"


def prediction_v2_provenance() -> dict[str, str]:
    return {
        "model_digest": "1" * 64,
        "tool_digest": "2" * 64,
        "implementation_digest": "3" * 64,
        "weight_digest": "none",
        "data_digest": "4" * 64,
        "environment_digest": "5" * 64,
        "plan_digest": "6" * 64,
        "plan_schema_version": "hyperspectrum-run-plan/v2",
        "dataset_code": "public-xas",
        "dataset_version": "2026.08.1",
    }


def test_prediction_v2_accepts_complete_selection_bound_provenance() -> None:
    prediction = PredictionBundleV2(
        schema_version="hyperspectrum-prediction/v2",
        run_id="run-v2",
        task_id="xas-denoising",
        predictions=(artifact(role="prediction"),),
        failures=(),
        provenance=prediction_v2_provenance(),
    )

    assert prediction.provenance["plan_schema_version"] == (
        "hyperspectrum-run-plan/v2"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_digest", "not-a-sha"),
        ("tool_digest", "A" * 64),
        ("implementation_digest", ""),
        ("weight_digest", "missing"),
        ("data_digest", "f" * 63),
        ("environment_digest", None),
        ("plan_digest", "0"),
        ("plan_schema_version", "hyperspectrum-run-plan/v1"),
        ("dataset_code", "   "),
        ("dataset_version", ""),
    ],
)
def test_prediction_v2_rejects_incomplete_or_unformatted_provenance(
    field: str, value: object
) -> None:
    provenance: dict[str, object] = prediction_v2_provenance()
    provenance[field] = value

    with pytest.raises(ValidationError, match=field):
        PredictionBundleV2(
            schema_version="hyperspectrum-prediction/v2",
            run_id="run-v2",
            task_id="xas-denoising",
            predictions=(artifact(role="prediction"),),
            failures=(),
            provenance=provenance,
        )


def test_prediction_v2_rejects_missing_required_provenance_field() -> None:
    provenance = prediction_v2_provenance()
    del provenance["implementation_digest"]

    with pytest.raises(ValidationError, match="implementation_digest"):
        PredictionBundleV2(
            schema_version="hyperspectrum-prediction/v2",
            run_id="run-v2",
            task_id="xas-denoising",
            predictions=(artifact(role="prediction"),),
            failures=(),
            provenance=provenance,
        )


@pytest.mark.parametrize(
    "field",
    (
        "model_digest",
        "tool_digest",
        "implementation_digest",
        "weight_digest",
        "data_digest",
        "environment_digest",
        "plan_digest",
    ),
)
def test_prediction_v2_rejects_all_zero_sha256_sentinels(field: str) -> None:
    provenance = prediction_v2_provenance()
    provenance[field] = "0" * 64

    with pytest.raises(ValidationError, match=field):
        PredictionBundleV2(
            schema_version="hyperspectrum-prediction/v2",
            run_id="run-v2",
            task_id="xas-denoising",
            predictions=(artifact(role="prediction"),),
            failures=(),
            provenance=provenance,
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


def test_observation_model_copy_revalidates_and_freezes_metadata_update() -> None:
    observation = ObservationBundle(
        schema_version="hyperspectrum-observation/v1",
        sample_id="sample-001",
        modality="xas",
        artifacts=(artifact(),),
        context={},
        labels={},
        provenance={},
    )
    updated_context = {"nested": ["original"]}

    copied = observation.model_copy(update={"context": updated_context})
    updated_context["nested"].append("caller-mutation")

    assert copied.model_dump()["context"] == {"nested": ["original"]}
    with pytest.raises(AttributeError):
        copied.context["nested"].append("forbidden")  # type: ignore[union-attr]
    with pytest.raises(ValidationError):
        observation.model_copy(update={"context": {"unsafe": bytearray(b"mutable")}})


def test_prediction_model_copy_revalidates_and_freezes_metadata_update() -> None:
    prediction = PredictionBundle(
        schema_version="hyperspectrum-prediction/v1",
        run_id="run-001",
        task_id="xas-denoising",
        predictions=(artifact(role="prediction"),),
        failures=(),
        provenance={
            "model_digest": "c" * 64,
            "tool_digest": "d" * 64,
            "data_digest": "e" * 64,
            "environment_digest": "f" * 64,
        },
    )
    updated_provenance = {
        "model_digest": "c" * 64,
        "tool_digest": "d" * 64,
        "data_digest": "e" * 64,
        "environment_digest": "f" * 64,
        "nested": ["original"],
    }
    updated_failures = [{"details": {"reasons": ["original"]}}]

    copied = prediction.model_copy(
        update={"provenance": updated_provenance, "failures": updated_failures}
    )
    updated_provenance["nested"].append("caller-mutation")
    updated_failures[0]["details"]["reasons"].append("caller-mutation")

    assert copied.model_dump()["provenance"]["nested"] == ["original"]
    assert copied.model_dump()["failures"] == [{"details": {"reasons": ["original"]}}]
    with pytest.raises(AttributeError):
        copied.provenance["nested"].append("forbidden")  # type: ignore[union-attr]
    with pytest.raises(AttributeError):
        copied.failures[0]["details"]["reasons"].append("forbidden")  # type: ignore[union-attr]
    with pytest.raises(ValidationError):
        prediction.model_copy(update={"provenance": {}})
