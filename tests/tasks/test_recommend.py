"""Behavior tests for conservative XAS task readiness recommendations."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from hyperspectrum.hyperdata.models import DatasetCandidate
from hyperspectrum.tasks.recommend import profile_xas_candidate, recommend_xas_tasks


def candidate(
    *,
    access_status: str = "admitted_catalog",
    axis_valid: bool = True,
    label_roles: Sequence[str] = (),
    pairing_roles: Sequence[str] = (),
    proxy_kind: str | None = None,
    observations: Sequence[dict[str, object]] | None = None,
) -> DatasetCandidate:
    """Build declared aggregate evidence and small raw-observation evidence."""
    label_evidence: dict[str, object] = {
        "verified": bool(label_roles),
        "ground_truth_roles": list(label_roles),
    }
    pairing_evidence: dict[str, object] = {
        "verified": bool(pairing_roles),
        "roles": list(pairing_roles),
    }
    if proxy_kind is not None:
        pairing_evidence["proxy_kind"] = proxy_kind
    if observations is None:
        observations = (
            {
                "label_evidence": label_evidence,
                "pairing_evidence": pairing_evidence,
            },
        )
    return DatasetCandidate(
        dataset_code="declared-xas-evidence",
        dataset_version="v1",
        content_digest="digest",
        title="Declared XAS evidence",
        description="",
        file_count=1,
        parsed_file_count=1,
        formats=("xdi",),
        license="CC-BY-4.0",
        evidence={
            "access_status": access_status,
            "axis_evidence": {"energy_axis": {"valid": axis_valid, "unit": "eV"}},
            "label_evidence": label_evidence,
            "pairing_evidence": pairing_evidence,
            "observations": observations,
        },
    )


@pytest.mark.parametrize(
    ("name", "dataset", "status", "tasks", "roles", "limitations"),
    [
        (
            "noisy_clean_pairs",
            candidate(pairing_roles=("noisy_spectrum", "clean_spectrum")),
            "scoreable",
            ("denoising",),
            ("clean_spectrum",),
            (),
        ),
        (
            "documented_repeated_scan_average",
            candidate(
                pairing_roles=("repeated_scan", "average_spectrum"),
                proxy_kind="repeated_scan_average",
            ),
            "scoreable",
            ("denoising",),
            ("average_spectrum",),
            ("xas_proxy_ground_truth_repeated_scan_average",),
        ),
        (
            "structure_spectrum_pairs",
            candidate(pairing_roles=("structure", "spectrum")),
            "scoreable",
            ("forward_spectrum_prediction",),
            ("spectrum",),
            (),
        ),
        (
            "oxidation_labels",
            candidate(label_roles=("oxidation_state",)),
            "scoreable",
            ("oxidation_state_classification",),
            ("oxidation_state",),
            (),
        ),
        (
            "spectra_only",
            candidate(),
            "inference_only",
            (),
            (),
            (),
        ),
        (
            "inaccessible_archive",
            candidate(access_status="archive_only"),
            "blocked",
            (),
            (),
            (),
        ),
        (
            "invalid_energy_axis",
            candidate(axis_valid=False),
            "blocked",
            (),
            (),
            (),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_recommendations_follow_verified_xas_evidence(
    name: str,
    dataset: DatasetCandidate,
    status: str,
    tasks: tuple[str, ...],
    roles: tuple[str, ...],
    limitations: tuple[str, ...],
) -> None:
    """Catches task admission from titles, aggregates, or unverified claims."""
    _ = name

    verdicts = recommend_xas_tasks(profile_xas_candidate(dataset))

    assert [(verdict.status, verdict.candidate_tasks) for verdict in verdicts] == [(status, tasks)]
    assert verdicts[0].ground_truth_roles == roles
    assert verdicts[0].limitations == limitations
    if status == "scoreable":
        assert verdicts[0].ground_truth_roles
        assert verdicts[0].split_group_keys
        assert set(verdicts[0].split_group_keys) & {
            "sample_id",
            "compound_id",
            "acquisition_id",
        }
    else:
        assert verdicts[0].split_group_keys == ()


def test_repeated_scan_proxy_groups_by_sample_and_acquisition() -> None:
    """Catches adjacent repeated scans leaking between denoising splits."""
    verdict = recommend_xas_tasks(
        profile_xas_candidate(
            candidate(
                pairing_roles=("repeated_scan", "average_spectrum"),
                proxy_kind="repeated_scan_average",
            )
        )
    )[0]

    assert verdict.split_group_keys == ("sample_id", "acquisition_id")
    assert "xas_proxy_ground_truth_repeated_scan_average" in verdict.reasons
    assert verdict.details == (
        "The repeated-scan average is a documented proxy ground truth, not independent clean truth.",
    )


def test_raw_observations_must_contain_complete_verified_pairing_evidence() -> None:
    """Catches scoreable pair roles assembled across independent observations."""
    dataset = candidate(
        pairing_roles=("noisy_spectrum", "clean_spectrum"),
        observations=(
            {
                "label_evidence": {"verified": False, "ground_truth_roles": []},
                "pairing_evidence": {"verified": True, "roles": ["noisy_spectrum"]},
            },
            {
                "label_evidence": {"verified": False, "ground_truth_roles": []},
                "pairing_evidence": {"verified": True, "roles": ["clean_spectrum"]},
            },
        ),
    )

    verdicts = recommend_xas_tasks(profile_xas_candidate(dataset))

    assert [(verdict.status, verdict.candidate_tasks) for verdict in verdicts] == [
        ("inference_only", ())
    ]
    assert verdicts[0].reasons == ("xas_no_verified_scoreable_ground_truth",)


def test_lcf_weight_regression_is_independent_of_primary_denoising_recommendation() -> None:
    """Catches using composition truth as a denoising target or primary metric."""
    verdicts = recommend_xas_tasks(
        profile_xas_candidate(
            candidate(
                label_roles=("mixture_composition",),
                pairing_roles=("noisy_spectrum", "clean_spectrum"),
            )
        )
    )

    assert [(verdict.candidate_tasks, verdict.ground_truth_roles) for verdict in verdicts] == [
        (("denoising",), ("clean_spectrum",)),
        (("lcf_weight_regression",), ("mixture_composition",)),
    ]
    assert all("lcf_weight_regression" not in verdict.candidate_tasks for verdict in verdicts[:1])


def test_lcf_weight_regression_requires_verified_composition_in_one_observation() -> None:
    """Catches LCF admission from an aggregate label claim or an unverified observation."""
    dataset = candidate(
        label_roles=("mixture_composition",),
        observations=(
            {
                "label_evidence": {
                    "verified": False,
                    "ground_truth_roles": ["mixture_composition"],
                },
                "pairing_evidence": {"verified": False, "roles": []},
            },
        ),
    )

    verdicts = recommend_xas_tasks(profile_xas_candidate(dataset))

    assert [(verdict.status, verdict.candidate_tasks) for verdict in verdicts] == [
        ("inference_only", ())
    ]
