"""Behavior tests for conservative XAS task readiness recommendations."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from hyperspectrum.hyperdata.models import DatasetCandidate
from hyperspectrum.tasks.recommend import (
    ReadinessVerdict,
    profile_xas_candidate,
    recommend_xas_tasks,
)


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
        content_digest="a" * 64,
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

    assert {
        (verdict.dataset_code, verdict.dataset_version, verdict.content_digest)
        for verdict in verdicts
    } == {("declared-xas-evidence", "v1", "a" * 64)}
    assert [(verdict.status, verdict.candidate_tasks) for verdict in verdicts] == [
        (status, tasks)
    ]
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
    # Label and pairing evidence were declared here, so only the parser claim is
    # reported absent; the pair roles simply never co-occurred in one observation.
    assert verdicts[0].reasons == (
        "xas_no_verified_scoreable_ground_truth",
        "xas_parser_evidence_unavailable",
    )


def test_lcf_weight_regression_is_independent_of_primary_denoising_recommendation() -> (
    None
):
    """Catches using composition truth as a denoising target or primary metric."""
    verdicts = recommend_xas_tasks(
        profile_xas_candidate(
            candidate(
                label_roles=("mixture_composition",),
                pairing_roles=("noisy_spectrum", "clean_spectrum"),
            )
        )
    )

    assert [
        (verdict.candidate_tasks, verdict.ground_truth_roles) for verdict in verdicts
    ] == [
        (("denoising",), ("clean_spectrum",)),
        (("lcf_weight_regression",), ("mixture_composition",)),
    ]
    assert all(
        "lcf_weight_regression" not in verdict.candidate_tasks
        for verdict in verdicts[:1]
    )


def test_lcf_weight_regression_requires_verified_composition_in_one_observation() -> (
    None
):
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


@pytest.mark.parametrize(
    "task,roles,split_keys,limitations",
    [
        ("denoising", ("unverified_clean_claim",), ("sample_id",), ()),
        ("denoising", ("average_spectrum",), ("sample_id",), ()),
        ("forward_spectrum_prediction", ("clean_spectrum",), ("compound_id",), ()),
        ("oxidation_state_classification", ("unknown_label",), ("sample_id",), ()),
        ("lcf_weight_regression", ("unknown_composition",), ("compound_id",), ()),
        ("forward_spectrum_prediction", ("spectrum",), ("row_id",), ()),
    ],
)
def test_scoreable_verdict_rejects_noncanonical_truth_or_unsafe_splits(
    task: str,
    roles: tuple[str, ...],
    split_keys: tuple[str, ...],
    limitations: tuple[str, ...],
) -> None:
    """Catches public constructors bypassing task truth or leakage contracts."""
    with pytest.raises(ValidationError):
        ReadinessVerdict(
            dataset_code="declared-xas-evidence",
            dataset_version="v1",
            content_digest="a" * 64,
            status="scoreable",
            reasons=("declared_test_evidence",),
            candidate_tasks=(task,),
            ground_truth_roles=roles,
            split_group_keys=split_keys,
            limitations=limitations,
        )


@pytest.mark.parametrize(
    "task,roles,split_keys,limitations",
    [
        ("denoising", ("clean_spectrum",), ("sample_id",), ()),
        (
            "denoising",
            ("average_spectrum",),
            ("sample_id", "acquisition_id"),
            ("xas_proxy_ground_truth_repeated_scan_average",),
        ),
        ("forward_spectrum_prediction", ("spectrum",), ("compound_id",), ()),
        ("oxidation_state_classification", ("oxidation_state",), ("sample_id",), ()),
        ("lcf_weight_regression", ("composition",), ("compound_id",), ()),
    ],
)
def test_scoreable_verdict_accepts_canonical_task_contracts(
    task: str,
    roles: tuple[str, ...],
    split_keys: tuple[str, ...],
    limitations: tuple[str, ...],
) -> None:
    """Catches rejecting a legitimate public scoreable task contract."""
    verdict = ReadinessVerdict(
        dataset_code="declared-xas-evidence",
        dataset_version="v1",
        content_digest="a" * 64,
        status="scoreable",
        reasons=("declared_test_evidence",),
        candidate_tasks=(task,),
        ground_truth_roles=roles,
        split_group_keys=split_keys,
        limitations=limitations,
    )

    assert verdict.candidate_tasks == (task,)


def _identity_gap_verdict(**overrides: object) -> ReadinessVerdict:
    """Run the readiness pass over a candidate with one identity field removed."""
    dataset = candidate(pairing_roles=("noisy_spectrum", "clean_spectrum")).model_copy(
        update=overrides
    )
    verdicts = recommend_xas_tasks(profile_xas_candidate(dataset))
    assert len(verdicts) == 1
    return verdicts[0]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"dataset_version": None}, "xas_dataset_version_unavailable"),
        ({"content_digest": None}, "xas_content_digest_unavailable"),
    ],
)
def test_absent_identity_degrades_to_inference_only_rather_than_blocking(
    overrides: dict[str, object], reason: str
) -> None:
    """Catches an unidentifiable candidate being refused the inference it can run.

    Inference needs neither a pinned revision nor a digest — only a *scoreable*
    claim does, because only a scoreable claim has to be reproducible later.
    Blocking is reserved for candidates we cannot read or cannot parse.
    """
    verdict = _identity_gap_verdict(**overrides)

    assert verdict.status == "inference_only"
    assert reason in verdict.reasons


def test_absent_axis_evidence_degrades_but_a_declared_invalid_axis_still_blocks() -> None:
    """Catches collapsing 'the catalog said nothing' into 'the catalog said wrong'."""
    undeclared = candidate(pairing_roles=("noisy_spectrum", "clean_spectrum"))
    evidence = dict(undeclared.evidence)
    del evidence["axis_evidence"]
    undeclared = undeclared.model_copy(update={"evidence": evidence})

    absent = recommend_xas_tasks(profile_xas_candidate(undeclared))[0]
    invalid = recommend_xas_tasks(
        profile_xas_candidate(
            candidate(axis_valid=False, pairing_roles=("noisy_spectrum", "clean_spectrum"))
        )
    )[0]

    assert absent.status == "inference_only"
    assert "xas_energy_axis_evidence_unavailable" in absent.reasons
    assert invalid.status == "blocked"
    assert "xas_energy_axis_invalid" in invalid.reasons


def test_identity_absence_still_denies_a_scoreable_verdict() -> None:
    """Catches the degraded route leaking verified truth into a scoreable claim.

    The admission gate is unchanged: a quantitative claim needs an identity to
    be reproducible against. Verified noisy/clean pairing would otherwise make
    this candidate scoreable for denoising.
    """
    verdict = _identity_gap_verdict(content_digest=None)

    assert verdict.status != "scoreable"
    assert verdict.candidate_tasks == ()
    assert verdict.ground_truth_roles == ()
    assert verdict.split_group_keys == ()


def test_unreadable_candidates_stay_blocked_and_keep_every_gap_named() -> None:
    """Catches an access failure being softened into inference-only by an
    identity gap that happens to co-occur with it."""
    dataset = candidate(access_status="archive_only").model_copy(
        update={"dataset_version": None, "content_digest": None}
    )

    verdict = recommend_xas_tasks(profile_xas_candidate(dataset))[0]

    assert verdict.status == "blocked"
    assert set(verdict.reasons) == {
        "xas_access_unavailable",
        "xas_dataset_version_unavailable",
        "xas_content_digest_unavailable",
    }


def test_a_degraded_verdict_reports_the_identity_it_still_has() -> None:
    """Catches dropping the half-identity that survives — a pinned version with
    no digest is still worth carrying to whoever has to fix it."""
    verdict = _identity_gap_verdict(content_digest=None)

    assert verdict.dataset_version == "v1"
    assert verdict.content_digest is None
    assert "xas_dataset_version_unavailable" not in verdict.reasons


def test_degraded_verdicts_say_why_absence_is_not_a_blockage() -> None:
    """Catches reason codes travelling without the sentence a human needs."""
    verdict = _identity_gap_verdict(content_digest=None)

    assert any("bytes cannot be bound" in detail for detail in verdict.details)
    assert any("Inference requires none of these" in detail for detail in verdict.details)
