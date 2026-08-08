"""Pure, evidence-led readiness recommendations for XAS candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from hyperspectrum.hyperdata.models import DatasetCandidate


class ReadinessVerdict(BaseModel):
    """An immutable task recommendation with its scoring and split contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["scoreable", "inference_only", "blocked"]
    reasons: tuple[str, ...]
    candidate_tasks: tuple[str, ...]
    ground_truth_roles: tuple[str, ...] = ()
    split_group_keys: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    details: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_scoreable_contract(self) -> Self:
        """Prevent a quantitative recommendation without truth and leakage guards."""
        if self.status == "scoreable":
            if not self.candidate_tasks:
                raise ValueError("scoreable verdicts require a candidate task")
            if not self.ground_truth_roles:
                raise ValueError("scoreable verdicts require a ground-truth role")
            if not self.split_group_keys:
                raise ValueError("scoreable verdicts require split group keys")
        return self


class XASEvidenceObservation(BaseModel):
    """Verified task-relevant facts retained from one discovery observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label_verified: bool
    label_roles: tuple[str, ...]
    pairing_verified: bool
    pairing_roles: tuple[str, ...]
    proxy_kind: str | None


class XASCandidateProfile(BaseModel):
    """The conservative, task-relevant view of one discovered candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_code: str
    access_available: bool
    energy_axis_valid: bool
    verified_label_roles: tuple[str, ...]
    verified_pairing_roles: tuple[str, ...]
    observations: tuple[XASEvidenceObservation, ...]


def profile_xas_candidate(candidate: DatasetCandidate) -> XASCandidateProfile:
    """Extract immutable task evidence without promoting raw catalog claims.

    Aggregate access, axis, and role evidence supplies the conservative gate.
    Individual observations remain necessary because an aggregate role union does
    not prove that all facts needed for one task occurred together.
    """
    evidence = _mapping(candidate.evidence)
    aggregate_labels = _mapping(evidence.get("label_evidence"))
    aggregate_pairings = _mapping(evidence.get("pairing_evidence"))
    observations = tuple(
        _profile_observation(observation)
        for observation in _mappings(evidence.get("observations"))
    )
    energy_axis = _mapping(_mapping(evidence.get("axis_evidence")).get("energy_axis"))
    return XASCandidateProfile(
        dataset_code=candidate.dataset_code,
        access_available=evidence.get("access_status") == "admitted_catalog",
        energy_axis_valid=(
            energy_axis.get("valid") is True and _nonblank(energy_axis.get("unit")) is not None
        ),
        verified_label_roles=(
            _strings(aggregate_labels.get("ground_truth_roles"))
            if aggregate_labels.get("verified") is True
            else ()
        ),
        verified_pairing_roles=(
            _strings(aggregate_pairings.get("roles"))
            if aggregate_pairings.get("verified") is True
            else ()
        ),
        observations=observations,
    )


def recommend_xas_tasks(profile: XASCandidateProfile) -> tuple[ReadinessVerdict, ...]:
    """Recommend only XAS tasks whose required evidence is independently verified."""
    blocked = _blocked_verdict(profile)
    if blocked is not None:
        return (blocked,)

    verdicts: list[ReadinessVerdict] = []
    if _aggregate_and_observation_pair(profile, "noisy_spectrum", "clean_spectrum"):
        verdicts.append(
            _scoreable(
                task="denoising",
                reason="xas_verified_noisy_clean_pair",
                roles=("clean_spectrum",),
                split_keys=("sample_id", "compound_id"),
                details=("A verified noisy/clean spectrum pair supplies clean-spectrum truth.",),
            )
        )
    elif _aggregate_and_observation_pair(profile, "repeated_scan", "average_spectrum") and any(
        observation.proxy_kind == "repeated_scan_average" for observation in profile.observations
        if observation.pairing_verified
        and {"repeated_scan", "average_spectrum"}.issubset(observation.pairing_roles)
    ):
        verdicts.append(
            _scoreable(
                task="denoising",
                reason="xas_verified_repeated_scan_average_proxy",
                roles=("average_spectrum",),
                split_keys=("sample_id", "acquisition_id"),
                limitations=("xas_proxy_ground_truth_repeated_scan_average",),
                details=(
                    "The repeated-scan average is a documented proxy ground truth, not independent clean truth.",
                ),
            )
        )
    if _aggregate_and_observation_pair(profile, "structure", "spectrum"):
        verdicts.append(
            _scoreable(
                task="forward_spectrum_prediction",
                reason="xas_verified_structure_spectrum_pair",
                roles=("spectrum",),
                split_keys=("compound_id",),
                details=("A verified structure/spectrum pair supplies spectrum truth.",),
            )
        )
    if _aggregate_and_observation_label(profile, "oxidation_state"):
        verdicts.append(
            _scoreable(
                task="oxidation_state_classification",
                reason="xas_verified_oxidation_state_labels",
                roles=("oxidation_state",),
                split_keys=("sample_id", "compound_id"),
                details=("Verified oxidation-state labels support classification.",),
            )
        )
    if _aggregate_and_observation_label(profile, "mixture_composition"):
        verdicts.append(_lcf_verdict("mixture_composition"))
    elif _aggregate_and_observation_label(profile, "composition"):
        verdicts.append(_lcf_verdict("composition"))

    if verdicts:
        return tuple(verdicts)
    return (
        ReadinessVerdict(
            status="inference_only",
            reasons=("xas_no_verified_scoreable_ground_truth",),
            candidate_tasks=(),
            details=("No verified task truth is available for quantitative scoring.",),
        ),
    )


def _blocked_verdict(profile: XASCandidateProfile) -> ReadinessVerdict | None:
    reasons: list[str] = []
    details: list[str] = []
    if not profile.access_available:
        reasons.append("xas_access_unavailable")
        details.append("The candidate is not available through the admitted catalog.")
    if not profile.energy_axis_valid:
        reasons.append("xas_energy_axis_invalid")
        details.append("The candidate has no verified energy axis with a declared unit.")
    if not reasons:
        return None
    return ReadinessVerdict(
        status="blocked",
        reasons=tuple(reasons),
        candidate_tasks=(),
        details=tuple(details),
    )


def _scoreable(
    *,
    task: str,
    reason: str,
    roles: tuple[str, ...],
    split_keys: tuple[str, ...],
    details: tuple[str, ...],
    limitations: tuple[str, ...] = (),
) -> ReadinessVerdict:
    """Build one separately scoreable task recommendation."""
    reasons = (reason, *limitations)
    return ReadinessVerdict(
        status="scoreable",
        reasons=reasons,
        candidate_tasks=(task,),
        ground_truth_roles=roles,
        split_group_keys=split_keys,
        limitations=limitations,
        details=details,
    )


def _lcf_verdict(role: str) -> ReadinessVerdict:
    """Keep LCF weight regression a separate composition-truth recommendation."""
    return _scoreable(
        task="lcf_weight_regression",
        reason="xas_verified_mixture_composition_labels",
        roles=(role,),
        split_keys=("compound_id",),
        details=("Verified mixture/composition truth supports independent LCF-weight regression.",),
    )


def _aggregate_and_observation_pair(
    profile: XASCandidateProfile, *roles: str
) -> bool:
    """Require the complete verified pair in aggregate and one raw observation."""
    required = set(roles)
    return required.issubset(profile.verified_pairing_roles) and any(
        observation.pairing_verified and required.issubset(observation.pairing_roles)
        for observation in profile.observations
    )


def _aggregate_and_observation_label(profile: XASCandidateProfile, role: str) -> bool:
    """Require a verified label in aggregate evidence and the same observation."""
    return role in profile.verified_label_roles and any(
        observation.label_verified and role in observation.label_roles
        for observation in profile.observations
    )


def _profile_observation(observation: Mapping[str, object]) -> XASEvidenceObservation:
    """Retain only raw facts that can prove a task's scoring contract."""
    labels = _mapping(observation.get("label_evidence"))
    pairings = _mapping(observation.get("pairing_evidence"))
    return XASEvidenceObservation(
        label_verified=labels.get("verified") is True,
        label_roles=_strings(labels.get("ground_truth_roles")),
        pairing_verified=pairings.get("verified") is True,
        pairing_roles=_strings(pairings.get("roles")),
        proxy_kind=_nonblank(pairings.get("proxy_kind")),
    )


def _mappings(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(_mapping(item) for item in value if isinstance(item, Mapping))


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _nonblank(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None
