"""Pure, evidence-led readiness recommendations for XAS candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from hyperspectrum.hyperdata.models import DatasetCandidate

_SAFE_SPLIT_GROUP_KEYS = frozenset({"sample_id", "compound_id", "acquisition_id"})
_PROXY_LIMITATION = "xas_proxy_ground_truth_repeated_scan_average"
_TASK_GROUND_TRUTH_ROLES = {
    "denoising": frozenset({"clean_spectrum", "average_spectrum"}),
    "forward_spectrum_prediction": frozenset({"spectrum"}),
    "oxidation_state_classification": frozenset({"oxidation_state"}),
    "lcf_weight_regression": frozenset({"mixture_composition", "composition"}),
}


class ReadinessVerdict(BaseModel):
    """An immutable task recommendation with its scoring and split contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_code: str
    dataset_version: str | None
    content_digest: str | None
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
        if self.status != "scoreable":
            if self.candidate_tasks or self.ground_truth_roles or self.split_group_keys:
                raise ValueError("non-scoreable verdicts require empty task contracts")
            return self
        if len(self.candidate_tasks) != 1:
            raise ValueError("scoreable verdicts require exactly one candidate task")
        task = self.candidate_tasks[0]
        allowed_roles = _TASK_GROUND_TRUTH_ROLES.get(task)
        if allowed_roles is None:
            raise ValueError("scoreable verdicts require a canonical candidate task")
        if (
            len(self.ground_truth_roles) != 1
            or self.ground_truth_roles[0] not in allowed_roles
        ):
            raise ValueError(
                "scoreable verdicts require a task-consistent ground-truth role"
            )
        if not _SAFE_SPLIT_GROUP_KEYS.intersection(self.split_group_keys):
            raise ValueError("scoreable verdicts require a safe split group key")
        if (
            task == "denoising"
            and self.ground_truth_roles == ("average_spectrum",)
            and _PROXY_LIMITATION not in self.limitations
        ):
            raise ValueError(
                "average-spectrum denoising requires a documented proxy limitation"
            )
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
    dataset_version: str | None
    content_digest: str | None
    access_available: bool
    energy_axis_valid: bool
    # Whether the catalog published each claim at all.  An undeclared claim is a
    # different fact from a declared claim that fails validation, and the two
    # must not collapse into one reason code.
    energy_axis_declared: bool
    parser_declared: bool
    label_declared: bool
    pairing_declared: bool
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
    aggregate_parser = _mapping(evidence.get("parser_status"))
    aggregate_axis = _mapping(evidence.get("axis_evidence"))
    aggregate_labels = _mapping(evidence.get("label_evidence"))
    aggregate_pairings = _mapping(evidence.get("pairing_evidence"))
    observations = tuple(
        _profile_observation(observation)
        for observation in _mappings(evidence.get("observations"))
    )
    energy_axis = _mapping(aggregate_axis.get("energy_axis"))
    return XASCandidateProfile(
        dataset_code=candidate.dataset_code,
        dataset_version=candidate.dataset_version,
        content_digest=candidate.content_digest,
        access_available=evidence.get("access_status") == "admitted_catalog",
        energy_axis_valid=(
            energy_axis.get("valid") is True
            and _nonblank(energy_axis.get("unit")) is not None
        ),
        energy_axis_declared=_declared(aggregate_axis, energy_axis),
        parser_declared=_declared(aggregate_parser, aggregate_parser),
        label_declared=_declared(aggregate_labels, aggregate_labels),
        pairing_declared=_declared(aggregate_pairings, aggregate_pairings),
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
                profile=profile,
                task="denoising",
                reason="xas_verified_noisy_clean_pair",
                roles=("clean_spectrum",),
                split_keys=("sample_id", "compound_id"),
                details=(
                    "A verified noisy/clean spectrum pair supplies clean-spectrum truth.",
                ),
            )
        )
    elif _aggregate_and_observation_pair(
        profile, "repeated_scan", "average_spectrum"
    ) and any(
        observation.proxy_kind == "repeated_scan_average"
        for observation in profile.observations
        if observation.pairing_verified
        and {"repeated_scan", "average_spectrum"}.issubset(observation.pairing_roles)
    ):
        verdicts.append(
            _scoreable(
                profile=profile,
                task="denoising",
                reason="xas_verified_repeated_scan_average_proxy",
                roles=("average_spectrum",),
                split_keys=("sample_id", "acquisition_id"),
                limitations=(_PROXY_LIMITATION,),
                details=(
                    "The repeated-scan average is a documented proxy ground truth, not independent clean truth.",
                ),
            )
        )
    if _aggregate_and_observation_pair(profile, "structure", "spectrum"):
        verdicts.append(
            _scoreable(
                profile=profile,
                task="forward_spectrum_prediction",
                reason="xas_verified_structure_spectrum_pair",
                roles=("spectrum",),
                split_keys=("compound_id",),
                details=(
                    "A verified structure/spectrum pair supplies spectrum truth.",
                ),
            )
        )
    if _aggregate_and_observation_label(profile, "oxidation_state"):
        verdicts.append(
            _scoreable(
                profile=profile,
                task="oxidation_state_classification",
                reason="xas_verified_oxidation_state_labels",
                roles=("oxidation_state",),
                split_keys=("sample_id", "compound_id"),
                details=("Verified oxidation-state labels support classification.",),
            )
        )
    if _aggregate_and_observation_label(profile, "mixture_composition"):
        verdicts.append(_lcf_verdict(profile, "mixture_composition"))
    elif _aggregate_and_observation_label(profile, "composition"):
        verdicts.append(_lcf_verdict(profile, "composition"))

    if verdicts:
        return tuple(verdicts)
    # Missing task truth does not block inference, but the reason it is missing
    # is worth stating: an undeclared claim and a declared-but-unverified one
    # lead to different fixes.
    reasons = ["xas_no_verified_scoreable_ground_truth"]
    details = ["No verified task truth is available for quantitative scoring."]
    for declared, code, detail in (
        (
            profile.label_declared,
            "xas_label_evidence_unavailable",
            "The catalog published no ground-truth label evidence.",
        ),
        (
            profile.pairing_declared,
            "xas_pairing_evidence_unavailable",
            "The catalog published no spectrum-pairing evidence.",
        ),
        (
            profile.parser_declared,
            "xas_parser_evidence_unavailable",
            "The catalog published no parser evidence for the candidate's files.",
        ),
    ):
        if not declared:
            reasons.append(code)
            details.append(detail)
    return (
        ReadinessVerdict(
            dataset_code=profile.dataset_code,
            dataset_version=profile.dataset_version,
            content_digest=profile.content_digest,
            status="inference_only",
            reasons=tuple(reasons),
            candidate_tasks=(),
            details=tuple(details),
        ),
    )


def _blocked_verdict(profile: XASCandidateProfile) -> ReadinessVerdict | None:
    """Enumerate every unmet admission condition, naming absence as absence.

    A candidate the catalog never described is not the same as one it described
    badly, and the M0 gate needs both a pinned version and a content digest
    before any quantitative claim can be reproduced.
    """
    reasons: list[str] = []
    details: list[str] = []
    if not profile.access_available:
        reasons.append("xas_access_unavailable")
        details.append("The candidate is not available through the admitted catalog.")
    if profile.dataset_version is None:
        reasons.append("xas_dataset_version_unavailable")
        details.append(
            "The catalog record carries no dataset version, so the candidate "
            "cannot be pinned to a reproducible revision."
        )
    if profile.content_digest is None:
        reasons.append("xas_content_digest_unavailable")
        details.append(
            "The catalog record carries no content digest, so the candidate's "
            "bytes cannot be bound to any later result."
        )
    if not profile.energy_axis_declared:
        reasons.append("xas_energy_axis_evidence_unavailable")
        details.append(
            "The catalog published no energy-axis evidence for this candidate; "
            "the axis is unknown, not known to be wrong."
        )
    elif not profile.energy_axis_valid:
        reasons.append("xas_energy_axis_invalid")
        details.append(
            "The candidate has no verified energy axis with a declared unit."
        )
    if not reasons:
        return None
    return ReadinessVerdict(
        dataset_code=profile.dataset_code,
        dataset_version=profile.dataset_version,
        content_digest=profile.content_digest,
        status="blocked",
        reasons=tuple(reasons),
        candidate_tasks=(),
        details=tuple(details),
    )


def _scoreable(
    *,
    profile: XASCandidateProfile,
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
        dataset_code=profile.dataset_code,
        dataset_version=profile.dataset_version,
        content_digest=profile.content_digest,
        status="scoreable",
        reasons=reasons,
        candidate_tasks=(task,),
        ground_truth_roles=roles,
        split_group_keys=split_keys,
        limitations=limitations,
        details=details,
    )


def _lcf_verdict(profile: XASCandidateProfile, role: str) -> ReadinessVerdict:
    """Keep LCF weight regression a separate composition-truth recommendation."""
    return _scoreable(
        profile=profile,
        task="lcf_weight_regression",
        reason="xas_verified_mixture_composition_labels",
        roles=(role,),
        split_keys=("compound_id",),
        details=(
            "Verified mixture/composition truth supports independent LCF-weight regression.",
        ),
    )


def _aggregate_and_observation_pair(profile: XASCandidateProfile, *roles: str) -> bool:
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


def _declared(
    container: Mapping[str, object], claim: Mapping[str, object]
) -> bool:
    """Read the explicit declaration flag, or infer it from a pre-flag claim.

    Evidence assembled before the flag existed carries no `declared` key; such
    evidence was hand-built from a source that did publish the claim, so a
    non-empty claim still counts as declared.
    """
    flag = container.get("declared")
    if isinstance(flag, bool):
        return flag
    return bool(claim)


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
    return tuple(
        item.strip() for item in value if isinstance(item, str) and item.strip()
    )


def _nonblank(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None
