"""The inference-only sibling of the ACE execution facade.

:func:`~hyperspectrum.execution.ace.execute_ace_xas_denoising` is handed a whole
benchmark directory: it reads the manifest, loads denoising *pairs*, picks the
test split itself and hands the noisy half to a tool. That is the right shape for
the self-scored path, where one container holds both halves and ACE scores the
bundle afterwards.

It is the wrong shape for a dual-asset candidate. There, the candidate mount is
one file — the track's ``inference.npz`` — and the target, the split manifest and
the profile are scorer-only assets the container never receives. This entry point
is that path: same tool, same parameters, same numerics, one asset, no ground
truth anywhere in reach.

Three things follow from that and are enforced rather than documented:

* **It refuses anything that is not exactly the inference contract.** An asset
  carrying ``proxy_full_count``, ``splits`` or any other scorer-only array is not
  a narrower benchmark, it is the scorer's half arriving where it may not, and
  the refusal names the contract rather than the stray key.
* **It plans as a** :class:`~hyperspectrum.execution.plan.RunPlanV2`. That plan's
  data identity is one asset digest. It has no benchmark-asset field, so "this
  run touched no ground truth" is a property of the plan's shape and not of this
  function remembering not to fill one in.
* **It answers exactly the rows it was given**, in the asset's own order. Which
  rows those are is the profile's decision, taken where the split lives; a
  candidate that selected its own would be answering a question nobody asked.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np

from hyperspectrum.contracts import PredictionBundleV2, TaskSpec
from hyperspectrum.registry import ResourceBudget, ToolRegistry

from .local import execute_local_run
from .plan import RunPlanV2, build_inference_run_plan
from .tools import load_tool_by_id

#: The exact key set of a candidate-facing inference asset. Stated here as well
#: as in ``hyperspectrum.datasets.cu_cha`` because this module must be able to
#: refuse a foreign asset without importing a dataset profile — a candidate
#: facade that had to know which profile produced its input would be a facade
#: only that profile could use.
INFERENCE_ASSET_KEYS = frozenset(
    {"energy", "energy_unit", "group_ids", "noisy", "sample_ids"}
)
_TASK_ID = "xas-denoising"


@dataclass(frozen=True, slots=True)
class AceXasInferenceResult:
    """One complete inference run plus the exact files ACE may collect."""

    plan: RunPlanV2
    bundle: PredictionBundleV2
    artifact_paths: tuple[Path, ...]
    #: The digest of the bytes actually read. ACE pins the asset it published and
    #: re-checks this, so it is returned rather than left for a caller to
    #: recompute from a path that may no longer hold the same file.
    inference_sha256: str
    answered_sample_ids: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        """Return the number of complete predictions in this successful run."""

        return len(self.bundle.predictions)


def execute_ace_xas_inference_only(
    *,
    inference_asset: Path,
    output_directory: Path,
    task_id: str,
    tool_id: str,
    parameters: Mapping[str, object],
    dataset_code: str,
    dataset_version: str,
    max_samples: int | None = None,
    weight_files: Sequence[Path] = (),
) -> AceXasInferenceResult:
    """Denoise one mounted inference asset without reading any ground truth."""

    if task_id != _TASK_ID:
        raise ValueError(f"ACE XAS inference supports only task {_TASK_ID!r}")
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples < 1
    ):
        raise ValueError("max_samples must be a positive integer or None")
    if not dataset_code.strip() or not dataset_version.strip():
        raise ValueError("inference dataset identity must be non-blank")

    try:
        asset_bytes = inference_asset.read_bytes()
    except OSError as error:
        raise ValueError(f"inference asset is not readable: {error}") from error
    inference_sha256 = sha256(asset_bytes).hexdigest()
    sample_ids = _asset_sample_ids(asset_bytes)
    selected_sample_ids = (
        sample_ids if max_samples is None else sample_ids[:max_samples]
    )
    if not selected_sample_ids:
        raise ValueError("inference asset must declare at least one sample")

    task = TaskSpec(
        schema_version="hyperspectrum-task/v1",
        id=task_id,
        modality="xas",
        task_type="denoising",
        input_roles=("raw_signal",),
        output_kind="dense_array",
        # Declared, and then never used to reach one: the tool-compatibility gate
        # is defined over the task, and a task that lied about its shape to avoid
        # naming ground truth would be a different task from the scored one.
        ground_truth_roles=("clean_spectrum",),
        split_group_keys=("compound_id",),
        metrics=(),
    )
    tool = load_tool_by_id(tool_id)
    availability = ToolRegistry((tool,)).availability(tool)
    device = parameters.get("device")
    plan = build_inference_run_plan(
        task=task,
        dataset_code=dataset_code,
        dataset_version=dataset_version,
        inference_asset_digest=inference_sha256,
        tool=tool,
        availability=availability,
        backend="local",
        resources=ResourceBudget(
            cpu=tool.resources.cpu,
            memory_gb=tool.resources.memory_gb,
            gpu_available=isinstance(device, str) and device.startswith("cuda"),
        ),
        max_samples=(
            len(selected_sample_ids) if max_samples is None else max_samples
        ),
        selected_sample_ids=selected_sample_ids,
        output_directory=output_directory,
        dry_run=False,
        parameters=parameters,
        data_origin="real",
    )
    bundle = cast(
        PredictionBundleV2,
        execute_local_run(
            plan,
            tool=tool,
            selected_sample_ids=selected_sample_ids,
            source_npz=inference_asset,
            weight_files=tuple(weight_files),
        ),
    )
    if bundle.failures or len(bundle.predictions) != len(selected_sample_ids):
        raise RuntimeError(
            "ACE XAS inference is incomplete: "
            f"predictions={len(bundle.predictions)}, "
            f"failures={len(bundle.failures)}, "
            f"planned={len(selected_sample_ids)}"
        )
    artifact_paths = (
        output_directory / "predictions.json",
        output_directory / "run.json",
        *(output_directory / prediction.uri for prediction in bundle.predictions),
    )
    return AceXasInferenceResult(
        plan=plan,
        bundle=bundle,
        artifact_paths=artifact_paths,
        inference_sha256=inference_sha256,
        answered_sample_ids=selected_sample_ids,
    )


def _asset_sample_ids(asset_bytes: bytes) -> tuple[str, ...]:
    """Read the asset's declared sample order, refusing any wider key set.

    Deliberately before planning and before any tool code is resolved: a mount
    that arrived with the answers in it should cost nothing but a refusal, and a
    refusal that happened after a model ran would be a refusal issued by a
    process that had already read what it was refusing.
    """

    try:
        with np.load(io.BytesIO(asset_bytes), allow_pickle=False) as loaded:
            observed = set(loaded.files)
            if observed != INFERENCE_ASSET_KEYS:
                unexpected = sorted(observed - INFERENCE_ASSET_KEYS)
                missing = sorted(INFERENCE_ASSET_KEYS - observed)
                raise ValueError(
                    "inference asset keys do not match the exact inference-only "
                    f"input contract; unexpected={unexpected}, missing={missing}"
                )
            values = np.array(loaded["sample_ids"], copy=True)
    except (OSError, EOFError) as error:
        raise ValueError("inference asset is not a readable NPZ") from error
    if values.dtype.fields is not None or values.dtype.kind != "U" or values.ndim != 1:
        raise ValueError("inference asset sample_ids must be a plain string array")
    sample_ids = tuple(str(value) for value in values.tolist())
    if any(not sample_id.strip() for sample_id in sample_ids) or len(
        set(sample_ids)
    ) != len(sample_ids):
        raise ValueError("inference asset sample_ids must be unique and non-blank")
    return sample_ids


__all__ = ["AceXasInferenceResult", "execute_ace_xas_inference_only"]
