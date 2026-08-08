from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from hyperspectrum.registry.loader import ToolRegistry, load_tool_manifest
from hyperspectrum.registry.models import (
    LicensePolicy,
    ResourceBudget,
    ToolAvailability,
    ToolMatchRequest,
)

ROOT = Path(__file__).resolve().parents[2]


def manifest() -> dict[str, object]:
    """A hand-authored, valid external tool declaration for loader boundary tests."""
    return {
        "schema_version": "hyperspectrum-tool/v1",
        "id": "xasdenoise",
        "version": "22363b96cd1e797f7b34d39948f50f4d4b899a2c",
        "modalities": ["xas"],
        "tasks": ["denoising"],
        "runtime": {"kind": "python"},
        "entrypoint": "adapters/xasdenoise/run.py",
        "inputs": [{"role": "raw_signal", "kind": "dense_array"}],
        "outputs": [{"role": "denoised_signal", "kind": "dense_array"}],
        "resources": {"cpu": 4, "memory_gb": 16, "gpu": "optional"},
        "verify": ["python", "-m", "hyperspectrum.verify_xasdenoise"],
        "source": {
            "kind": "git",
            "repository": "https://github.com/Even-Ma/xas.git",
            "commit": "22363b96cd1e797f7b34d39948f50f4d4b899a2c",
        },
        "license": "unknown",
        "distribution": "private-validation-only",
        "weights": {
            "required": True,
            "state": "required-missing",
            "allow_download": False,
        },
        "training": {"enabled": False},
    }


def test_loads_built_in_classical_baseline_with_local_revision() -> None:
    # Break caught: a built-in baseline could be registered without reproducible local source metadata.
    tool = load_tool_manifest(ROOT / "tools/xas/savgol/tool.yaml")

    assert tool.id == "savgol"
    assert tool.source.kind == "local-package"
    assert tool.source.package == "hyperspectrum"
    assert tool.source.version == "0.1.0"
    assert tool.source.revision
    assert tool.weights.state == "not-required"
    assert tool.training.enabled is False


def test_loads_pinned_external_adapter_without_source_or_weights() -> None:
    # Break caught: the adapter declaration may falsely claim vendored external source or weights.
    tool = load_tool_manifest(ROOT / "tools/xas/xasdenoise/tool.yaml")

    assert tool.source.commit == "22363b96cd1e797f7b34d39948f50f4d4b899a2c"
    assert tool.license == "unknown"
    assert tool.distribution == "private-validation-only"
    assert tool.weights.state == "required-missing"
    assert tool.weights.allow_download is False


def test_rejects_mutable_runtime_image() -> None:
    # Break caught: changing an image from a digest to a mutable tag would make runs irreproducible.
    data = manifest()
    data["runtime"] = {"kind": "python", "image": "registry.example/xasdenoise:latest"}

    with pytest.raises(ValidationError, match="immutable digest"):
        load_tool_manifest(data)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"kind": "container"}, "container runtime requires an image"),
        (
            {
                "kind": "git",
                "repository": "https://github.com/Even-Ma/xas.git",
                "release": "latest",
            },
            "immutable release",
        ),
        (
            {
                "kind": "local-package",
                "package": "hyperspectrum",
                "version": "0.1.0",
                "revision": "main",
            },
            "immutable revision",
        ),
    ],
)
def test_schema_and_model_reject_mutable_runtime_or_source_pin(
    replacement: dict[str, str], message: str
) -> None:
    # Break caught: a container or source could resolve differently on a later run.
    data = manifest()
    if "repository" in replacement or replacement["kind"] == "local-package":
        data["source"] = replacement
    else:
        data["runtime"] = replacement
    schema = json.loads((ROOT / "schemas/hyperspectrum-tool-v1.schema.json").read_text())

    with pytest.raises(ValidationError, match=message):
        load_tool_manifest(data)
    assert not Draft202012Validator(schema).is_valid(data)


def test_rejects_git_source_without_commit_or_release() -> None:
    # Break caught: an external adapter could track an unpinned source revision.
    data = manifest()
    data["source"] = {"kind": "git", "repository": "https://github.com/Even-Ma/xas.git"}

    with pytest.raises(ValidationError, match="commit or release"):
        load_tool_manifest(data)


def test_schema_and_model_require_digest_for_declared_present_weights() -> None:
    # Break caught: a claimed present checkpoint could be selected without integrity evidence.
    data = manifest()
    data["weights"] = {"required": True, "state": "present", "allow_download": False}
    schema = json.loads((ROOT / "schemas/hyperspectrum-tool-v1.schema.json").read_text())

    with pytest.raises(ValidationError, match="weight digest"):
        load_tool_manifest(data)
    assert not Draft202012Validator(schema).is_valid(data)


def test_rejects_unknown_output_role() -> None:
    # Break caught: a misspelled output role could be silently treated as task-compatible.
    data = manifest()
    data["outputs"] = [{"role": "denoised_singal", "kind": "dense_array"}]

    with pytest.raises(ValidationError, match="artifact role"):
        load_tool_manifest(data)


def test_published_schema_rejects_duplicate_artifact_roles() -> None:
    # Break caught: schema-only consumers could accept an ambiguous role that the runtime rejects.
    data = manifest()
    data["outputs"] = [
        {"role": "denoised_signal", "kind": "dense_array"},
        {"role": "denoised_signal", "kind": "dense_array"},
    ]
    schema = json.loads((ROOT / "schemas/hyperspectrum-tool-v1.schema.json").read_text())

    assert not Draft202012Validator(schema).is_valid(data)


def test_rejects_empty_verify_argv() -> None:
    # Break caught: a tool could be marked verified without an executable verification command.
    data = manifest()
    data["verify"] = []

    with pytest.raises(ValidationError, match="verify"):
        load_tool_manifest(data)


def test_rejects_shell_verify_string() -> None:
    # Break caught: shell parsing could make verification behavior platform-dependent.
    data = manifest()
    data["verify"] = "python -m hyperspectrum.verify_xasdenoise"

    with pytest.raises(ValidationError, match="verify"):
        load_tool_manifest(data)


def test_rejects_missing_license_state() -> None:
    # Break caught: a tool lacking a license state could bypass distribution policy.
    data = manifest()
    del data["license"]

    with pytest.raises(ValidationError, match="license"):
        load_tool_manifest(data)


@pytest.mark.parametrize(
    ("section", "replacement", "message"),
    [
        ("weights", {"required": True, "state": "required-missing", "allow_download": True}, "download"),
        ("training", {"enabled": True}, "training"),
    ],
)
def test_rejects_manifest_that_enables_unsafe_default(
    section: str, replacement: object, message: str
) -> None:
    # Break caught: an agent could trigger unapproved downloads or training merely by selecting a tool.
    data = manifest()
    data[section] = replacement

    with pytest.raises(ValidationError, match=message):
        load_tool_manifest(data)


def test_digest_depends_on_canonical_manifest_semantics_not_yaml_key_order() -> None:
    # Break caught: formatting-only YAML edits could change provenance digests.
    first = load_tool_manifest(manifest())
    reordered = deepcopy(manifest())
    reordered["tasks"] = ["denoising"]
    second = load_tool_manifest(dict(reversed(list(reordered.items()))))

    assert first.tool_digest == second.tool_digest
    assert len(first.tool_digest) == 64


def test_match_fails_closed_on_every_required_gate() -> None:
    # Break caught: omitting any one compatibility gate could select an unsafe or unusable tool.
    registry = ToolRegistry(
        (load_tool_manifest(manifest()),),
        availability_resolver=lambda _: ToolAvailability(available=True),
    )
    matching = ToolMatchRequest(
        modality="xas",
        task="denoising",
        input_roles=("raw_signal",),
        output_roles=("denoised_signal",),
        license_policy=LicensePolicy.private_validation(),
        weights_state="required-missing",
        resources=ResourceBudget(cpu=4, memory_gb=16, gpu_available=False),
    )

    assert registry.match(matching) == (registry.tools[0],)

    for mismatch in (
        matching.model_copy(update={"modality": "raman"}),
        matching.model_copy(update={"task": "classification"}),
        matching.model_copy(update={"input_roles": ("spectrum",)}),
        matching.model_copy(update={"output_roles": ("prediction",)}),
        matching.model_copy(update={"license_policy": LicensePolicy.open_distribution()}),
        matching.model_copy(update={"weights_state": "present"}),
        matching.model_copy(
            update={"resources": ResourceBudget(cpu=2, memory_gb=16, gpu_available=False)}
        ),
    ):
        assert registry.match(mismatch) == ()


def test_default_match_excludes_tool_with_unresolvable_entrypoint_verify_or_weights() -> None:
    # Break caught: selection could return a manifest that cannot safely be executed.
    tool = load_tool_manifest(manifest())
    registry = ToolRegistry((tool,))
    request = ToolMatchRequest(
        modality="xas",
        task="denoising",
        input_roles=("raw_signal",),
        output_roles=("denoised_signal",),
        license_policy=LicensePolicy.private_validation(),
        weights_state="required-missing",
        resources=ResourceBudget(cpu=4, memory_gb=16, gpu_available=False),
    )

    assert registry.match(request) == ()
    availability = registry.availability(tool)
    assert availability.available is False
    assert {
        "entrypoint-unresolvable",
        "verify-module-unresolvable",
        "weights-required-missing",
    }.issubset(availability.reasons)


def test_injected_availability_resolver_allows_deterministic_executable_match() -> None:
    # Break caught: tests could depend on importing or executing an external adapter to select it.
    tool = load_tool_manifest(manifest())
    registry = ToolRegistry(
        (tool,), availability_resolver=lambda _: ToolAvailability(available=True)
    )
    request = ToolMatchRequest(
        modality="xas",
        task="denoising",
        input_roles=("raw_signal",),
        output_roles=("denoised_signal",),
        license_policy=LicensePolicy.private_validation(),
        weights_state="required-missing",
        resources=ResourceBudget(cpu=4, memory_gb=16, gpu_available=False),
    )

    assert registry.match(request) == (tool,)


def test_registered_tools_are_discoverable_but_not_default_executable_until_available() -> None:
    # Break caught: an unavailable registered adapter or baseline could be returned for execution.
    savgol = load_tool_manifest(ROOT / "tools/xas/savgol/tool.yaml")
    xasdenoise = load_tool_manifest(ROOT / "tools/xas/xasdenoise/tool.yaml")
    registry = ToolRegistry((savgol, xasdenoise))
    request = ToolMatchRequest(
        modality="xas",
        task="denoising",
        input_roles=("raw_signal",),
        output_roles=("denoised_signal",),
        license_policy=LicensePolicy.private_validation(),
        weights_state="not-required",
        resources=ResourceBudget(cpu=4, memory_gb=16, gpu_available=False),
    )

    assert registry.tools == (savgol, xasdenoise)
    assert registry.match(request) == ()
    assert "entrypoint-unresolvable" in registry.availability(savgol).reasons
    assert "entrypoint-unresolvable" in registry.availability(xasdenoise).reasons
