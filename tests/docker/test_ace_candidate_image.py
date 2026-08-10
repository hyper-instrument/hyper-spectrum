"""The candidate image recipe, checked on a machine that cannot build it.

`docker/ace-candidate/Dockerfile` is this repository's copy of a recipe ACE
already has, with exactly one change: the scientific runtime is installed from
*this checkout* instead of from a tarball staged out of it. Every other
property is a contract with the platform on the other side — the entrypoint
ACE invokes, the mounts it fills, the one environment variable it is entitled
to send, and the dependency versions the acceptance probe refuses to run
without.

None of that can be checked by building here: there is no Docker on the
machines this suite runs on, and a build that reached PyPI would not be a test
anyway. What *can* be checked is that the recipe still says what it has to say,
and — the part worth having — that the constants the probe will hold the built
image to are still true of the source tree it will be built from. A drift in
`src/hyperspectrum/` is then a red test here, rather than a probe failure
twenty minutes into a remote build.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_DIR = ROOT / "docker" / "ace-candidate"
DOCKERFILE = CANDIDATE_DIR / "Dockerfile"
VENDOR_DIR = CANDIDATE_DIR / "vendor"
REVISION_FILE = CANDIDATE_DIR / "hyperspectrum-sdk.rev"
DOCKERIGNORE = ROOT / ".dockerignore"

#: The interpreter ACE's research base pins, by digest rather than by tag.
#: Restated here rather than imported, because the two repositories never import
#: each other and this is the value that has to agree.
PINNED_INTERPRETER = (
    "python:3.11-slim@sha256:"
    "90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff"
)
#: The image's `python_full_version`, which is what selects a row out of the
#: lock's per-interpreter resolution.
IMAGE_PYTHON = "3.11"


def instructions() -> list[tuple[str, str]]:
    """The Dockerfile as (instruction, argument) pairs, continuations joined."""

    text = DOCKERFILE.read_text(encoding="utf-8")
    joined = re.sub(r"\\\n\s*", " ", text)
    parsed: list[tuple[str, str]] = []
    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        head, _, tail = stripped.partition(" ")
        parsed.append((head.upper(), tail.strip()))
    return parsed


def arguments_of(instruction: str) -> list[str]:
    return [argument for head, argument in instructions() if head == instruction]


def copies() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for argument in arguments_of("COPY"):
        parts = [part for part in argument.split() if not part.startswith("--")]
        assert len(parts) == 2, argument
        pairs.append((parts[0], parts[1]))
    return pairs


def vendored_digests() -> dict[str, dict[str, object]]:
    document = json.loads(
        (CANDIDATE_DIR / "vendored.sha256.json").read_text(encoding="utf-8")
    )
    files = document["files"]
    assert isinstance(files, dict)
    return files


def probe_constant(name: str) -> str:
    """One pinned constant out of the vendored acceptance probe.

    Read from the file rather than imported: `verify.py` imports the runtime at
    module scope in a way that only makes sense inside the built image, and the
    constants are the whole of what this suite needs from it.
    """

    source = (VENDOR_DIR / "verify.py").read_text(encoding="utf-8")
    match = re.search(rf'^{name} = "([^"]+)"$', source, flags=re.MULTILINE)
    assert match is not None, name
    return match.group(1)


def locked_version(package: str) -> tuple[str, list[str]]:
    """The version of `package` an interpreter at IMAGE_PYTHON resolves to."""

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    selected = [
        entry
        for entry in lock["package"]
        if entry["name"] == package
        and any(
            f"python_full_version == '{IMAGE_PYTHON}.*'" in marker
            for marker in entry.get("resolution-markers", [])
        )
    ]
    assert len(selected) == 1, f"{package}: {len(selected)} rows for {IMAGE_PYTHON}"
    entry = selected[0]
    hashes = [wheel["hash"] for wheel in entry.get("wheels", [])]
    return entry["version"], hashes


# -- what the platform on the other side is entitled to ----------------------


def test_the_image_starts_the_adapter_ace_invokes() -> None:
    """The entrypoint is the contract. A shell wrapper here would be a new one."""

    assert arguments_of("ENTRYPOINT") == ['["python", "/opt/acebench/run.py"]']


def test_the_adapter_and_its_two_siblings_land_where_the_image_imports_them() -> None:
    """`run.py` does `from answer import ...` and `from entry import ...`.

    `answer` resolves off the script's own directory and `entry` off
    `PYTHONPATH`; both point at `/opt/acebench`, so all three files have to be
    there under exactly those names.
    """

    destinations = dict(copies())

    assert destinations["docker/ace-candidate/vendor/run.py"] == "/opt/acebench/run.py"
    assert (
        destinations["docker/ace-candidate/vendor/answer.py"]
        == "/opt/acebench/answer.py"
    )
    assert (
        destinations["docker/ace-candidate/vendor/entry.py"] == "/opt/acebench/entry.py"
    )
    assert "ENV" in {head for head, _ in instructions()}
    assert any("PYTHONPATH=/opt/acebench" in argument for argument in arguments_of("ENV"))


def test_the_probe_is_vendored_and_deliberately_not_installed() -> None:
    """`verify.py` is fed to a built image from outside; copying it in would put
    a file in the recipe that the running program never reads."""

    assert (VENDOR_DIR / "verify.py").is_file()
    assert all(source != "docker/ace-candidate/vendor/verify.py" for source, _ in copies())


def test_every_copied_adapter_file_is_one_the_vendor_manifest_pins() -> None:
    """An adapter file copied from outside `vendor/` is unpinned code."""

    recorded = set(vendored_digests())
    for source, destination in copies():
        if not destination.startswith("/opt/acebench/"):
            continue
        if source.endswith(".rev"):
            continue
        assert source.startswith("docker/ace-candidate/vendor/"), source
        assert Path(source).name in recorded, source


def test_the_image_provides_the_mount_points_and_workdir_the_contract_names() -> None:
    """`/out` is a plain directory, not a mount: outputs are `docker cp`'d out
    after the container exits, and a missing one is a failure at the last step."""

    assert any(
        set(argument.split()) >= {"-p", "/out", "/data", "/weights", "/archive"}
        for argument in arguments_of("RUN")
    )
    assert arguments_of("WORKDIR") == ["/workspace"]


def test_the_interpreter_is_pinned_to_the_digest_aces_research_base_pins() -> None:
    """A tag would let the two repositories build against different bytes while
    both claiming `python:3.11-slim`, which is the failure mode a digit-identical
    claim cannot survive."""

    assert arguments_of("FROM") == [PINNED_INTERPRETER]


def test_the_image_is_unbuffered_and_writes_no_bytecode() -> None:
    environment = " ".join(arguments_of("ENV"))

    assert "PYTHONUNBUFFERED=1" in environment
    assert "PYTHONDONTWRITEBYTECODE=1" in environment


# -- the one change from ACE's recipe ----------------------------------------


def test_the_runtime_is_installed_from_this_repository_not_a_staged_tarball() -> None:
    """The whole point of this directory. ACE copies a tarball its own staging
    script produced; here the build context *is* the source, so a build service
    handed this repository's GitHub URL needs nothing else."""

    sources = {source for source, _ in copies()}
    assert "." in sources
    assert dict(copies())["."] == "/opt/hyperspectrum-sdk"
    assert not any(source.endswith((".tar.gz", ".sha256")) for source in sources)


def test_dependencies_are_installed_from_the_frozen_lock_with_hashes() -> None:
    """`--require-hashes` is the reason the lock is worth exporting at all.

    Without it the export is a version list and the image is whatever PyPI
    served that morning; with it, a dependency that changed under the same
    version number fails the build instead of the comparison.
    """

    runs = " ".join(arguments_of("RUN"))

    assert "uv==0.10.10" in runs
    assert "uv export --frozen --no-dev --no-emit-project" in runs
    assert "--require-hashes" in runs
    assert "pip install --no-cache-dir --no-deps /opt/hyperspectrum-sdk" in runs


def test_the_build_context_excludes_everything_the_pinned_export_excluded() -> None:
    """ACE installs a `git archive` of one commit: tracked files, nothing else.

    A `COPY .` without this reaches the same install with a different tree —
    a local `.venv`, a stale `dist/`, someone's `runs/` — under the same claim
    of being that commit.
    """

    ignored = {
        line.strip().rstrip("/")
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert {".git", ".venv", "dist", "runs", "__pycache__"} <= ignored


# -- what the acceptance probe will hold the built image to ------------------


def test_the_revision_the_image_declares_is_the_one_the_probe_admits() -> None:
    """`verify.py` reads `/opt/acebench/hyperspectrum-sdk.rev` and refuses any
    other value. The file and the constant can only ever move together."""

    destinations = dict(copies())
    assert (
        destinations["docker/ace-candidate/hyperspectrum-sdk.rev"]
        == "/opt/acebench/hyperspectrum-sdk.rev"
    )
    declared = REVISION_FILE.read_text(encoding="utf-8").strip()

    assert declared == probe_constant("PINNED_REVISION")


def test_the_runtime_digests_the_probe_admits_are_this_checkouts() -> None:
    """The guard that makes the pinned revision an honest claim rather than a
    string. If `src/hyperspectrum/` moves under a revision file that did not,
    the probe would fail inside a built image; this fails here instead.
    """

    from importlib.resources import files

    import yaml

    from hyperspectrum.execution.plan import resolve_local_entrypoint
    from hyperspectrum.registry.loader import load_tool_manifest

    manifest = files("hyperspectrum.resources").joinpath("tools/xas/savgol/tool.yaml")
    tool = load_tool_manifest(yaml.safe_load(manifest.read_text(encoding="utf-8")))
    resolved = resolve_local_entrypoint(tool)

    assert tool.id == "savgol"
    assert tool.tool_digest == probe_constant("EXPECTED_TOOL_DIGEST")
    assert resolved.implementation_digest == probe_constant(
        "EXPECTED_IMPLEMENTATION_DIGEST"
    )


def test_both_execution_facades_the_probe_requires_are_exported() -> None:
    """An image missing the inference-only facade passes every other check and
    then cannot read a candidate mount at all."""

    from hyperspectrum import execution

    assert callable(execution.execute_ace_xas_denoising)
    assert callable(execution.execute_ace_xas_inference_only)

    from hyperspectrum.datasets import load_cu_cha_denoising_pairs

    assert callable(load_cu_cha_denoising_pairs)


@pytest.mark.parametrize(
    ("package", "constant"),
    [("numpy", "EXPECTED_NUMPY_VERSION"), ("scipy", "EXPECTED_SCIPY_VERSION")],
)
def test_the_lock_resolves_the_numeric_stack_the_probe_pins(
    package: str, constant: str
) -> None:
    """The lock is the only thing standing between "pinned numpy" and a build
    that silently took a newer one, and the probe's constants are where the
    claim is written down. They are in different repositories; this is the seam.
    """

    version, hashes = locked_version(package)

    assert version == probe_constant(constant)
    assert hashes, f"{package} has no hashed wheels; --require-hashes would fail"
