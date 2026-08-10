"""Probe the exact HyperSpectrum/SavGol stack installed in the CPU image."""

from __future__ import annotations

import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

PINNED_REVISION = "c3063e27e1e27f83fbdfbb90dbe85f6c5a1bf4c8"
EXPECTED_TOOL_DIGEST = "d1ce4a5ad56800e0b72ad7a8f4255d232e0069c89e98fad7ba92beff1a71320a"
EXPECTED_IMPLEMENTATION_DIGEST = "b077ac6b4dd700e159cde8726a1557b1febb4300f9713b690b4381d735d1a316"
EXPECTED_NUMPY_VERSION = "2.4.6"
EXPECTED_SCIPY_VERSION = "1.17.1"
REVISION_PATH = Path("/opt/acebench/hyperspectrum-sdk.rev")


def validate_runtime_identity(
    *,
    revision: str,
    tool_id: str,
    tool_digest: str,
    implementation_digest: str,
    numpy_version: str,
    scipy_version: str,
    facade: Any,
    inference_facade: Any,
    profile_v2_loader: Any,
) -> dict[str, str]:
    """Fail closed when the image's scientific runtime drifts from its recipe."""

    if revision != PINNED_REVISION:
        raise RuntimeError(f"revision drift: expected {PINNED_REVISION}, got {revision}")
    if tool_id != "savgol":
        raise RuntimeError(f"tool drift: expected savgol, got {tool_id}")
    if tool_digest != EXPECTED_TOOL_DIGEST:
        raise RuntimeError(f"tool digest drift: expected {EXPECTED_TOOL_DIGEST}, got {tool_digest}")
    if implementation_digest != EXPECTED_IMPLEMENTATION_DIGEST:
        raise RuntimeError(
            "implementation digest drift: expected "
            f"{EXPECTED_IMPLEMENTATION_DIGEST}, got {implementation_digest}"
        )
    if numpy_version != EXPECTED_NUMPY_VERSION:
        raise RuntimeError(f"NumPy version drift: expected {EXPECTED_NUMPY_VERSION}, got {numpy_version}")
    if scipy_version != EXPECTED_SCIPY_VERSION:
        raise RuntimeError(f"SciPy version drift: expected {EXPECTED_SCIPY_VERSION}, got {scipy_version}")
    if not callable(facade):
        raise RuntimeError("facade is missing or not callable")
    # Probed beside the self-scored one because the image serves two lanes now.
    # An sdist staged without it builds an image that passes every other check
    # and then cannot read a candidate mount at all — which is the failure this
    # whole probe exists to move from run time to build time.
    if not callable(inference_facade):
        raise RuntimeError("inference-only facade is missing or not callable")
    if not callable(profile_v2_loader):
        raise RuntimeError("profile-v2 loader is missing or not callable")
    return {
        "revision": revision,
        "tool_digest": tool_digest,
        "implementation_digest": implementation_digest,
        "numpy_version": numpy_version,
        "scipy_version": scipy_version,
    }


def main() -> int:
    import numpy
    import scipy
    import yaml
    from hyperspectrum.datasets import load_cu_cha_denoising_pairs
    from hyperspectrum.execution import (
        execute_ace_xas_denoising,
        execute_ace_xas_inference_only,
    )
    from hyperspectrum.execution.plan import resolve_local_entrypoint
    from hyperspectrum.registry.loader import load_tool_manifest

    revision = REVISION_PATH.read_text(encoding="utf-8").strip()
    manifest_resource = files("hyperspectrum.resources").joinpath("tools/xas/savgol/tool.yaml")
    raw = yaml.safe_load(manifest_resource.read_text(encoding="utf-8"))
    tool = load_tool_manifest(raw)
    resolved = resolve_local_entrypoint(tool)
    identity = validate_runtime_identity(
        revision=revision,
        tool_id=tool.id,
        tool_digest=tool.tool_digest,
        implementation_digest=resolved.implementation_digest,
        numpy_version=numpy.__version__,
        scipy_version=scipy.__version__,
        facade=execute_ace_xas_denoising,
        inference_facade=execute_ace_xas_inference_only,
        profile_v2_loader=load_cu_cha_denoising_pairs,
    )
    print(
        " ".join(
            [
                f"hyperspectrum_revision={identity['revision']}",
                f"tool_digest={identity['tool_digest']}",
                f"implementation_digest={identity['implementation_digest']}",
                f"numpy={identity['numpy_version']}",
                f"scipy={identity['scipy_version']}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
