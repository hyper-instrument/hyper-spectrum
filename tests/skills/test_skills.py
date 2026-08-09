"""Static contracts for reusable XAS agent skills."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def skill_text(name: str) -> str:
    return (ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")


def test_skill_frontmatter_and_openai_interface_contracts() -> None:
    for name in ("discover-xas-dataset", "run-xas-baseline"):
        text = skill_text(name)
        match = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
        assert match is not None
        frontmatter = yaml.safe_load(match.group(1))
        assert set(frontmatter) == {"name", "description"}
        assert frontmatter["name"] == name
        assert frontmatter["description"].startswith("Use when")

        interface_path = ROOT / "skills" / name / "agents/openai.yaml"
        raw_interface = interface_path.read_text(encoding="utf-8")
        interface = yaml.safe_load(raw_interface)["interface"]
        assert set(interface) == {
            "display_name",
            "short_description",
            "default_prompt",
        }
        assert f"${name}" in interface["default_prompt"]
        assert all(
            re.match(r'^  [a-z_]+: ".*"$', line)
            for line in raw_interface.splitlines()[1:]
        )


def test_discovery_skill_is_read_only_evidence_led_and_fail_closed() -> None:
    text = " ".join(skill_text("discover-xas-dataset").casefold().split())

    for required in (
        "hyperspectrum doctor --json",
        "hyd --profile volcano whoami",
        "server-confirmed identity",
        "cached or unverified output cannot pass",
        "data discover --modality xas --profile volcano --json",
        "read-only",
        "credential/profile files",
        "single stdout envelope",
        "ground-truth evidence",
        "authentication/connection failure",
        "do not download",
        "does not download data, score models, or own a leaderboard",
    ):
        assert required in text


def test_run_skill_enforces_dry_run_leakage_overwrite_and_authority_gates() -> None:
    text = " ".join(skill_text("run-xas-baseline").casefold().split())

    for required in (
        "hyperspectrum doctor --json",
        "task recommend --candidate-file",
        "tools match --task xas-denoising --json",
        "run plan",
        "--benchmark-manifest-file",
        "three separate sha-256 classes",
        "bind every predeclared sample id during planning",
        "--dry-run",
        "does not exist",
        "never read clean",
        "stop after the dry-run",
        "external/remote execution",
        "adapter or tool adaptation",
        "download",
        "installation",
        "training/fine-tuning",
        "publication",
        "hyperspectrum run local",
        "exactly `energy`, `noisy`, `sample_ids`, `group_ids`, and `energy_unit`",
        "canonical xanes benchmark npz",
        "pseudo-clean frozen measurement",
        "never physical noiseless ground truth",
        "reject every unrecognized extra member",
        "ace owns formal",
        "zenodo-17434349",
        "cc-by-4.0 weights",
        "--weight-file model.pth",
        "identity_raw",
        "hyperspectrum-xasdenoise-step-baseline/v1",
        "same `hyperspectrum.adapters.xasdenoise:denoise_spectra` entrypoint",
    ):
        assert required in text
