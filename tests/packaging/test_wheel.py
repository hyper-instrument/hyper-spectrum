"""Non-editable wheel and installed-console acceptance tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def command(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.fixture(scope="module")
def installed_console(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, str], Path]:
    root = tmp_path_factory.mktemp("installed-wheel")
    dist = root / "dist"
    clean_environment = dict(os.environ)
    clean_environment.pop("PYTHONPATH", None)
    built = command(
        "uv",
        "build",
        "--wheel",
        "--offline",
        "--no-cache",
        "--out-dir",
        str(dist),
        env=clean_environment,
    )
    assert built.returncode == 0, built.stderr
    wheel = next(dist.glob("hyperspectrum-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "hyperspectrum/resources/schemas/hyperspectrum-tool-v1.schema.json" in names
    assert "hyperspectrum/resources/tools/xas/savgol/tool.yaml" in names
    assert "hyperspectrum/resources/tools/xas/xasdenoise/tool.yaml" in names

    environment_dir = root / "venv"
    created = command(
        sys.executable,
        "-m",
        "venv",
        "--without-pip",
        str(environment_dir),
        env=clean_environment,
    )
    assert created.returncode == 0, created.stderr
    python = environment_dir / "bin/python"
    purelib = command(
        str(python),
        "-c",
        "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        env=clean_environment,
    )
    assert purelib.returncode == 0, purelib.stderr
    dependency_path = Path(purelib.stdout.strip()) / "local-dependencies.pth"
    dependency_path.write_text(
        sysconfig.get_paths()["purelib"] + "\n", encoding="utf-8"
    )
    installed = command(
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--no-deps",
        "--offline",
        "--no-cache",
        str(wheel),
        env=clean_environment,
    )
    assert installed.returncode == 0, installed.stderr
    located = command(
        str(python),
        "-c",
        "import hyperspectrum; print(hyperspectrum.__file__)",
        env=clean_environment,
    )
    assert located.returncode == 0, located.stderr
    installed_package = Path(located.stdout.strip()).resolve()
    assert installed_package.is_relative_to(Path(purelib.stdout.strip()).resolve())

    fake_bin = root / "bin"
    fake_bin.mkdir()
    hyd = fake_bin / "hyd"
    hyd.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hyd.chmod(0o755)
    environment = clean_environment
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    console = environment_dir / "bin/hyperspectrum"
    return console, environment, root


def assert_success_envelope(
    completed: subprocess.CompletedProcess[str],
) -> dict[str, object]:
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("\n") == 1
    envelope = json.loads(completed.stdout)
    assert set(envelope) == {"schema_version", "ok", "result", "warnings", "error"}
    assert envelope["ok"] is True
    return envelope


def assert_error_envelope(
    completed: subprocess.CompletedProcess[str], exit_code: int, code: str
) -> dict[str, object]:
    assert completed.returncode == exit_code, completed.stderr
    assert completed.stdout.count("\n") == 1
    envelope = json.loads(completed.stdout)
    assert set(envelope) == {"schema_version", "ok", "result", "warnings", "error"}
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == code
    assert "Traceback" not in completed.stderr
    return envelope


def write_plan_contracts(root: Path) -> tuple[Path, Path, Path]:
    candidate = root / "candidate.json"
    task = root / "task.json"
    verdict = root / "verdict.json"
    digest = "a" * 64
    candidate.write_text(
        json.dumps(
            {
                "dataset_code": "XAS-WHEEL",
                "dataset_version": "v1",
                "content_digest": digest,
                "title": "Wheel fixture",
                "description": "",
                "file_count": 1,
                "parsed_file_count": 1,
                "formats": ["npz"],
                "license": "test-only",
                "evidence": {},
            }
        ),
        encoding="utf-8",
    )
    task.write_text(
        json.dumps(
            {
                "schema_version": "hyperspectrum-task/v1",
                "id": "xas-denoising",
                "modality": "xas",
                "task_type": "denoising",
                "input_roles": ["raw_signal"],
                "output_kind": "dense_array",
                "ground_truth_roles": ["clean_spectrum"],
                "split_group_keys": ["sample_id", "compound_id"],
                "metrics": [],
            }
        ),
        encoding="utf-8",
    )
    verdict.write_text(
        json.dumps(
            {
                "dataset_code": "XAS-WHEEL",
                "dataset_version": "v1",
                "content_digest": digest,
                "status": "scoreable",
                "reasons": ["verified"],
                "candidate_tasks": ["denoising"],
                "ground_truth_roles": ["clean_spectrum"],
                "split_group_keys": ["sample_id", "compound_id"],
            }
        ),
        encoding="utf-8",
    )
    return task, candidate, verdict


def plan_command(
    console: Path,
    root: Path,
    task: Path,
    candidate: Path,
    verdict: Path,
    tool_id: str,
) -> tuple[str, ...]:
    return (
        str(console),
        "run",
        "plan",
        "--task-file",
        str(task),
        "--candidate-file",
        str(candidate),
        "--verdict-file",
        str(verdict),
        "--tool-id",
        tool_id,
        "--output-directory",
        str(root / f"run-{tool_id}"),
        "--sample-id",
        "sample-1",
        "--dry-run",
        "--json",
    )


def test_root_and_packaged_resources_have_exact_byte_parity() -> None:
    from importlib.resources import files

    resources = files("hyperspectrum.resources")
    pairs = (
        (
            ROOT / "schemas/hyperspectrum-tool-v1.schema.json",
            resources.joinpath("schemas/hyperspectrum-tool-v1.schema.json"),
        ),
        (
            ROOT / "tools/xas/savgol/tool.yaml",
            resources.joinpath("tools/xas/savgol/tool.yaml"),
        ),
        (
            ROOT / "tools/xas/xasdenoise/tool.yaml",
            resources.joinpath("tools/xas/xasdenoise/tool.yaml"),
        ),
    )
    for root_file, packaged in pairs:
        assert root_file.read_bytes() == packaged.read_bytes()


def test_fresh_wheel_console_help_doctor_match_and_plan(
    installed_console: tuple[Path, dict[str, str], Path],
) -> None:
    console, environment, root = installed_console
    help_result = command(str(console), "--help", env=environment)
    assert help_result.returncode == 0, help_result.stderr
    assert "Agent-ready spectroscopy runtime" in help_result.stdout
    assert_success_envelope(command(str(console), "doctor", "--json", env=environment))
    matched = assert_success_envelope(
        command(
            str(console),
            "tools",
            "match",
            "--task",
            "xas-denoising",
            "--json",
            env=environment,
        )
    )
    assert matched["result"]["matches"][0]["id"] == "savgol"  # type: ignore[index]

    task, candidate, verdict = write_plan_contracts(root)
    planned = command(
        *plan_command(console, root, task, candidate, verdict, "savgol"),
        env=environment,
    )
    envelope = assert_success_envelope(planned)
    assert envelope["result"]["plan"]["tool_id"] == "savgol"  # type: ignore[index]


def test_installed_console_exit_categories_and_secret_probes(
    installed_console: tuple[Path, dict[str, str], Path],
) -> None:
    console, environment, root = installed_console
    assert_error_envelope(
        command(str(console), "doctor", "--json", "--bogus", env=environment),
        2,
        "invalid_or_not_ready",
    )
    assert_error_envelope(
        command(str(console), "data", "discover", "--json", env=environment),
        2,
        "invalid_or_not_ready",
    )
    assert_error_envelope(
        command(
            str(console),
            "task",
            "recommend",
            "--candidate-file",
            str(root / "missing.json"),
            "--json",
            env=environment,
        ),
        4,
        "missing_asset_or_tool",
    )
    task, candidate, verdict = write_plan_contracts(root)
    constrained = (
        *plan_command(console, root, task, candidate, verdict, "savgol")[:-1],
        "--max-samples",
        "0",
        "--json",
    )
    assert_error_envelope(
        command(*constrained, env=environment),
        2,
        "invalid_or_not_ready",
    )
    secret_constrained = (
        *plan_command(console, root, task, candidate, verdict, "savgol")[:-1],
        "--max-samples",
        "api_key=INSTALLED_PARSE_SECRET",
        "--json",
    )
    secret_parse = command(*secret_constrained, env=environment)
    assert_error_envelope(secret_parse, 2, "invalid_or_not_ready")
    assert "INSTALLED_PARSE_SECRET" not in secret_parse.stdout + secret_parse.stderr
    assert_error_envelope(
        command(
            *plan_command(console, root, task, candidate, verdict, "unknown-tool"),
            env=environment,
        ),
        4,
        "missing_asset_or_tool",
    )
    assert_error_envelope(
        command(
            *plan_command(console, root, task, candidate, verdict, "xasdenoise"),
            env=environment,
        ),
        4,
        "missing_asset_or_tool",
    )
    dry_plan = assert_success_envelope(
        command(
            *plan_command(console, root, task, candidate, verdict, "savgol"),
            env=environment,
        )
    )["result"]["plan"]  # type: ignore[index]
    plan_file = root / "dry-plan.json"
    plan_file.write_text(json.dumps(dry_plan), encoding="utf-8")
    source = root / "existing.npz"
    source.write_bytes(b"not-read-for-dry-run")
    assert_error_envelope(
        command(
            str(console),
            "run",
            "local",
            "--plan-file",
            str(plan_file),
            "--source-npz",
            str(source),
            "--sample-id",
            "sample-1",
            "--json",
            env=environment,
        ),
        5,
        "execution_failure",
    )

    raw = json.loads(candidate.read_text(encoding="utf-8"))
    raw["extra_secret"] = {
        "authorization": "Bearer INSTALLED_INVALID_BEARER",
        "token": "INSTALLED_INVALID_TOKEN",
        "api-key": "INSTALLED_INVALID_APIKEY",
        "password": "INSTALLED_INVALID_PASSWORD",
        "url": "https://example.invalid/?signature=INSTALLED_INVALID_QUERY",
    }
    candidate.write_text(json.dumps(raw), encoding="utf-8")
    invalid = command(
        str(console),
        "task",
        "recommend",
        "--candidate-file",
        str(candidate),
        "--json",
        env=environment,
    )
    assert_error_envelope(invalid, 2, "invalid_or_not_ready")
    assert all(
        secret not in invalid.stdout + invalid.stderr
        for secret in (
            "INSTALLED_INVALID_BEARER",
            "INSTALLED_INVALID_TOKEN",
            "INSTALLED_INVALID_APIKEY",
            "INSTALLED_INVALID_PASSWORD",
            "INSTALLED_INVALID_QUERY",
        )
    )

    raw_without_mapping = json.loads(candidate.read_text(encoding="utf-8"))
    raw_without_mapping.pop("extra_secret")
    raw_without_mapping["evidence"] = []
    candidate.write_text(json.dumps(raw_without_mapping), encoding="utf-8")
    invalid_evidence = command(
        str(console),
        "task",
        "recommend",
        "--candidate-file",
        str(candidate),
        "--json",
        env=environment,
    )
    assert_error_envelope(invalid_evidence, 2, "invalid_or_not_ready")

    raw.pop("extra_secret")
    raw["evidence"] = {
        "safe_context": "installed-keep-me",
        "authorization": "Bearer INSTALLED_BEARER_LITERAL",
        "token": "INSTALLED_TOKEN_LITERAL",
        "nested": {
            "api-key": "INSTALLED_APIKEY_LITERAL",
            "password": "INSTALLED_PASSWORD_LITERAL",
            "accessToken": "INSTALLED_ACCESS_TOKEN_LITERAL",
            "refreshToken": "INSTALLED_REFRESH_TOKEN_LITERAL",
            "landing_page": (
                "https://example.invalid/data?signature=INSTALLED_QUERY_LITERAL"
                "&public=yes"
            ),
            "oauth_url": (
                "https://example.invalid/data?"
                "client_secret=INSTALLED_CLIENT_SECRET_LITERAL"
            ),
            "userinfo_url": (
                "https://safe-user:INSTALLED_USERINFO_LITERAL@example.invalid/data"
            ),
        },
    }
    candidate.write_text(json.dumps(raw), encoding="utf-8")
    valid = command(
        str(console),
        "task",
        "recommend",
        "--candidate-file",
        str(candidate),
        "--json",
        env=environment,
    )
    valid_envelope = assert_success_envelope(valid)
    rendered = valid.stdout + valid.stderr
    assert "installed-keep-me" in json.dumps(valid_envelope)
    assert all(
        secret not in rendered
        for secret in (
            "INSTALLED_BEARER_LITERAL",
            "INSTALLED_TOKEN_LITERAL",
            "INSTALLED_APIKEY_LITERAL",
            "INSTALLED_PASSWORD_LITERAL",
            "INSTALLED_QUERY_LITERAL",
            "INSTALLED_ACCESS_TOKEN_LITERAL",
            "INSTALLED_REFRESH_TOKEN_LITERAL",
            "INSTALLED_CLIENT_SECRET_LITERAL",
            "INSTALLED_USERINFO_LITERAL",
        )
    )


@pytest.mark.parametrize(
    ("script", "exit_code", "code"),
    [
        (
            "#!/bin/sh\necho 'authentication required' >&2\nexit 1\n",
            3,
            "auth_or_connection",
        ),
        ("invalid executable format\n", 3, "auth_or_connection"),
        (
            "#!/bin/sh\necho 'unknown command' >&2\nexit 127\n",
            4,
            "missing_asset_or_tool",
        ),
        ("#!/bin/sh\necho 'not-json'\nexit 0\n", 4, "missing_asset_or_tool"),
        (
            "#!/bin/sh\necho 'deterministic failure' >&2\nexit 1\n",
            5,
            "execution_failure",
        ),
    ],
)
def test_installed_discovery_gateway_faults(
    installed_console: tuple[Path, dict[str, str], Path],
    script: str,
    exit_code: int,
    code: str,
) -> None:
    console, environment, root = installed_console
    hyd = root / "bin/hyd"
    hyd.write_text(script, encoding="utf-8")
    hyd.chmod(0o755)
    isolated_environment = dict(environment)
    isolated_environment["PATH"] = str(root / "bin")

    completed = command(
        str(console),
        "data",
        "discover",
        "--modality",
        "xas",
        "--profile",
        "volcano",
        "--json",
        env=isolated_environment,
    )

    assert_error_envelope(completed, exit_code, code)


def test_installed_discovery_missing_client_is_exit_four(
    installed_console: tuple[Path, dict[str, str], Path],
) -> None:
    console, environment, root = installed_console
    empty_bin = root / "empty-bin"
    empty_bin.mkdir(exist_ok=True)
    isolated_environment = dict(environment)
    isolated_environment["PATH"] = str(empty_bin)

    completed = command(
        str(console),
        "data",
        "discover",
        "--modality",
        "xas",
        "--profile",
        "volcano",
        "--json",
        env=isolated_environment,
    )

    assert_error_envelope(completed, 4, "missing_asset_or_tool")


def test_installed_generic_failure_redacts_quoted_diagnostics(
    installed_console: tuple[Path, dict[str, str], Path],
) -> None:
    console, environment, root = installed_console
    hyd = root / "bin/hyd"
    hyd.write_text(
        (
            "#!/bin/sh\n"
            "printf '%s\\n' "
            '\'{"clientSecret":"INSTALLED_QUOTED_CAMEL",'
            '"nested":{"refresh_token":"INSTALLED_NESTED_OAUTH"},'
            '"authorization":"Bearer INSTALLED_QUOTED_BEARER",'
            '"context":"keep-installed-json"}\' >&2\n'
            "printf '%s\\n' "
            "\"{'client_secret': 'INSTALLED_PY_REPR', "
            "'accessToken': 'INSTALLED_ESCAPED', "
            "'context': 'keep-installed-repr'}\" >&2\n"
            "printf '%s\\n' "
            "'https://INSTALLED_USERINFO@example.invalid/data?"
            "client_secret=INSTALLED_QUERY_OAUTH' >&2\n"
            "printf '%s\\n' "
            '\'malformed {"clientSecret":"INSTALLED_MALFORMED"\' >&2\n'
            "exit 1\n"
        ),
        encoding="utf-8",
    )
    hyd.chmod(0o755)
    isolated_environment = dict(environment)
    isolated_environment["PATH"] = str(root / "bin")

    completed = command(
        str(console),
        "data",
        "discover",
        "--modality",
        "xas",
        "--profile",
        "volcano",
        "--json",
        env=isolated_environment,
    )

    envelope = assert_error_envelope(completed, 5, "execution_failure")
    rendered = completed.stdout + completed.stderr
    secrets = (
        "INSTALLED_QUOTED_CAMEL",
        "INSTALLED_NESTED_OAUTH",
        "INSTALLED_QUOTED_BEARER",
        "INSTALLED_PY_REPR",
        "INSTALLED_ESCAPED",
        "INSTALLED_USERINFO",
        "INSTALLED_QUERY_OAUTH",
        "INSTALLED_MALFORMED",
    )
    leaked = [secret for secret in secrets if secret in rendered]
    assert leaked == [], rendered
    for safe_context in ("keep-installed-json", "keep-installed-repr"):
        assert safe_context in completed.stdout
        assert safe_context in completed.stderr
        assert safe_context in envelope["error"]["message"]  # type: ignore[index]


@pytest.mark.parametrize(
    "command_path",
    [
        ("doctor",),
        ("data", "discover"),
        ("task", "recommend"),
        ("tools", "match"),
        ("run", "plan"),
        ("run", "local"),
    ],
)
@pytest.mark.parametrize("agent_options", [("--json", "--help"), ("--help", "--json")])
def test_installed_json_help_is_one_success_envelope(
    installed_console: tuple[Path, dict[str, str], Path],
    command_path: tuple[str, ...],
    agent_options: tuple[str, ...],
) -> None:
    console, environment, _ = installed_console

    completed = command(str(console), *command_path, *agent_options, env=environment)

    envelope = assert_success_envelope(completed)
    assert completed.stderr == ""
    assert isinstance(envelope["result"], dict)
    assert "Usage:" in envelope["result"]["help"]  # type: ignore[index]


def test_installed_non_json_help_remains_human_readable(
    installed_console: tuple[Path, dict[str, str], Path],
) -> None:
    console, environment, _ = installed_console

    completed = command(str(console), "doctor", "--help", env=environment)

    assert completed.returncode == 0
    assert completed.stdout.count("\n") > 1
    assert "Usage:" in completed.stdout
