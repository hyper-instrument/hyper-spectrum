# HyperSpectrum XAS Agent M0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended for this session) or `superpowers:executing-plans` to execute this plan task by task. Use `superpowers:test-driven-development` for every implementation task and `superpowers:verification-before-completion` before claiming a milestone complete.

**Goal:** Build the first agent-ready HyperSpectrum runtime so an agent can discover real XAS datasets in the configured HyperData `volcano` profile, determine whether each dataset is scoreable, select a registered zero-shot or classical tool, produce an immutable run plan, and execute a local smoke run without inventing labels, weights, or results.

**Architecture:** Implement a small typed Python core around three immutable contracts (`ObservationBundle`, `TaskSpec`, `PredictionBundle`), a subprocess-only HyperData gateway, an evidence-led XAS discovery/readiness layer, a declarative tool registry, and a local executor. HyperData remains the data catalog, HyperSpectrum remains the domain runtime, and ACE Benchmark remains the formal evaluator and leaderboard publisher.

**Tech Stack:** Python 3.10+, Pydantic v2, Typer, NumPy, SciPy, PyYAML, JSON Schema, pytest, Ruff, mypy, `uv`, external `hyd` CLI.

## Global Constraints

- Never read `~/.hyperdata/credentials.json`, profile files, tokens, or environment secrets. Authentication checks use only `hyd whoami`, `hyd profile list`, and masked CLI output.
- Never treat `/usr/bin/hd` as the HyperData client on Linux; it is the util-linux hexdump binary. The default executable for this plan is `hyd` and it remains configurable.
- Never infer physical axes or units from array dimensionality. Every dense artifact must carry explicit axes and units.
- Never silently download model weights, modify a HyperData dataset, or start training. M0 is read-only discovery plus zero-shot/classical local smoke execution.
- Never admit an unlabeled or weakly evidenced dataset to a quantitative board. Return `inference_only` or `blocked` with machine-readable reasons.
- Never copy or redistribute code from `Even-Ma/xas` while its license remains unknown. Register it only as a pinned external source/tool.
- Run all Python commands from `/Users/duranze/Documents/hyperdata-codex/hyper-spectrum` using `.venv/bin/...` after `uv sync --all-extras --dev`.
- Keep generated data, weights, credentials, and raw remote command transcripts out of Git. Commit only redacted evidence summaries and content digests.

---

## File Map

Create these files in this repository:

```text
pyproject.toml
.gitignore
src/hyperspectrum/__init__.py
src/hyperspectrum/contracts/__init__.py
src/hyperspectrum/contracts/artifact.py
src/hyperspectrum/contracts/observation.py
src/hyperspectrum/contracts/task.py
src/hyperspectrum/contracts/prediction.py
src/hyperspectrum/hyperdata/__init__.py
src/hyperspectrum/hyperdata/gateway.py
src/hyperspectrum/hyperdata/models.py
src/hyperspectrum/hyperdata/discovery.py
src/hyperspectrum/tasks/__init__.py
src/hyperspectrum/tasks/recommend.py
src/hyperspectrum/registry/__init__.py
src/hyperspectrum/registry/models.py
src/hyperspectrum/registry/loader.py
src/hyperspectrum/plugins/__init__.py
src/hyperspectrum/plugins/xas/__init__.py
src/hyperspectrum/plugins/xas/arrays.py
src/hyperspectrum/plugins/xas/metrics.py
src/hyperspectrum/plugins/xas/baselines.py
src/hyperspectrum/execution/__init__.py
src/hyperspectrum/execution/plan.py
src/hyperspectrum/execution/local.py
src/hyperspectrum/cli.py
schemas/hyperspectrum-tool-v1.schema.json
tools/xas/savgol/tool.yaml
tools/xas/xasdenoise/tool.yaml
skills/discover-xas-dataset/SKILL.md
skills/run-xas-baseline/SKILL.md
tests/contracts/test_contracts.py
tests/hyperdata/test_gateway.py
tests/hyperdata/test_discovery.py
tests/tasks/test_recommend.py
tests/registry/test_loader.py
tests/plugins/xas/test_arrays.py
tests/plugins/xas/test_metrics.py
tests/plugins/xas/test_baselines.py
tests/execution/test_plan.py
tests/execution/test_local.py
tests/cli/test_cli.py
tests/fixtures/hyperdata/xas-search.json
tests/fixtures/xas/denoising-pairs.npz
docs/evidence/xas-m0-selection.schema.json
docs/dev/2026-08-08-xas-m0.md
```

The following generated paths are ignored:

```text
.venv/
runs/
dist/
*.egg-info/
```

## Public Interfaces

The implementation must expose these stable interfaces before the ACE plan starts:

```python
class AxisSpec(BaseModel):
    name: str
    unit: str
    direction: Literal["increasing", "decreasing", "unordered"]
    values_uri: str | None = None

class ArtifactRef(BaseModel):
    role: str
    kind: Literal["dense_array", "peak_table", "image", "mask", "region", "structure", "graph", "scalar", "class", "multilabel", "distribution", "metadata"]
    uri: str
    sha256: str | None = None
    axes: tuple[AxisSpec, ...] = ()

class ObservationBundle(BaseModel):
    schema_version: Literal["hyperspectrum-observation/v1"]
    sample_id: str
    modality: str
    artifacts: tuple[ArtifactRef, ...]
    context: dict[str, object]
    labels: dict[str, object]
    provenance: dict[str, object]

class MetricSpec(BaseModel):
    key: str
    direction: Literal["min", "max"]
    aggregation: Literal["sample_mean", "group_mean", "grouped_bootstrap"]
    primary: bool = False

class TaskSpec(BaseModel):
    schema_version: Literal["hyperspectrum-task/v1"]
    id: str
    modality: str
    task_type: str
    input_roles: tuple[str, ...]
    output_kind: str
    ground_truth_roles: tuple[str, ...]
    split_group_keys: tuple[str, ...]
    metrics: tuple[MetricSpec, ...]

class PredictionBundle(BaseModel):
    schema_version: Literal["hyperspectrum-prediction/v1"]
    run_id: str
    task_id: str
    predictions: tuple[ArtifactRef, ...]
    failures: tuple[dict[str, object], ...]
    provenance: dict[str, object]

class HydCommandResult(BaseModel):
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    payload: object | None

class HydGateway:
    def __init__(self, binary: str = "hyd", profile: str | None = None) -> None: ...
    def run(self, args: Sequence[str], *, expect_json: bool = True) -> HydCommandResult: ...
    def search(self, query: str) -> tuple["DatasetCandidate", ...]: ...

class DatasetCandidate(BaseModel):
    dataset_code: str
    dataset_version: str | None
    content_digest: str | None
    title: str
    description: str
    file_count: int
    parsed_file_count: int
    formats: tuple[str, ...]
    license: str | None
    evidence: dict[str, object]

class ReadinessVerdict(BaseModel):
    status: Literal["scoreable", "inference_only", "blocked"]
    reasons: tuple[str, ...]
    candidate_tasks: tuple[str, ...]
```

## Task 1: Scaffold the Package and Immutable Contracts

**Files:** `pyproject.toml`, `.gitignore`, `src/hyperspectrum/__init__.py`, all files under `src/hyperspectrum/contracts/`, `tests/contracts/test_contracts.py`.

1. Write failing tests that construct every public contract, reject a dense array with no axes, reject a blank axis unit, reject a prediction missing provenance digests, and confirm model instances are frozen.
2. Run `.venv/bin/pytest tests/contracts/test_contracts.py -q`; expected result: collection fails because `hyperspectrum` does not exist.
3. Create `pyproject.toml` with runtime dependencies `pydantic>=2.8,<3`, `typer>=0.12,<1`, `numpy>=2,<3`, `scipy>=1.13,<2`, `pyyaml>=6,<7`, and development dependencies `pytest`, `pytest-cov`, `ruff`, `mypy`.
4. Implement the contracts with `ConfigDict(frozen=True, extra="forbid")`. Validate `sha256` as 64 lowercase hexadecimal characters when present. Validate that `dense_array` artifacts have at least one axis and that all axis names are unique.
5. Require prediction provenance keys `model_digest`, `tool_digest`, `data_digest`, and `environment_digest`; values must be non-empty strings.
6. Run `.venv/bin/pytest tests/contracts/test_contracts.py -q`; expected result: all contract tests pass.
7. Run `.venv/bin/ruff check src tests` and `.venv/bin/mypy src`.
8. Commit: `feat: add immutable hyperspectrum contracts`.

## Task 2: Add a Safe HyperData CLI Gateway

**Files:** `src/hyperspectrum/hyperdata/gateway.py`, `src/hyperspectrum/hyperdata/models.py`, `tests/hyperdata/test_gateway.py`.

1. Write tests with an injected command runner. Assert the exact argv for a profile search starts with `("hyd", "--profile", "volcano")`, never invokes a shell, never reads a filesystem credential path, parses the last non-empty JSON line, and redacts bearer/token-looking text from raised errors.
2. Add tests for `FileNotFoundError`, exit code 1, invalid JSON, a human-readable table response, and a timeout. The structured path must fail closed on table output instead of scraping columns.
3. Run `.venv/bin/pytest tests/hyperdata/test_gateway.py -q`; expected result: import failure for the missing gateway.
4. Implement `HydGateway` with `subprocess.run(argv, shell=False, text=True, capture_output=True, timeout=...)`. Accept a runner dependency in the constructor for tests.
5. Implement `run()` and `search()`. `search(query)` invokes the current machine-readable contract exactly as `hyd [--profile PROFILE] search QUERY --ilike --json`. A client without this command is `unsupported_client`, not an invitation to scrape the legacy `dataset list --query` table.
6. Represent absent CLI, unauthenticated profile, unsupported JSON output, and transport failure with distinct exception types and stable error codes.
7. Run the gateway tests, Ruff, and mypy; expected result: pass.
8. Commit: `feat: add safe hyperdata gateway`.

## Task 3: Discover XAS Candidates from Multiple Evidence Channels

**Files:** `src/hyperspectrum/hyperdata/discovery.py`, `tests/hyperdata/test_discovery.py`, `tests/fixtures/hyperdata/xas-search.json`.

1. Create a redacted fixture containing the nine previously audited XAS candidates, including `zenodo-10606662`, `zenodo-16892323`, and `zenodo-154112`, plus duplicated hits across queries. The fixture records only public dataset code, title, description, counts, formats, license, parser status, and label evidence.
2. Write tests proving `discover_xas()` executes the fixed query set `("XAS", "XANES", "EXAFS", "absorption edge")`, deduplicates by dataset code, retains evidence from every query, and ranks candidates by explicit label/pairing/axis/parser evidence rather than name alone.
3. Write a regression test that an ASC file misparsed as an ESRI grid reduces readiness and never counts as a valid XAS energy axis.
4. Run `.venv/bin/pytest tests/hyperdata/test_discovery.py -q`; expected result: missing implementation failure.
5. Implement `discover_xas(gateway) -> tuple[DatasetCandidate, ...]` and a deterministic scoring function. Use only catalog metadata and small header evidence; do not download full datasets during discovery.
6. Ensure source query, parser status, axis/unit evidence, candidate ground-truth roles, license, and access status remain visible in `DatasetCandidate.evidence`.
7. Run focused tests, then the full suite; expected result: pass.
8. Commit: `feat: add evidence-led XAS discovery`.

## Task 4: Recommend Only Scoreable XAS Tasks

**Files:** `src/hyperspectrum/tasks/recommend.py`, `tests/tasks/test_recommend.py`.

1. Write table-driven tests for: noisy/clean pairs → scoreable denoising; repeated scans with a documented average proxy → scoreable denoising with proxy limitation; structure/spectrum pairs → scoreable forward prediction; oxidation labels → scoreable classification; spectra only → inference-only; inaccessible archive or invalid axis → blocked.
2. Assert that every scoreable verdict includes group split keys and at least one ground-truth role. Assert that `sample_id`, `compound_id`, or `acquisition_id` is used to prevent adjacent/repeated scans from crossing splits.
3. Run `.venv/bin/pytest tests/tasks/test_recommend.py -q`; expected result: missing implementation failure.
4. Implement pure functions `profile_xas_candidate(candidate)` and `recommend_xas_tasks(profile)`. Make reasons stable machine-readable codes, with optional human text in a separate field.
5. Add an explicit `lcf_weight_regression` recommendation only when mixture/composition truth exists. It must never be used as the primary denoising metric.
6. Run tests, Ruff, and mypy; expected result: pass.
7. Commit: `feat: add XAS task readiness recommendations`.

## Task 5: Validate and Register Declarative Tools

**Files:** `schemas/hyperspectrum-tool-v1.schema.json`, `src/hyperspectrum/registry/models.py`, `src/hyperspectrum/registry/loader.py`, `tools/xas/savgol/tool.yaml`, `tools/xas/xasdenoise/tool.yaml`, `tests/registry/test_loader.py`.

1. Write tests for valid classical and external-model manifests. Reject mutable image tags without a digest, missing source commit/release, unknown output roles, missing verify command, missing license state, and any manifest that enables weight download or training by default.
2. Run `.venv/bin/pytest tests/registry/test_loader.py -q`; expected result: missing implementation failure.
3. Implement schema version `hyperspectrum-tool/v1`. Required fields: id, version, modalities, tasks, runtime, entrypoint, inputs, outputs, resources, verify, source, license, and distribution policy.
4. Register `savgol` as a built-in classical baseline. Register `xasdenoise` as an external pinned adapter with `license: unknown` and `distribution: private-validation-only`; do not include source or weights.
5. Compute `tool_digest` from canonicalized manifest bytes. `ToolRegistry.match()` returns only tools whose modality, task, input/output roles, license policy, weights state, and resources match.
6. Run focused tests and full suite; expected result: pass.
7. Commit: `feat: add gated spectroscopy tool registry`.

## Task 6: Implement Canonical XAS Arrays, Metrics, and Baselines

**Files:** `src/hyperspectrum/plugins/xas/arrays.py`, `src/hyperspectrum/plugins/xas/metrics.py`, `src/hyperspectrum/plugins/xas/baselines.py`, `tests/plugins/xas/test_arrays.py`, `tests/plugins/xas/test_metrics.py`, `tests/plugins/xas/test_baselines.py`, `tests/fixtures/xas/denoising-pairs.npz`.

1. Generate a deterministic tiny fixture in the test source with three compound groups and explicit energy/eV axes; save the stable NPZ fixture through the test-data generation command documented in the test module. It is test-only and must be marked synthetic.
2. Write failing tests for monotonic energy validation, eV unit enforcement, interpolation only inside overlapping energy ranges, NaN rejection, and preservation of sample/group IDs.
3. Write metric tests for per-sample normalized spectrum RMSE, compound-grouped mean, grouped bootstrap confidence intervals with a fixed seed, and failure counting. The primary metric key is `normalized_spectrum_rmse`, direction `min`.
4. Write baseline tests for identity, moving average, Gaussian, Savitzky-Golay, PCA, and Gaussian Process. All methods must preserve the original energy coordinates and output one prediction/failure entry per input sample.
5. Run `.venv/bin/pytest tests/plugins/xas -q`; expected result: missing implementation failure.
6. Implement the array validator, metrics, deterministic aggregators, and baselines. Runtime-fitted PCA/GP must report `optimization_kind: runtime_fit`, never `training`.
7. Run focused tests, full suite, Ruff, and mypy; expected result: pass.
8. Commit: `feat: add XAS denoising baselines and metrics`.

## Task 7: Produce Immutable Run Plans and Execute Locally

**Files:** `src/hyperspectrum/execution/plan.py`, `src/hyperspectrum/execution/local.py`, `tests/execution/test_plan.py`, `tests/execution/test_local.py`.

1. Write tests for `RunPlan` containing task, dataset version/digest, tool digest, weight digest or explicit `none`, backend, resources, max samples, output directory, and dry-run state.
2. Assert planning fails if a model tool lacks verified weights, if a dataset is not scoreable, or if the requested backend cannot meet resources. Assert classical tools explicitly use `weight_digest: "none"`.
3. Write a local-executor test that runs Savitzky-Golay on the tiny fixture, emits `predictions.json`, array artifacts, `run.json`, and no leaderboard metrics. Formal scoring remains an ACE responsibility.
4. Run `.venv/bin/pytest tests/execution -q`; expected result: missing implementation failure.
5. Implement canonical digesting and atomic run-directory writes. The executor must fail the whole run on partial input materialization and list per-sample model failures without dropping samples.
6. Run focused tests and full suite; expected result: pass.
7. Commit: `feat: add reproducible local execution plans`.

## Task 8: Expose an Agent-Ready CLI and Skills

**Files:** `src/hyperspectrum/cli.py`, `skills/discover-xas-dataset/SKILL.md`, `skills/run-xas-baseline/SKILL.md`, `tests/cli/test_cli.py`, `README.md`.

1. Write CLI tests for `hyperspectrum doctor --json`, `data discover --modality xas --profile volcano --json`, `task recommend --candidate-file ... --json`, `tools match --task xas-denoising --json`, `run plan ... --json`, and `run local ... --json`.
2. Require a single JSON envelope on stdout in agent mode: `{schema_version, ok, result, warnings, error}`. Human logs go to stderr. Exit codes: 0 success, 2 invalid request/readiness, 3 auth/connection, 4 missing asset/tool, 5 execution failure.
3. Run `.venv/bin/pytest tests/cli/test_cli.py -q`; expected result: missing command failures.
4. Implement Typer commands as thin wrappers around public services. No business rules may live only in CLI callbacks.
5. Write both skills with explicit preflight, read-only boundaries, JSON commands, refusal behavior, and evidence output. The run skill must stop before external execution or adaptation unless the user authorized it.
6. Update README with the ownership boundary and the exact first story commands.
7. Run full tests, Ruff, mypy, and `hyperspectrum --help`; expected result: pass.
8. Commit: `feat: expose XAS agent CLI and skills`.

## Task 9: Prove M0 Against the Real Volcano Catalog and 5090

**Files:** `docs/evidence/xas-m0-selection.schema.json`, `docs/dev/2026-08-08-xas-m0.md`, generated redacted `docs/evidence/xas-m0-selection.json` after a successful run.

1. On `5090dzr`, run read-only preflight commands: `command -v hyd`, `hyd -V`, `hyd --profile volcano whoami`, and `hyd search --help`. Do not inspect credential or profile files.
2. If `hyd` is absent, request approval before installing exactly `hyperdata-client[em]` from the approved HyperData repository with `uv tool install --force --python 3.12 "hyperdata-client[em] @ git+https://github.com/hyper-instrument/hyper-data.git#subdirectory=client"`. Re-run preflight. If authentication is absent, stop with error code 3 and request `hyd login` from the user; do not bypass auth.
3. Run `hyperspectrum data discover --modality xas --profile volcano --json` on the 5090. Save the raw response only under ignored `runs/`; generate a redacted selection file containing query time, profile name, client version, selected dataset code/version, public metadata, task verdict, ground-truth evidence, split group keys, asset digests, and blockers.
4. Validate the file against `docs/evidence/xas-m0-selection.schema.json`. The schema requires all ACE handoff fields and forbids secrets, signed URLs, local private paths, null dataset versions, and null content digests.
5. Run a local 5090 Savitzky-Golay smoke prediction with `--max-samples 8`. This proves data materialization and the runtime contract, not model quality. Record only digests, sample counts, status, and redacted artifact names.
6. Update `docs/dev/2026-08-08-xas-m0.md` with observed facts. If no scoreable dataset exists, document `inference_only` or `blocked`; do not create `xas-m0-selection.json` and do not proceed to the ACE M1 plan.
7. Re-run the complete verification suite:

   ```bash
   .venv/bin/pytest -q
   .venv/bin/ruff check src tests
   .venv/bin/mypy src
   .venv/bin/hyperspectrum doctor --json
   git diff --check
   ```

8. Commit redacted evidence only. Commit: `docs: record real XAS M0 evidence`.

## M0 Completion Gate

M0 is complete only when all of the following are true:

- The full local test/type/lint suite passes.
- The agent can search the active `volcano` profile without reading credentials.
- At least one real XAS dataset has a non-null version and content digest.
- Its scoreability verdict is backed by explicit ground-truth and split-group evidence.
- A local 5090 classical smoke prediction emitted a valid `PredictionBundle` for every materialized sample.
- The handoff evidence validates against its schema and contains no secret, signed URL, or private raw path.

If any condition fails, M0 remains `blocked` with the observed reason. A cached audit, synthetic fixture, or random model output cannot satisfy this gate.
