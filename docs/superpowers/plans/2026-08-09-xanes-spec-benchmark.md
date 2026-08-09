# XANES SPEC Benchmark Implementation Plan

> Execute in strict red-green-refactor order. Do not use synthetic pass-through
> assumptions that make a source digest equal derived NPZ bytes.

**Goal:** Add a deterministic Zenodo 10606662 XANES benchmark materializer and
explicit v3 digest contracts while retaining read-only/execution compatibility
for v2 artifacts.

**Architecture:** A new `hyperspectrum.datasets.xanes_spec` module owns source
manifest verification, strict SPEC parsing, fixed split/grid/noise logic,
canonical bundle serialization, QC, and denoising-contract loading. Execution
and prediction introduce separate v3 models and schema-dispatched loaders.
Evidence validation dispatches between frozen v1 and explicit-digest v2.

**Tech stack:** Python 3.10+, Pydantic v2, NumPy, Typer, pytest, Ruff, mypy.

## Task 1: Freeze adapter behavior in RED tests

**Files:**

- Add `tests/fixtures/xas/spec-real-header.txt`
- Add `tests/datasets/test_xanes_spec.py`

1. Add a compact genuine `#L` header fixture from Zenodo 10606662 with 135
   deterministic rows.
2. Test strict parsing, empty/short rejection, extensionless and `.txt`
   classification, size-drift acceptance with SHA match, and SHA mismatch
   refusal.
3. Test exact fixed split rules and composition leakage rejection.
4. Test reference grid comes only from the first sorted valid train sample.
5. Test Binomial thinning determinism, input-order independence, dose-separated
   bytes, target semantics, and `DenoisingPair` conversion.
6. Run the new test file and confirm it fails because the adapter is absent.

## Task 2: Implement the materializer and CLI

**Files:**

- Add `src/hyperspectrum/datasets/__init__.py`
- Add `src/hyperspectrum/datasets/xanes_spec.py`
- Modify `src/hyperspectrum/agent.py`
- Modify `src/hyperspectrum/cli.py`
- Modify CLI/service tests

1. Implement immutable manifest, QC, parsed-spectrum, and materialization result
   models with canonical JSON digests.
2. Implement inventory verification and strict single-scan SPEC parsing.
3. Implement path classification, fixed split, train-only reference grid,
   linear interpolation, and count-space Binomial thinning.
4. Implement fixed-metadata canonical NPZ serialization and bundle loading into
   `SpectrumSample`/`DenoisingPair`.
5. Add `hyperspectrum data materialize-xanes-spec` with JSON-envelope output.
6. Run focused adapter/CLI tests to green, then refactor without changing
   externally observed behavior.

## Task 3: Migrate execution and prediction digest contracts

**Files:**

- Modify `src/hyperspectrum/execution/plan.py`
- Modify `src/hyperspectrum/execution/local.py`
- Modify `src/hyperspectrum/execution/__init__.py`
- Modify `src/hyperspectrum/contracts/prediction.py`
- Modify `src/hyperspectrum/contracts/__init__.py`
- Modify execution/contract/agent/CLI tests

1. Add RED tests proving a v3 plan accepts three unequal digests and local
   execution checks NPZ bytes only against `benchmark_asset_digest`.
2. Add RED tests proving v2 artifacts still parse and execute through the
   explicit legacy branch.
3. Add `RunPlanV2`, `RunPlanV3`, and schema-dispatched parsing. New planning
   requires a benchmark manifest and returns v3.
4. Add `PredictionBundleV3`; emit v2 or v3 according to the plan version.
5. Update service/CLI inputs to carry the benchmark manifest into planning.
6. Run focused execution and contract tests to green.

## Task 4: Migrate evidence with compatibility

**Files:**

- Add `docs/evidence/xas-m0-selection-v2.schema.json`
- Add `src/hyperspectrum/resources/schemas/xas-m0-selection-v2.schema.json`
- Modify `src/hyperspectrum/evidence.py`
- Modify evidence and packaging tests

1. Add RED evidence tests for explicit source dataset, verified content
   manifest, and derived asset digests with no cross-class equality.
2. Add the v2 schema and dispatch validation by `schema_version`; keep the
   existing v1 schema unchanged.
3. Validate each v2 provenance chain against the corresponding identity class.
4. Verify both schemas ship in the wheel.

## Task 5: Verification and handoff

1. Run all focused suites covering datasets, denoising, execution, contracts,
   evidence, CLI, agent services, and packaging.
2. Run the complete pytest suite.
3. Run `ruff check src tests` and strict mypy.
4. Build a fresh wheel in a clean temporary virtual environment and run the
   packaged CLI smoke tests.
5. Request an independent review, address findings with new RED tests, and
   repeat affected gates.
   The review hardening validates the complete planning manifest and its
   recomputed source chains, parses and hashes identical byte snapshots, bounds
   grid endpoint jitter, and keeps five-key execution on the legacy v2 branch.
6. Inspect the diff for source/share-fix artifacts and remote changes.
7. Commit locally on `codex/xas-agent-m0`; do not push.
