# XASDenoise Runtime Adapter Implementation Plan

> Superseded scientific assumption: the pinned Zenodo datasets prove that null
> model normalization follows external per-spectrum pre/post-edge normalization.
> Raw `ketek/i0` input is therefore blocked as `input_contract_unverified`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one provenance-complete HyperSpectrum adapter for the official XASDenoise nonuniform checkpoint that can execute unchanged in local Python, OCI, 5090, and Bohr container contexts.

**Architecture:** Historical plan only. The implemented correction rejects raw input until an evidenced, reversible upstream normalization state is available.

**Tech Stack:** Python 3.10+, NumPy, SciPy, optional runtime-provided PyTorch, Pydantic v2, pytest, Ruff, mypy.

## Global Constraints

- Upstream code is pinned to commit `bda749ee956f9e02acc6995f238d759682ee2ca8` and licensed MIT.
- The external Zenodo 17434349 asset is `xas_denoiser_model_noise2noise_nonuniformly_sampled_notnormalized.pth`, licensed CC-BY-4.0, exactly 780409 bytes, SHA-256 `09620ee9ea0c96585f534d76ce42aa72edf2cf71e481f5737e43e93116e24160`.
- Do not commit checkpoint or benchmark bytes and never download a checkpoint automatically.
- The checkpoint's declared `normalization_method=None` does not establish raw-input compatibility; production execution remains fail-closed.
- Official step-baseline preprocessing is versioned and reversible only within upstream-normalized absorption units; it cannot restore raw native units without the missing pre/post-edge state.
- Load torch checkpoints with `weights_only=True`, load state dictionaries strictly, use evaluation/inference mode, and verify shape, finiteness, energy order, and unchanged parameters.
- New execution emits the existing `hyperspectrum-prediction/v3` and `hyperspectrum-run/v2` artifacts and preserves all three independent dataset/asset digests.

---

### Task 1: Explicit raw normalization

**Files:** Modify `src/hyperspectrum/denoising/normalization.py`; test `tests/denoising/test_normalization.py`.

**Interfaces:** Historical normalization-kernel work is independent of checkpoint compatibility.

- [ ] Retain generic raw identity normalization without claiming checkpoint compatibility.
- [ ] Run `.venv/bin/pytest tests/denoising/test_normalization.py -q` and confirm failure because the registry method is absent.
- [ ] Add `identity_raw` to the method metadata, state parameter contract, default registry, and per-sample derivation.
- [ ] Re-run the focused normalization tests to green.

### Task 2: Safe official adapter and preprocessing

**Files:** Create `src/hyperspectrum/adapters/__init__.py`, `src/hyperspectrum/adapters/xasdenoise.py`; test `tests/adapters/test_xasdenoise.py`.

**Interfaces:** Produce `XASDenoiseAdapter`, `XASDenoisePreprocessingState`, `XASDenoisePrediction`, `XASDenoiseFailure`, `denoise_spectra`, and `verification_report`.

- [ ] Add failing tests with a deterministic fake torch runtime and tiny fake checkpoint bytes. Assert exact size/SHA checks happen before torch loading; `torch.load` receives `weights_only=True` and CPU map location; the state dict is strict; the model is in eval/inference mode; bad shape/non-finite/decreasing-energy/masked inputs and parameter mutation fail closed.
- [ ] Add tests that reject raw ratios and limit step restoration to upstream-normalized absorption units.
- [ ] Run `.venv/bin/pytest tests/adapters/test_xasdenoise.py -q` and confirm collection fails because the adapter is absent.
- [ ] Implement the exact four-layer, kernel-size-nine, bias-free upstream architecture behind a lazy torch import. Implement safe weight verification/loading and per-prediction parameter fingerprints.
- [ ] Implement deterministic official step-baseline fitting, subtraction, inference, restoration, and immutable preprocessing records. The batch wrapper returns one ordered prediction/failure per input without reading targets.
- [ ] Re-run adapter tests, then refactor only while green.

### Task 3: Declarative asset registry

**Files:** Modify `src/hyperspectrum/registry/models.py`, both `schemas/hyperspectrum-tool-v1.schema.json` copies, both `tools/xas/xasdenoise/tool.yaml` copies, and `tests/registry/test_loader.py`.

**Interfaces:** Extend present weights with immutable `asset_id`, `source_url`, `filename`, `size_bytes`, and `license`; register the local adapter entrypoint and exact upstream/asset facts.

- [ ] Add failing model/schema tests that accept the exact XASDenoise asset and reject a missing/invalid URL, filename, non-positive size, blank license, digest, source commit, or code license.
- [ ] Run `.venv/bin/pytest tests/registry/test_loader.py -q` and confirm the current manifest fails the new expectations.
- [ ] Implement the typed/schema asset fields and update the source and packaged manifests to MIT/open distribution, the pinned upstream commit, a resolvable adapter entrypoint, and the exact CC-BY-4.0 checkpoint declaration.
- [ ] Make default availability distinguish a declaratively complete external asset from bytes that execution has not yet mounted; execution remains the byte-verification authority.
- [ ] Re-run registry tests to green.

### Task 4: Shared v3 execution path

**Files:** Modify `src/hyperspectrum/execution/local.py`, `src/hyperspectrum/execution/plan.py`, `src/hyperspectrum/agent.py`, `src/hyperspectrum/cli.py`, their package exports, and focused execution/agent/CLI tests.

**Interfaces:** `execute_local_run(..., weight_files: Sequence[Path] = ())`, service/CLI repeated `--weight-file`, tool-specific immutable parameters, and unchanged PredictionBundle v3 filenames and digest fields.

- [ ] Add failing tests proving XASDenoise planning exposes `input_contract_unverified` even with exact declared weights.
- [ ] Add failing executor tests using the real adapter boundary with a deterministic injected runtime: exactly one mounted weight is required and checked; canonical XANES input is used; per-sample preprocessing states appear in prediction provenance; the existing five-key prediction NPZ and v3 digest protocol remain unchanged.
- [ ] Add failing CLI/service tests proving repeated `--weight-file` is forwarded while SavGol remains compatible without weights.
- [ ] Run the focused tests and confirm failures are caused by the missing dispatch/arguments.
- [ ] Generalize verified entrypoint loading/result handling and keep one atomic writer for both tools. Validate the declared weight before importing torch or invoking the adapter.
- [ ] Add fixed model parameters during planning and record actual device plus normalized-unit preprocessing semantics without adding ACE-owned metrics.
- [ ] Re-run execution, agent, CLI, contract, and packaging tests to green.

### Task 5: Agent metadata, documentation, and verification

**Files:** Modify `skills/run-xas-baseline/SKILL.md`, `skills/run-xas-baseline/agents/openai.yaml`, `README.md`, `docs/dev/2026-08-09-xasdenoise-runtime-adapter.md`, and packaging/skill tests as needed.

**Interfaces:** Document mounted-weight invocation, license/attribution, CPU/GPU/OCI/Bohr equivalence, explicit normalization/baseline semantics, and non-goals.

- [ ] Update skill and agent metadata so XASDenoise is a declared fixed-weight capability, not an unknown-license/missing-weight blocker; retain the no-download and user-authorization gates.
- [ ] Record the 5090 audit facts and exact source/asset identities without private paths or model bytes.
- [ ] Run focused suites for denoising, adapter, registry, execution, agent, CLI, skills, and packaging.
- [ ] Run the complete pytest suite, Ruff, strict mypy, `git diff --check`, and a fresh wheel/installed-console smoke.
- [ ] When source-transfer authority and the exact checkpoint are available together, copy only the changed source tree to a fresh `/tmp` directory on the configured 5090. Routing is restored, but the current-source probe remains blocked; the older checkpoint smoke is not a success claim for this revision.
- [ ] Inspect the full diff, commit locally on `codex/xas-agent-m0`, and do not push.
