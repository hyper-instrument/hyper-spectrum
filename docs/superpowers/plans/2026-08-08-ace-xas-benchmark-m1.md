# ACE XAS Benchmark M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended for this session) or `superpowers:executing-plans` to execute this plan task by task. Use `superpowers:test-driven-development` for every implementation task and `superpowers:verification-before-completion` before claiming a milestone complete.

**Goal:** Consume the verified HyperSpectrum M0 XAS evidence, register a real XAS denoising research board in ACE Benchmark, run classical and fixed-weight XASDenoise variants on a local 5090 and through Bohr with the same adapter contract, and publish a provenance-complete report suitable for later one/few-shot variants.

**Architecture:** ACE owns dataset/model/asset registration, deterministic splits, metrics, execution backends, reports, and boards. A thin ACE research adapter imports the pinned HyperSpectrum runtime and emits the standard `metrics.json` envelope. Local Docker and Bohr consume the same staged data, weights, adapter, and manifest. Missing real evidence, weights, license permission, or backend parity is a hard gate.

**Tech Stack:** ACE Benchmark `acebench-lb research` CLI, Python 3.10+, HyperSpectrum pinned commit, NumPy/SciPy, PyTorch only inside the fixed-weight model image, Docker/DeployMaster, Bohr CLI 2.5.17+, pytest.

## Global Constraints

- Start only after the M0 completion gate in `2026-08-08-hyper-spectrum-xas-agent-m0.md` passes and `docs/evidence/xas-m0-selection.json` exists with valid real digests.
- Work in a dedicated ACE Benchmark branch/worktree; do not modify or reopen the merged PR #109 contract ownership.
- Pin HyperSpectrum by immutable Git commit in the adapter/image. Do not duplicate its domain logic inside ACE.
- Use the standard research adapter interface: `IMAGE_ENTRYPOINT --data /data --weights /weights/model.bin[!inner] --max-samples N --out /out/metrics.json`.
- Bohr has no bind mounts. Stage data, weights, and adapter through the existing backend and record the resolved image digest.
- Never use random initialization when weights are absent or invalid. Mark the variant `blocked`.
- Never copy or publicly redistribute `Even-Ma/xas` or XASDenoise until licensing allows it. A private validation image may reference a pinned external checkout.
- Formal metrics are computed over all test samples; partial input, silent sample dropping, or test-set selection invalidates the run.
- Primary metric: compound-grouped mean `normalized_spectrum_rmse`, direction `min`, with grouped bootstrap confidence interval. LCF weight MAE is a separate optional board/track only when composition truth exists.

---

## Cross-Repository Inputs

The plan consumes these immutable inputs:

```text
HyperSpectrum plan: docs/superpowers/plans/2026-08-08-hyper-spectrum-xas-agent-m0.md
HyperSpectrum evidence: docs/evidence/xas-m0-selection.json
HyperSpectrum commit: recorded in the evidence file after M0
ACE Benchmark repository: hyper-instrument/ace-benchmark
Existing private image: registry.dp.tech/davinci/uni-xas:20260807163016
```

The existing Uni-XAS image is import/compile verified but has no weights and no XAS denoising adapter. It is not treated as a passing benchmark image.

## ACE File Map

Create or modify these files in the ACE Benchmark worktree:

```text
harness/docker/research/xasdenoise/run.py
harness/docker/research/xasdenoise/verify.py
harness/docker/research/xasdenoise/requirements.lock
harness/docker/research/xasdenoise/tool-manifest.json
harness/docker/research/xasdenoise/tests/test_run.py
harness/docker/research/xasdenoise/tests/test_verify.py
src/ace_leaderboard/seed/data/research_models.json
src/ace_leaderboard/seed/data/research_datasets.json
src/ace_leaderboard/seed/data/research_assets.json
tests/test_research_seed_xas.py
scripts/check_xas_m0_evidence.py
tests/test_check_xas_m0_evidence.py
scripts/check_xas_run_parity.py
tests/test_check_xas_run_parity.py
docs/dev/research/xas/2026-08-08-xas-denoising-m1.md
docs/dev/research/xas/evidence/local-run.json
docs/dev/research/xas/evidence/bohr-run.json
docs/dev/research/xas/evidence/parity.json
```

Raw datasets, weights, logs, signed URLs, and full reports remain in configured object storage or ignored run directories.

## Task 1: Enforce the Real-Evidence Handoff Gate

**Files:** `scripts/check_xas_m0_evidence.py`, `tests/test_check_xas_m0_evidence.py`.

1. Write tests that load valid, missing-field, null-digest, signed-URL, local-private-path, inference-only, and blocked evidence fixtures. A valid fixture must include dataset code/version/digest, label/ground-truth roles, split group keys, license/access state, sample/file counts, HyperSpectrum commit, and source query timestamp.
2. Run the focused tests; expected result: missing script failure.
3. Implement a pure validator plus CLI. Exit 0 and emit canonical JSON for valid scoreable evidence; exit 2 with stable error codes for invalid or non-scoreable evidence.
4. The validator must reject placeholder strings, zeroed SHA-256 values, `http` URLs with query credentials, `/home/` and `/data/` source paths, and missing license/access declarations.
5. Run `python scripts/check_xas_m0_evidence.py /absolute/path/to/hyper-spectrum/docs/evidence/xas-m0-selection.json`. If it exits non-zero, stop M1 and return to M0; do not seed an ACE board.
6. Run focused tests and the ACE repository's standard lint/type checks.
7. Commit: `test: gate XAS benchmark on real M0 evidence`.

## Task 2: Register the XAS Denoising Research Board

**Files:** `src/ace_leaderboard/seed/data/research_models.json`, `src/ace_leaderboard/seed/data/research_datasets.json`, `src/ace_leaderboard/seed/data/research_assets.json`, `tests/test_research_seed_xas.py`.

1. Write tests that derive the dataset/version/digests from the validated evidence and assert exactly three independent boards: `xas-denoising-zero-shot`, `xas-denoising-one-shot`, and `xas-denoising-few-shot`.
2. Assert the zero-shot board has primary metric `normalized_spectrum_rmse` with direction `min`, a grouped-bootstrap aggregation declaration, explicit sample/group split metadata, and no LCF metric unless the evidence declares composition truth.
3. Assert model variants distinguish `savgol`, `gaussian`, `moving-average`, `pca`, `gaussian-process`, and `xasdenoise-zero-shot`. One/few-shot boards exist with zero runs until M2; they must not inherit zero-shot results.
4. Run the focused seed test; expected result: missing entries.
5. Add dataset, asset, model, variant, and board entries using IDs deterministically derived from the real dataset code/version. Populate all asset hashes and counts from the evidence validator output; never type fake digests into seed files.
6. Mark XASDenoise as `blocked` until the weight asset and private distribution policy pass Task 4. Classical variants may be `ready` after adapter tests.
7. Run focused and full ACE seed tests; expected result: pass.
8. Commit: `feat: register XAS denoising research boards`.

## Task 3: Build the Thin ACE Adapter

**Files:** `harness/docker/research/xasdenoise/run.py`, `harness/docker/research/xasdenoise/verify.py`, `harness/docker/research/xasdenoise/requirements.lock`, `harness/docker/research/xasdenoise/tool-manifest.json`, adapter tests.

1. Write adapter tests with a real-format, tiny, redistribution-safe fixture selected during M0. If no such subset can be committed, materialize it at test time from an approved local test asset; do not substitute a synthetic benchmark input.
2. Assert `run.py` uses ACE `entry.parse_args`, `resolve_weights`, and `write_metrics`; honors `--max-samples`; fails on missing/partial samples; imports HyperSpectrum at the pinned commit; and emits exactly `{metrics, nSamples, durationS, artifacts}`.
3. Assert `verify.py` imports every runtime dependency used by `run.py`, checks the HyperSpectrum commit and tool manifest digest, but does not require weights.
4. Run adapter tests; expected result: missing adapter failure.
5. Implement `run.py` as translation only: ACE arguments → HyperSpectrum `RunPlan`/executor → ACE metrics envelope. Do not reimplement axis validation, XAS baselines, or metric formulas.
6. Implement tool selection from the staged manifest. Fixed-weight execution must call `resolve_weights`; classical baselines must reject a non-empty weights path to prevent accidental variant confusion.
7. Run adapter tests and direct probe commands:

   ```bash
   python harness/docker/research/xasdenoise/verify.py
   python harness/docker/research/xasdenoise/run.py --help
   ```

8. Commit: `feat: add thin XAS denoising research adapter`.

## Task 4: Verify the Fixed Weights and Build the Private Model Image

**Files:** `harness/docker/research/xasdenoise/tool-manifest.json`, `docs/dev/research/xas/2026-08-08-xas-denoising-m1.md`.

1. On `5090dzr`, inventory `/data2/cgs/model`, `/data2/cgs/dino`, `/data2/cgs/locateanything`, `/data2/cgs/sam3`, and permitted shared XAS paths with read-only `find`/`stat` commands. Do not access `/data/dzr/myw` unless the owner has granted read permission.
2. Locate the exact XASDenoise architecture, checkpoint, upstream commit, and license evidence. Calculate SHA-256 and run the upstream fixed-weight probe on a real-format sample.
3. If source or weight licensing is unknown, keep the image private and set `distribution: private-validation-only`. If the checkpoint is absent or fails the probe, mark the variant `blocked` and continue only with classical variants.
4. Rebuild a GPU-capable private image through the supported DeployMaster service/tooling, not the known-outdated `scripts/deploymaster_build.py`. The build declaration installs the pinned HyperSpectrum commit, pinned external source, adapter dependencies, CUDA-compatible PyTorch, and no embedded private dataset.
5. Pass `verify.py` as the image `verify_commands` so the same declaration is consumed during build and benchmark verification.
6. Run the built image's `verify.py`; expected result: exit 0 and exact import/commit/tool-digest checks pass. Then run `run.py` with a deliberately absent checkpoint; expected result: non-zero `missing_weights`, proving random initialization is disabled.
7. Record image tag and resolved digest, source commit, weight digest/location class, build task ID, CUDA/PyTorch versions, and probe output in the development document. Do not record tokens or signed URLs.
8. Update the ACE model/asset entries from `blocked` to `ready` only after checksum and probe pass. Run the focused seed and adapter tests; expected result: pass.
9. Commit: `docs: record verified XASDenoise image and weights`.

## Task 5: Run the Real Benchmark on the Local 5090

**Files:** `docs/dev/research/xas/evidence/local-run.json`, `docs/dev/research/xas/2026-08-08-xas-denoising-m1.md`.

1. Use `acebench-lb research match` and `research run` with the selected real dataset asset, deterministic manifest, staged weight asset when required, and the private image digest.
2. First run `--max-samples 8` for every ready classical variant and XASDenoise. Expected result: terminal success, eight accounted samples, explicit axes, finite metrics, and zero silent drops. Any other result blocks the full run.
3. Run the complete fixed test split. Any missing sample, changed split digest, NaN metric, or unresolved artifact invalidates the run.
4. Generate the ACE report and inspect best/worst spectra plus residual plots. Record `nSamples`, primary/secondary metrics, confidence intervals, duration, data/weight/adapter/image digests, and failure count in the redacted evidence JSON.
5. Confirm the fixed-weight variant loads the checkpoint by digest and does not train. Capture a no-gradient/no-optimizer assertion from the adapter/model probe.
6. Validate the redacted evidence JSON with the repository test; expected result: pass.
7. Commit: `test: record local 5090 XAS benchmark evidence`.

## Task 6: Run the Same Contract Through Bohr

**Files:** `docs/dev/research/xas/evidence/bohr-run.json`, `docs/dev/research/xas/2026-08-08-xas-denoising-m1.md`.

1. Confirm `bohr --version` is at least 2.5.17 and that the approved account/queue can access the private registry and staged assets. Use CLI status commands; do not read auth files.
2. Submit the exact local run manifest through `acebench-lb research run --backend bohr`. ACE must stage data, weights, and adapter because Bohr does not support bind mounts.
3. Poll through ACE/Bohr until terminal state. Expected result: `SUCCEEDED`, the same accounted sample set as local, finite metrics, and retrievable artifacts. Preserve job ID, timestamps, image tag and resolved digest, staged asset digests, exit status, metrics, and artifact URIs in redacted evidence.
4. If queue, registry, quota, or asset access blocks execution, record the explicit backend blocker; do not replace the Bohr result with the local result.
5. Generate the ACE report for the completed Bohr run.
6. Validate the redacted Bohr evidence JSON with the repository test; expected result: pass.
7. Commit: `test: record Bohr XAS benchmark evidence`.

## Task 7: Prove Local/Bohr Parity and Publish the Zero-Shot Board

**Files:** `scripts/check_xas_run_parity.py`, `tests/test_check_xas_run_parity.py`, `docs/dev/research/xas/evidence/parity.json`, `docs/dev/research/xas/2026-08-08-xas-denoising-m1.md`, relevant seed run counts/status fields.

1. Write failing tests for mismatched data, split, weight, adapter, tool, image digest, sample IDs, and failure sets; also test classical tolerance `1e-12` and fixed-weight GPU tolerance `1e-6`.
2. Run the focused parity tests; expected result: missing script failure.
3. Implement the parity check. It must require identical immutable digests and sample/failure sets before comparing metrics.
4. Run it against local and Bohr evidence; expected result: exit 0 and a redacted `parity.json`. Any mismatch blocks publication.
5. Verify report artifact roles and order match the canonical research contract established by PR #109. Ensure no artifact contains a signed URL, token, raw private path, or unredacted log.
6. Publish/update only the zero-shot board after parity passes. Leave one-shot and few-shot boards empty and ready for M2 derived variants.
7. Run the full ACE test suite and final commands required by the repository, plus `git diff --check`; expected result: pass.
8. Update the development document with achieved/blocked status and exact evidence links.
9. Commit: `feat: publish reproducible XAS zero-shot board`.

## M1 Completion Gate

M1 is complete only when:

- The board is seeded from a validated real HyperData dataset version and content digest.
- Classical baselines have full-test-split quantitative results.
- XASDenoise is either a verified fixed-weight result or explicitly `blocked`; random initialization never appears.
- Local 5090 and Bohr both run the same adapter and immutable asset/image digests.
- Parity passes within declared tolerances.
- ACE generates the report and zero-shot board without sample dropping.
- One-shot/few-shot boards exist but contain no fabricated or zero-shot-copied results.

If XASDenoise remains blocked, M1 may publish a classical-only zero-shot board, but the milestone report must say that the model comparison is incomplete. If Bohr remains blocked, M1 itself is not complete even when the local report exists.
