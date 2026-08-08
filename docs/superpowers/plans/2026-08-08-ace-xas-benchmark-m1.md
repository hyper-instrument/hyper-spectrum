# ACE XAS Benchmark M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended for this session) or `superpowers:executing-plans` to execute this plan task by task. Use `superpowers:test-driven-development` for every implementation task and `superpowers:verification-before-completion` before claiming a milestone complete.

**Goal:** Consume the verified HyperSpectrum M0 XAS evidence, register a real XAS denoising research board in ACE Benchmark, run classical and fixed-weight XASDenoise variants on a local 5090 and through Bohr with the same adapter contract, and publish a provenance-complete report suitable for later one/few-shot variants.

**Architecture:** ACE owns dataset/model/asset registration, deterministic splits, `ExecutionSpec`, outer execution backends, staging, metrics ingestion, reports, and boards. Each ACE `ResearchModel` has its own thin adapter that imports the pinned HyperSpectrum runtime and emits the standard `metrics.json` envelope. Local Docker and Bohr consume the same execution contract and immutable assets; inside either container, the HyperSpectrum `RunPlan` is always local. Missing real evidence, weights, license permission, or backend parity is a hard gate.

**Tech Stack:** ACE Benchmark `acebench-lb research` CLI, Python 3.11+, HyperSpectrum pinned commit, NumPy/SciPy, PyTorch only inside the fixed-weight model image, Docker/DeployMaster, Bohr CLI 2.5.17+, pytest.

## Global Constraints

- Start only after the M0 completion gate in `2026-08-08-hyper-spectrum-xas-agent-m0.md` passes and `docs/evidence/xas-m0-selection.json` exists with valid real digests.
- Work in a dedicated ACE Benchmark branch/worktree; do not modify or reopen the merged PR #109 contract ownership.
- Pin HyperSpectrum by immutable Git commit in the adapter/image. Do not duplicate its domain logic inside ACE.
- Use the standard research adapter interface: `IMAGE_ENTRYPOINT --data /data [--weights /weights/model.bin]... [--weights /archive/model.tar.gz!relative/internal/path] [--max-samples N] --out /out/metrics.json`. `--weights` is optional and repeatable; `!inner` is only the archive-plus-inner-path form. `--max-samples` is optional and must be omitted for a full-split run.
- Select the outer backend through ACE settings. For Bohr use `ACEBENCH_LB_RESEARCH_BACKEND=bohr-job` together with `ACEBENCH_LB_RESEARCH_BOHR_PROJECT_ID` and `ACEBENCH_LB_RESEARCH_BOHR_MACHINE_TYPE`; `research run` has no `--backend` option.
- Keep the HyperSpectrum `RunPlan` backend local inside the container. It must not duplicate ACE `ExecutionSpec`, backend selection, staging, metrics ingestion, or report generation.
- In M1, register each classical algorithm as an independent ACE `ResearchModel` with its own model directory and thin adapter. Do not select algorithms through a sibling runtime manifest; a shared parameterized adapter requires a separate ACE framework contract change first.
- Release images fail closed. With the frozen e62 CLI, never combine `research build --verify` with `--push`: that implementation pushes before it refuses a failed probe. Build and verify without push, push the exact repository/tag only after verification passes, then pull/re-resolve the registry digest and pullability before recording it.
- Bohr has no bind mounts. Stage data, weights, and adapter through the existing backend and record the resolved image digest.
- Never use random initialization when weights are absent or invalid. Set the ACE `ResearchModel.status` to `pending_weights` and preserve the exact blocker in evidence and milestone gates; ACE variants and assets have no `blocked` status field.
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

## Frozen Contract Mapping

| HyperSpectrum handoff | ACE owner/representation | Required rule |
| --- | --- | --- |
| Dataset/version/content digest and each file digest | ACE dataset plus `ResearchAsset` records | Preserve the verified values; never substitute placeholders. |
| Asset role, file, size, SHA-256, materialization mode, M0 `inner_path`, semantics, and full-split counts/source | M0 JSON `inner_path` → ACE Python `inner_path` → ACE wire/report `innerPath` | Preserve the snake_case M0 source field, map it explicitly at the ACE boundary, and record `mount-file`, `mount-dir`, or `unpack` per asset. Never infer materialization from a suffix. For a directly executable mount request, `inner_path` is present exactly when mode is `unpack`. |
| Split and selection manifest, selected sample IDs, count, and selection digest | Run inputs and redacted evidence | Local and Bohr must account for the exact same full test set. |
| HyperSpectrum plan/model/tool/implementation/weight/data/environment/selection identities | `PredictionBundle`, run manifest, and parity evidence | These are domain/runtime identities, not the ACE adapter digest. |
| ACE `adapterDigest` | Hash of the staged `common/entry.py` and model-specific `run.py` | ACE computes it; HyperSpectrum must not fabricate or relabel it. |
| ACE image digest | Image resolution performed by ACE | Record the resolved digest separately from tag and tool digest. |
| ACE backend and opaque `backendHandle` | ACE launcher/backend result | Record only values returned by ACE; use null/not-applicable when unavailable. |
| `PredictionBundle` predictions and failures | Adapter output consumed by ACE | Any partial input, missing prediction, or failed sample makes a formal XAS run invalid: exit non-zero and do not emit complete-looking metrics. |
| Adapter metrics envelope | `{metrics, nSamples, durationS, artifacts}` | Emit only after complete prediction/evaluation; ACE owns ingestion and the `research-report-context-1` report. |

M0 evidence predates ACE submission, so it cannot truthfully contain an ACE adapter digest, image digest, backend, or backend handle. Those fields must remain null/not-applicable until ACE produces them.

## ACE File Map

Create or modify these files in the ACE Benchmark worktree:

```text
harness/docker/research/xas-savgol/run.py
harness/docker/research/xas-gaussian/run.py
harness/docker/research/xas-moving-average/run.py
harness/docker/research/xas-pca/run.py
harness/docker/research/xas-gaussian-process/run.py
harness/docker/research/xasdenoise/run.py
harness/docker/research/<model-id>/Dockerfile
harness/docker/research/<model-id>/verify.py
harness/docker/research/<model-id>/requirements.lock
harness/docker/research/<model-id>/tests/test_run.py
harness/docker/research/<model-id>/tests/test_verify.py
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

1. Write tests that load valid, missing-field, null-digest, signed-URL, local-private-path, inference-only, and blocked evidence fixtures. A valid source M0 JSON fixture must include dataset code/version/digest; every asset's role, file, size, SHA-256, materialization mode, snake_case `inner_path` exactly when mode is `unpack`, and semantics; label/ground-truth roles; split group keys; selected sample IDs and selection digest; full-split sample/file counts and source; license/access state; HyperSpectrum commit; and source query timestamp. The validator must not require wire-format `innerPath`; the ACE handoff mapper converts M0 `inner_path` to ACE Python `inner_path`, whose serialized/report form is `innerPath`.
2. Run the focused tests; expected result: missing script failure.
3. Implement a pure validator plus CLI. Exit 0 and emit canonical JSON for valid scoreable evidence; exit 2 with stable error codes for invalid or non-scoreable evidence.
4. The validator must reject placeholder strings, zeroed SHA-256 values, suffix-inferred or missing materialization, `http` URLs with query credentials, `/home/` and `/data/` source paths, missing license/access declarations, and fabricated ACE-only `adapterDigest`, image digest, backend, or `backendHandle` values.
5. Run `python scripts/check_xas_m0_evidence.py /absolute/path/to/hyper-spectrum/docs/evidence/xas-m0-selection.json`. If it exits non-zero, stop M1 and return to M0; do not seed an ACE board.
6. Run focused tests and the ACE repository's standard lint/type checks.
7. Commit: `test: gate XAS benchmark on real M0 evidence`.

## Task 2: Register the XAS Denoising Research Board

**Files:** `src/ace_leaderboard/seed/data/research_models.json`, `src/ace_leaderboard/seed/data/research_datasets.json`, `src/ace_leaderboard/seed/data/research_assets.json`, `tests/test_research_seed_xas.py`.

1. Write tests that derive the dataset/version/digests from the validated evidence and assert exactly three independent boards: `xas-denoising-zero-shot`, `xas-denoising-one-shot`, and `xas-denoising-few-shot`.
2. Assert the zero-shot board's supported `metrics` list declares `normalized_spectrum_rmse` as the sole primary metric with direction `min`. Declare the secondary keys individually: `normalized_spectrum_rmse_ci_low` (`min`), `normalized_spectrum_rmse_ci_high` (`min`), `spectrum_mae` (`min`), `cosine_similarity` (`max`), `derivative_rmse` (`min`), and `failure_rate` (`min`). Register the exact split/selection manifest, including selected sample IDs and group assignments, as a data `ResearchAsset`; store its digest plus grouped-bootstrap method, confidence level, resample count, grouping key, deterministic split method, selection digest, and a recognized full-split `sample_count` in `ResearchAsset.semantics`. Do not add custom `aggregation` or `splitMetadata` fields to `ResearchDataset`/`ResearchMetric`: the frozen `CamelModel` ignores undeclared extras. Undeclared finite metric keys are merely flagged as `extraMetrics`, and structured values are rejected, so neither path can carry this contract. Do not declare an LCF metric unless the evidence declares composition truth.
3. Assert independent `ResearchModel` entries and model directories exist for `xas-savgol`, `xas-gaussian`, `xas-moving-average`, `xas-pca`, `xas-gaussian-process`, and `xasdenoise`. Each has its own thin adapter and zero-shot/default variant. One/few-shot boards exist with zero runs until M2; they must not inherit zero-shot results.
4. Run the focused seed test; expected result: missing entries.
5. Add dataset, asset, model, variant, and board entries using IDs deterministically derived from the real dataset code/version. Populate all asset hashes and counts from the evidence validator output; never type fake digests into seed files.
6. Set XASDenoise `ResearchModel.status` to `pending_weights` until the weight asset and private distribution policy pass Task 4, and record the blocker in the development evidence. Keep variants/assets schema-valid without invented status fields. Classical models may be `ready` after their model-specific image and adapter tests.
7. Run focused and full ACE seed tests; expected result: pass.
8. Commit: `feat: register XAS denoising research boards`.

## Task 3: Build the Thin ACE Adapter

**Files:** each `harness/docker/research/<model-id>/run.py`, `Dockerfile`, model-specific verification/dependency files, and adapter tests.

1. Write adapter tests with a real-format, tiny, redistribution-safe fixture selected during M0. If no such subset can be committed, materialize it at test time from an approved local test asset; do not substitute a synthetic benchmark input.
2. For every model directory, assert `run.py` uses ACE `entry.parse_args` and `write_metrics`; honors `--max-samples`; fails on missing/partial samples; imports HyperSpectrum at the pinned commit; constructs a local HyperSpectrum `RunPlan`; and emits exactly `{metrics, nSamples, durationS, artifacts}`. Assert XASDenoise also uses `resolve_weights`.
3. Assert each classical model directory contains a buildable `Dockerfile` that installs the pinned HyperSpectrum/runtime dependencies, copies `<model-id>/run.py` to `/opt/acebench/run.py`, and sets `ENTRYPOINT ["python", "/opt/acebench/run.py"]`. Assert each `verify.py` imports every runtime dependency used by its `run.py` and checks the pinned HyperSpectrum commit plus the model/tool implementation digest, but does not require weights.
4. Run adapter tests; expected result: missing adapter failure.
5. Implement each `run.py` as translation only: ACE arguments → model-fixed local HyperSpectrum `RunPlan`/executor → ACE metrics envelope. Do not reimplement axis validation, XAS baselines, or metric formulas, and do not duplicate ACE execution/staging/report logic.
6. Fix tool selection by the ACE model directory and adapter code. Do not depend on a sibling `tool-manifest.json`, because the frozen ACE adapter stages and hashes only `common/entry.py` plus the model-specific `run.py`. Fixed-weight execution must call `resolve_weights`; classical adapters must reject a non-empty weights path to prevent accidental model confusion. If one adapter must receive a variant identity at runtime, stop and propose a separate ACE framework change.
7. Assert any incomplete `PredictionBundle`, failed sample, or missing prediction exits non-zero without publishing `metrics.json`.
8. Build and probe every classical model through ACE so `verify.py` runs inside the built dependency environment, not through bare host Python. Do not pass `--push` in this step:

   ```bash
   uv run acebench-lb research build <model-id> \
     --registry <registry> --verify --json
   ```

   Assert exit 0, `verified: true`, and `verifyDeclared: true`. Inspect the built image through the configured Docker host and assert its entrypoint is `python /opt/acebench/run.py`. Treat the build response's `digest` as a local pre-push identity, not a registry digest.
9. Only after step 8 passes, push the exact returned `<repository>:<tag>` with the configured Docker host. Pull that tag from the registry (preferably on a clean daemon with the same access path Bohr will use), resolve its `RepoDigest`, and record the resulting `<repository>@sha256:...` plus pullability evidence. A failed probe, push, pull, missing `RepoDigest`, or changed digest is a hard gate; do not seed/use the image. If this cannot be automated safely with the frozen CLI, open an ACE prerequisite instead of using combined `--verify --push`.

10. Commit: `feat: add thin XAS denoising research adapters`.

## Task 4: Verify the Fixed Weights and Build the Private Model Image

**Files:** XASDenoise model-specific build/verification declarations and `docs/dev/research/xas/2026-08-08-xas-denoising-m1.md`.

1. On `5090dzr`, inventory `/data2/cgs/model`, `/data2/cgs/dino`, `/data2/cgs/locateanything`, `/data2/cgs/sam3`, and permitted shared XAS paths with read-only `find`/`stat` commands. Do not access `/data/dzr/myw` unless the owner has granted read permission.
2. Locate the exact XASDenoise architecture, checkpoint, upstream commit, implementation/tool digest, and license evidence. Calculate SHA-256 and run the upstream fixed-weight probe on a real-format sample.
3. If source or weight licensing is unknown, keep the image private and set `distribution: private-validation-only` in the development evidence. If the checkpoint is absent or fails the probe, keep the XASDenoise model at `pending_weights`, record the blocker, and continue only with classical models.
4. Rebuild a GPU-capable private image through the supported DeployMaster service/tooling, not the known-outdated `scripts/deploymaster_build.py`. The build declaration installs the pinned HyperSpectrum commit, pinned external source, adapter dependencies, CUDA-compatible PyTorch, and no embedded private dataset.
5. Make the private image pullable by Bohr and invoke it through the explicit, tested `--image <registry-tag>` override. Keep `harness/docker/research/xasdenoise/verify.py` beside the adapter so ACE streams that probe into the resolved image during `research run`; do not treat a host-side probe as image verification.
6. Before changing model status, use ACE's `verify_image()` helper or its exact `docker run --entrypoint sh ... python -` equivalent to stream `verify.py` into `<registry-tag>` and assert the in-image import/commit/implementation-digest probe passes. Then run the same image entrypoint with a deliberately absent checkpoint; expected result: non-zero `missing_weights`, proving random initialization is disabled. After status becomes `ready`, confirm `research run ... --image <registry-tag> --no-record --json` repeats the in-image probe before submission.
7. Record image tag and resolved digest, source commit, weight digest/location class, build task ID, CUDA/PyTorch versions, and probe output in the development document. Do not record tokens or signed URLs.
8. Update only the XASDenoise `ResearchModel.status` from `pending_weights` to `ready` after checksum, licensing gate, pullability, and in-image probe pass. Variants and assets receive no invented status transition. Run the focused seed and adapter tests; expected result: pass.
9. Commit: `docs: record verified XASDenoise image and weights`.

## Task 5: Run the Real Benchmark on the Local 5090

**Files:** `docs/dev/research/xas/evidence/local-run.json`, `docs/dev/research/xas/2026-08-08-xas-denoising-m1.md`.

1. Use `acebench-lb research match` and an explicit tested `research run ... --image <pullable-registry-tag>` for each model with the selected real dataset asset, deterministic manifest, and staged weight asset when required. Record the resolved digest and reuse the same pullable tag/digest pair for Bohr.
2. First run `research run ... --max-samples 8 --no-record --json` for every ready classical model and XASDenoise. Expected result: ACE job status `completed`, `recorded: null`, `metrics: {}`, `partial: null`, and `nSamples: null`. Inspect the retained reports directory/metrics artifact to prove eight accounted samples, explicit axes, finite metrics, and zero silent drops. No smoke run may publish or rank.
3. Run the complete fixed test split with `--max-samples` omitted and default `--record` only after the smoke evidence passes. Any missing sample, changed split digest, NaN metric, unresolved artifact, or sample count different from `ResearchAsset.semantics.sample_count` invalidates the run; only this complete run may be recorded/published.
4. Generate the ACE report and inspect best/worst spectra plus residual plots. Record `nSamples`, primary/secondary metrics, confidence intervals, duration, data/weight/adapter/image digests, and failure count in the redacted evidence JSON.
5. Confirm the fixed-weight variant loads the checkpoint by digest and does not train. Capture a no-gradient/no-optimizer assertion from the adapter/model probe.
6. Validate the redacted evidence JSON with the repository test; expected result: pass.
7. Commit: `test: record local 5090 XAS benchmark evidence`.

## Task 6: Run the Same Contract Through Bohr

**Files:** `docs/dev/research/xas/evidence/bohr-run.json`, `docs/dev/research/xas/2026-08-08-xas-denoising-m1.md`.

1. Confirm `bohr --version` is at least 2.5.17 and that the approved account/queue can access the private registry and staged assets. Use CLI status commands; do not read auth files.
2. Configure the existing ACE backend and submit an unrecorded smoke run with the normal research command; there is no `--backend` flag:

   ```bash
   export ACEBENCH_LB_RESEARCH_BACKEND=bohr-job
   export ACEBENCH_LB_RESEARCH_BOHR_PROJECT_ID=<project-id>
   export ACEBENCH_LB_RESEARCH_BOHR_MACHINE_TYPE=<machine-type>
   uv run acebench-lb research run <model-id> <board-id> --variant <variant-id> \
     --image <pullable-registry-tag> --max-samples 8 --no-record --json
   ```

   Use the same `ExecutionSpec` semantics and immutable inputs as the local run. ACE must stage data, weights, and the model-specific adapter because Bohr does not support bind mounts; HyperSpectrum still executes a local `RunPlan` inside the submitted container.
3. Poll through ACE/Bohr until terminal state. Expected ACE result: job status `completed` with `recorded: null`; retain Bohr's raw `SUCCEEDED` scheduler phase separately in backend evidence. Verify the same eight accounted smoke samples as local, finite metrics, and retrievable artifacts.
4. After the Bohr smoke passes, rerun with `--max-samples` and `--no-record` omitted so the complete fixed split is the only Bohr result ACE records/publishes. Preserve job ID, timestamps, image tag and resolved digest, staged asset digests, ACE status, raw Bohr phase, metrics, and artifact URIs in redacted evidence.
5. If queue, registry, quota, or asset access blocks execution, record the explicit backend blocker; do not replace the Bohr result with the local result.
6. Generate the ACE report from the recorded full-split Bohr run, never from the smoke output.
7. Validate the redacted Bohr evidence JSON with the repository test; expected result: pass.
8. Commit: `test: record Bohr XAS benchmark evidence`.

## Task 7: Prove Local/Bohr Parity and Publish the Zero-Shot Board

**Files:** `scripts/check_xas_run_parity.py`, `tests/test_check_xas_run_parity.py`, `docs/dev/research/xas/evidence/parity.json`, `docs/dev/research/xas/2026-08-08-xas-denoising-m1.md`, relevant seed run counts/status fields.

1. Write failing tests for mismatched data/materialization, split/selection, weight, ACE adapter, HyperSpectrum implementation/tool, image digest, sample IDs, and failure sets; also test classical tolerance `1e-12` and fixed-weight GPU tolerance `1e-6`.
2. Run the focused parity tests; expected result: missing script failure.
3. Implement the parity check over recorded full-split runs only. It must require identical immutable digests, asset materialization, selected sample IDs, and sample/failure sets before comparing metrics. Compare ACE `adapterDigest` separately from HyperSpectrum implementation/tool digests.
4. Run it against local and Bohr evidence; expected result: exit 0 and a redacted `parity.json`. Any mismatch blocks publication.
5. Verify report artifact roles and order match the canonical research contract established by PR #109. Ensure no artifact contains a signed URL, token, raw private path, or unredacted log.
6. Publish/update only the zero-shot board after parity passes. Leave one-shot and few-shot boards empty and ready for M2 derived variants.
7. Run the full ACE test suite and final commands required by the repository, plus `git diff --check`. Also prove that no plan or development command supplies a backend CLI flag, every classical `ResearchModel` has a model-specific `run.py`, and no adapter relies on a sibling runtime tool manifest; expected result: pass.
8. Update the development document with achieved status or exact blocker evidence and links.
9. Commit: `feat: publish reproducible XAS zero-shot board`.

## M1 Completion Gate

M1 is complete only when:

- The board is seeded from a validated real HyperData dataset version and content digest.
- Every classical algorithm is an independent ACE `ResearchModel` with a model-specific thin adapter and has full-test-split quantitative results.
- XASDenoise is either a verified fixed-weight result with model status `ready`, or remains `pending_weights` with an explicit evidence/gate blocker; random initialization never appears.
- Local 5090 and Bohr use the same ACE `ExecutionSpec` semantics and immutable asset/image/model-specific adapter digests; HyperSpectrum uses a local `RunPlan` inside both containers.
- Parity passes within declared tolerances.
- ACE generates the report and zero-shot board without sample dropping.
- All capped smoke runs use `--no-record`; only complete full-split runs are recorded, published, or ranked.
- One-shot/few-shot boards exist but contain no fabricated or zero-shot-copied results.

If XASDenoise remains `pending_weights`, M1 may publish a classical-only zero-shot board, but the milestone report must preserve the blocker and say that the model comparison is incomplete. If Bohr remains operationally blocked, M1 itself is not complete even when the local report exists.
