# HyperSpectrum

HyperSpectrum is a planned agent-ready runtime for scientific spectroscopy and
spectral imaging. The initial vertical slice is an end-to-end XAS workflow that
discovers versioned data in HyperData, runs fixed-weight and classical tools on
local GPU or Bohr, and publishes quantitative results through ACE Benchmark.

The approved design is documented in
[`docs/superpowers/specs/2026-08-08-hyper-spectrum-xas-agent-runtime-design.md`](docs/superpowers/specs/2026-08-08-hyper-spectrum-xas-agent-runtime-design.md).

## Reusable spectral denoising contracts

`hyperspectrum.denoising` provides the model-independent M0 contract for XAS/
XANES/EXAFS, Raman/IR, NMR, mass spectrometry, EELS, and hyperspectral data.
`SpectrumSample` keeps axes, units, masks, channel labels, sparse/complex
representation, metadata, and provenance explicit. It never infers a wavelength
or energy axis from shape and never flattens spectral images for a 1-D model.

Normalization stays outside the denoising model. Per-spectrum transforms persist
their sample-bound parameters; dataset-fitted transforms accept only samples
bound to the train partition of a digest-pinned `SplitManifest` and persist the
manifest digest, ordered training sample/group IDs, training-data digest, and
parameters. Synthetic noise is likewise injected in its declared native signal
or count domain before normalization with a fixed seed and provenance.

Models consume `CanonicalDenoisingInput` and declare `ModelCapabilities` for
axis rank, representation, channels, required normalization, and native-unit
recovery. The same compatible model can therefore run across modality suites.
The evaluator emits normalized-space and recovered native-space RMSE/MAE,
per-modality means, equal-modality macro means for dimensionless normalized
metrics, and coverage. Native metrics are never averaged across incompatible
physical units. Incompatible complex, sparse, multi-channel, or multi-axis
samples are structured skips and are never converted into zero scores.

The detailed contract is documented in
[`docs/superpowers/specs/2026-08-09-unified-spectral-denoising-design.md`](docs/superpowers/specs/2026-08-09-unified-spectral-denoising-design.md).

### Official XASDenoise adapter

The adapter implementation pins upstream XASDenoise commit
`bda749ee956f9e02acc6995f238d759682ee2ca8` (MIT) and the CC-BY-4.0 Zenodo
17434349 checkpoint
`xas_denoiser_model_noise2noise_nonuniformly_sampled_notnormalized.pth`
(780409 bytes, SHA-256
`09620ee9ea0c96585f534d76ce42aa72edf2cf71e481f5737e43e93116e24160`).
HyperSpectrum never downloads it. Supply the already authorized, mounted file to
both planning and execution with one `--weight-file PATH` option.

The checkpoint's `normalization_method: null` disables only model-internal
scaling. The pinned Zenodo training/test spectra were already normalized with
per-spectrum pre-edge/post-edge fits before the symmetric-tanh step was removed.
The raw benchmark does not retain the upstream E0 values, fit windows, fitted
curves, or an exact native-unit inverse. XASDenoise therefore remains
fail-closed with `input_contract_unverified`, even when the mounted checkpoint
bytes pass verification. Do not plan or execute it on raw `ketek/i0` ratios.
The `denoise_spectra` entrypoint and runtime-identity contract are retained for
future inputs that carry the evidenced upstream normalization state.

## Agent CLI

HyperSpectrum owns read-only catalog discovery, evidence-backed task and tool
selection, immutable run planning, and guarded local prediction. It never reads
credential/profile files, downloads missing assets, adapts or trains tools, or
overwrites an existing run directory. HyperSpectrum local runs emit predictions
and provenance only; ACE owns formal evaluation, reports, rankings, and
leaderboards.

Agent mode writes exactly one `hyperspectrum-cli/v1` JSON envelope to stdout;
warnings and diagnostics go to stderr. Exit codes are `0` success, `2`
invalid/not ready, `3` authentication/connection, `4` missing asset/tool, and `5`
execution failure.

The first XAS story uses these commands:

```bash
.venv/bin/hyperspectrum doctor --json
.venv/bin/hyperspectrum data discover --modality xas --profile volcano --json
.venv/bin/hyperspectrum task recommend --candidate-file CANDIDATE.json --json
.venv/bin/hyperspectrum tools match --task xas-denoising --json
.venv/bin/hyperspectrum data materialize-xanes-spec --source-root SOURCE_DIR --source-declaration-file SOURCE_DECLARATION.json --output-directory BENCHMARK_DIR --dose-fraction 0.25 --global-seed 0 --json
.venv/bin/hyperspectrum run plan --task-file TASK.json --candidate-file CANDIDATE.json --verdict-file VERDICT.json --benchmark-manifest-file BENCHMARK_DIR/manifest.json --tool-id savgol --output-directory RUN_DIR --sample-id SAMPLE_ID --max-samples 8 --dry-run --json
.venv/bin/hyperspectrum run local --plan-file PLAN.json --source-npz BENCHMARK_DIR/benchmark.npz --sample-id SAMPLE_ID --json
.venv/bin/hyperspectrum evidence validate --evidence-file XAS_M0_SELECTION.json --json
```

For the fixed-weight adapter, replace `--tool-id savgol` with
`--tool-id xasdenoise`, add `--weight-file MODEL.pth` to both `run plan` and
`run local`, and optionally request `--device cpu`, `--device cuda:N`, or the
explicit fallback policy `--device auto` during planning. An explicit CUDA
request fails when CUDA is unavailable; only `auto` may select CPU.

## Wire compatibility

- New planning emits `hyperspectrum-run-plan/v3`. Every v3 plan binds an exact,
  ordered sample selection plus separate source-dataset, verified-content, and
  derived-asset SHA-256 identities. Planning validates the complete benchmark
  manifest, recomputes both source chains, and binds the declared manifest to
  the catalog digest. V3 execution accepts only the canonical 16-key benchmark
  NPZ. Unbound v1 plan files are refused; v2 plans
  remain readable/executable for existing artifacts only. There is no implicit
  migration because reconstructing missing selection or digest semantics would
  change run identity.
- `hyperspectrum-prediction/v1` remains available for existing constructors and
  keeps its original non-empty provenance contract. It cannot satisfy the XAS
  M0 v2 success handoff. New local writes emit `hyperspectrum-prediction/v3`,
  which requires the three explicit data identity classes, formatted execution
  digests, non-blank dataset identity, and a v3 selection-bound plan identity.
  Prediction v2 remains paired with legacy plan v2 execution.
- Successful M0 evidence must pass `evidence validate`. The packaged semantic
  validator runs JSON Schema first, then structural redaction, locator policy,
  provenance equality, asset mapping, and count-order checks.

Review the dry-run first. Local execution requires a newly generated non-dry
`result.plan` saved as `PLAN.json`, an absent `RUN_DIR`, noisy inference inputs,
and explicit user authorization.
