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
.venv/bin/hyperspectrum run plan --task-file TASK.json --candidate-file CANDIDATE.json --verdict-file VERDICT.json --tool-id savgol --output-directory RUN_DIR --sample-id SAMPLE_ID --max-samples 8 --dry-run --json
.venv/bin/hyperspectrum run local --plan-file PLAN.json --source-npz NOISY_INPUT.npz --sample-id SAMPLE_ID --json
.venv/bin/hyperspectrum evidence validate --evidence-file XAS_M0_SELECTION.json --json
```

## Wire compatibility

- New planning emits `hyperspectrum-run-plan/v2`. Every v2 plan binds an exact,
  ordered sample selection; unbound v1 plan files are refused with instructions
  to create a new plan using repeated `--sample-id` options. There is no implicit
  v1 migration because reconstructing a missing selection would change run
  identity.
- `hyperspectrum-prediction/v1` remains available for existing constructors and
  keeps its original non-empty provenance contract. It cannot satisfy the XAS
  M0 success handoff. The local writer emits `hyperspectrum-prediction/v2`, which
  requires formatted execution digests, non-blank dataset identity, and a v2
  selection-bound plan identity.
- Successful M0 evidence must pass `evidence validate`. The packaged semantic
  validator runs JSON Schema first, then structural redaction, locator policy,
  provenance equality, asset mapping, and count-order checks.

Review the dry-run first. Local execution requires a newly generated non-dry
`result.plan` saved as `PLAN.json`, an absent `RUN_DIR`, noisy inference inputs,
and explicit user authorization.
