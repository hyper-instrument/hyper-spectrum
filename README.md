# HyperSpectrum

HyperSpectrum is a planned agent-ready runtime for scientific spectroscopy and
spectral imaging. The initial vertical slice is an end-to-end XAS workflow that
discovers versioned data in HyperData, runs fixed-weight and classical tools on
local GPU or Bohr, and publishes quantitative results through ACE Benchmark.

The approved design is documented in
[`docs/superpowers/specs/2026-08-08-hyper-spectrum-xas-agent-runtime-design.md`](docs/superpowers/specs/2026-08-08-hyper-spectrum-xas-agent-runtime-design.md).

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
```

Review the dry-run first. Local execution requires a newly generated non-dry
`result.plan` saved as `PLAN.json`, an absent `RUN_DIR`, noisy inference inputs,
and explicit user authorization.
