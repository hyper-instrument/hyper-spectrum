---
name: run-xas-baseline
description: Use when an agent needs to prepare or execute a reproducible local XAS denoising baseline from explicit candidate, verdict, task, and source artifacts.
---

# Run an XAS Baseline

Use HyperSpectrum only for guarded planning and prediction. ACE owns formal
scoring, reports, and leaderboards.

## Preflight and Selection

1. Run `.venv/bin/hyperspectrum doctor --json`. Stop if `result.ready` is false.
   Never inspect credential/profile files or install/download a missing component.
2. Confirm the candidate and task without opening spectral assets:

   ```bash
   .venv/bin/hyperspectrum task recommend --candidate-file CANDIDATE.json --json
   .venv/bin/hyperspectrum tools match --task xas-denoising --json
   ```

   Require an exact `scoreable` denoising verdict, pinned dataset version/digest,
   declared ground-truth role, and safe split keys. `savgol` is the unweighted
   match. `xasdenoise` is the registered fixed-weight option: its blocked record
   must expose upstream commit
   `bda749ee956f9e02acc6995f238d759682ee2ca8`, MIT code, Zenodo asset
   `zenodo-17434349`, CC-BY-4.0 weights, size 780409, and SHA-256
   `09620ee9ea0c96585f534d76ce42aa72edf2cf71e481f5737e43e93116e24160`.
   `weights-unverified` is resolved only by an explicitly authorized, already
   mounted matching file. Never fetch weights, install, adapt an adapter, or
   substitute code.

## Plan First

Confirm the requested output directory does not exist; choose a new path rather
than deleting or overwriting it. The benchmark manifest must identify three
separate SHA-256 classes: the source dataset identity, the verified source
content manifest, and the derived benchmark NPZ bytes. Create a non-executable
review plan:

```bash
.venv/bin/hyperspectrum run plan \
  --task-file TASK.json \
  --candidate-file CANDIDATE.json \
  --verdict-file VERDICT.json \
  --benchmark-manifest-file BENCHMARK_DIR/manifest.json \
  --tool-id savgol \
  --output-directory RUN_DIR \
  --sample-id SAMPLE_ID \
  --max-samples 8 \
  --dry-run \
  --json
```

Preserve the single envelope, plan digest, all three data identity digests,
tool/implementation/weight/environment digests, resource budget, parameters,
sample limit, destination, warnings, and blockers as evidence. Never assert
that the derived asset SHA equals either source identity.

Repeat `--sample-id` to bind every predeclared sample ID during planning, in the
exact deterministic execution order. Use the identical ordered IDs for dry-run,
the executable plan, and `run local`; changing a member or its order requires a
new `hyperspectrum-run-plan/v3` and plan digest. Never execute or migrate a v1
plan: it cannot prove a bound sample selection. v2 remains executable only for
existing artifacts. Re-plan with repeated
`--sample-id` options instead.

Ground truth is readiness and later ACE-evaluation evidence only. Never read clean
arrays to choose parameters, samples, preprocessing, or outputs. Keep the declared
SavGol parameters fixed; do not tune, fit, score, rank, or select favorable cases.

For XASDenoise, use `--tool-id xasdenoise --weight-file MODEL.pth` for both the
dry and executable plans and retain the same option for `run local`. The mounted
file must match the declared size and SHA-256 before any model module or PyTorch
checkpoint loader runs. Choose `--device cpu`, `--device cuda:N`, or the explicit
`--device auto` fallback policy. The checkpoint normalization is null, so require
canonical `identity_raw`; never substitute per-spectrum range normalization.
Preserve the recorded `hyperspectrum-xasdenoise-step-baseline/v1` fit and inverse
state. CPU, local GPU, OCI, and Bohr must all call the same
`hyperspectrum.adapters.xasdenoise:denoise_spectra` entrypoint.

## Authorization Gate

Stop after the dry-run unless the user explicitly authorizes local execution.
Also stop separately before any external/remote execution, adapter or tool
adaptation, download, installation, training/fine-tuning, upload, publication, or
ACE handoff. A request to plan, benchmark, or “make it runnable” is not permission
for those actions.

After explicit local-run authorization, re-check that `RUN_DIR` is absent, repeat
`run plan` without `--dry-run`, and save only `result.plan` as `PLAN.json`. Then run:

```bash
.venv/bin/hyperspectrum run local \
  --plan-file PLAN.json \
  --source-npz NOISY_INPUT.npz \
  --sample-id SAMPLE_ID \
  --json
```

Add the exact same `--weight-file MODEL.pth` for an authorized XASDenoise run.

Repeat `--sample-id` for each predeclared sample. Supply noisy inference inputs;
do not inspect targets to choose parameters or samples. A legacy v2
inference-only NPZ must contain exactly `energy`, `noisy`, `sample_ids`,
`group_ids`, and `energy_unit`. A v3 run requires the canonical XANES benchmark
NPZ with the exact 16-key materializer contract recorded in its fully validated
manifest; the executor exposes
only energy, noisy signal, IDs, groups, and unit to model code. Its target must
be labeled `pseudo-clean frozen measurement` and never physical noiseless ground
truth. Reject every unrecognized extra member, label, metric, or score. Return
the unaltered envelope and artifact/digest evidence.
Exit `2` means invalid/not ready, `3` auth/connection, `4` missing asset/tool,
and `5` execution failure: report and stop without fallback.
