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
   declared ground-truth role, safe split keys, and an available `savgol` match.
   Treat blocked tools as evidence, never as permission to fetch weights, install,
   adapt an adapter, or substitute code.

## Plan First

Confirm the requested output directory does not exist; choose a new path rather
than deleting or overwriting it. Create a non-executable review plan:

```bash
.venv/bin/hyperspectrum run plan \
  --task-file TASK.json \
  --candidate-file CANDIDATE.json \
  --verdict-file VERDICT.json \
  --tool-id savgol \
  --output-directory RUN_DIR \
  --max-samples 8 \
  --dry-run \
  --json
```

Preserve the single envelope, plan digest, dataset/tool/implementation/weight/
environment digests, resource budget, parameters, sample limit, destination,
warnings, and blockers as evidence.

Ground truth is readiness and later ACE-evaluation evidence only. Never read clean
arrays to choose parameters, samples, preprocessing, or outputs. Keep the declared
SavGol parameters fixed; do not tune, fit, score, rank, or select favorable cases.

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

Repeat `--sample-id` for each predeclared sample. Supply noisy inference inputs;
do not materialize clean targets for HyperSpectrum. The inference-only NPZ must
contain exactly `energy`, `noisy`, `sample_ids`, `group_ids`, and `energy_unit`.
Reject every extra member, including metadata, labels, targets, ground truth,
metrics, and scores. Return the unaltered envelope and artifact/digest evidence.
Exit `2` means invalid/not ready, `3` auth/connection, `4` missing asset/tool,
and `5` execution failure: report and stop without fallback.
