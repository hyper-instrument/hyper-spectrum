# XASDenoise runtime adapter evidence

This increment replaces the former unknown-license/missing-weight placeholder
with a thin production adapter. It does not vendor or download checkpoint bytes.

## Pinned identities

- Source: `https://github.com/tomasaidukas/XASDenoise.git` at commit
  `bda749ee956f9e02acc6995f238d759682ee2ca8`; code license MIT.
- Weight record: Zenodo 17434349, DOI `10.5281/zenodo.17434349`; asset license
  CC-BY-4.0.
- Filename:
  `xas_denoiser_model_noise2noise_nonuniformly_sampled_notnormalized.pth`.
- Size: 780409 bytes.
- SHA-256:
  `09620ee9ea0c96585f534d76ce42aa72edf2cf71e481f5737e43e93116e24160`.
- Upstream architecture/configuration: four encoder and decoder levels, kernel
  size nine, bias disabled, 16-point reflect padding/crop, nonuniformly sampled
  noise-to-noise checkpoint, and checkpoint normalization method null.

The registry contains those exact identities and an HTTPS source locator, but
availability remains fail-closed until the caller supplies exactly one local
file whose size and SHA-256 match. The runtime never downloads a replacement and
never falls back to random initialization.

## Scientific and runtime semantics

Canonical normalization is explicitly `identity_raw`; no per-spectrum range
normalization is applied. Before model inference, the adapter reproduces the
upstream symmetric-tanh step-baseline fit using the maximum first derivative as
the edge guess and five percent of the energy span as the width guess. After
inference it adds the exact same fitted baseline. Each prediction records the
versioned fit parameters, baseline digest, inverse operation, null checkpoint
normalization, canonical normalization, and native-output semantics.

Checkpoint loading uses `torch.load(..., map_location="cpu", weights_only=True)`
after byte verification. State loading is strict. The model enters evaluation
mode and runs under `torch.inference_mode()`. Input axis order, shape, finite
values, output shape/finite values, and unchanged parameter bytes are checked.
An explicit CUDA request fails when CUDA is unavailable; only `device=auto` may
select CPU. PredictionBundle v3 and `run.json` retain the existing model, tool,
implementation, weight, environment, source-dataset, source-content-manifest,
benchmark-asset, selection, and plan identities and add requested/actual device,
PyTorch version, and per-sample preprocessing evidence.

## One adapter across backends

Planning and execution both receive the authorized asset explicitly:

```bash
hyperspectrum run plan ... --tool-id xasdenoise --weight-file MODEL.pth --device auto --json
hyperspectrum run local ... --weight-file MODEL.pth --json
```

Local CPU/GPU execution calls
`hyperspectrum.adapters.xasdenoise:denoise_spectra`. OCI and Bohr stage the same
immutable benchmark, plan, source closure, and checkpoint and invoke that same
entrypoint; no backend-specific scientific adapter is permitted. HyperSpectrum
emits predictions and provenance only. ACE remains responsible for grouped
metrics, reports, boards, and local/Bohr parity decisions.

## Verification status and remaining gate

The deterministic test suite, static checks, source/wheel packaging, installed
wheel verification command, and installed tool-metadata query pass without real
weights or benchmark data. A prior read-only 5090 audit established the official
checkpoint byte identity and exercised the upstream architecture only. It was
not an end-to-end smoke of this adapter. The required adapter E2E CUDA smoke was
attempted twice on 2026-08-09, but SSH failed before authentication with `No
route to host`. Therefore local-5090, OCI, and Bohr result/parity evidence remain
explicitly unverified gates; no runtime-success or model-quality claim is made.
