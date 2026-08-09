# XASDenoise runtime adapter evidence

This increment pins and verifies a thin adapter, but production use on the raw
benchmark remains blocked by an unverified scientific input contract. It does
not vendor or download checkpoint bytes.

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

The registry contains those exact identities and an HTTPS source locator. A
matching local checkpoint resolves only the weight gate; it cannot resolve the
independent `input_contract_unverified` gate. The runtime never downloads a
replacement and never falls back to random initialization.

## Scientific and runtime semantics

Pinned upstream and Zenodo evidence shows the checkpoint consumes spectra that
were already normalized using spectrum-specific pre-edge and post-edge fits.
Only then does the upstream pipeline fit and subtract its symmetric-tanh step.
The Zenodo HDF5 metadata marks spectra normalized and carries differing E0,
pre/post windows, and V/V fit metadata per spectrum. The raw HyperSpectrum
benchmark carries none of those fitted states. Consequently
`normalization_method=None` is not evidence for raw compatibility, and the
adapter validation fails unconditionally with `input_contract_unverified`.
Neither a mutable method label nor a digest detached from a typed state proves
that preprocessing occurred. The pure step transform remains available only for
architecture and checkpoint probes in a known normalized numeric space; it does
not authorize production inference or claim raw native-unit recovery.

Checkpoint loading uses `torch.load(io.BytesIO(verified_bytes),
map_location="cpu", weights_only=True)`. The same immutable bytes are hashed and
loaded, eliminating pathname swaps. State loading is strict. The model enters
evaluation mode and runs under `torch.inference_mode()`. Input axis order, shape, finite
values, output shape/finite values, and unchanged parameter bytes are checked.
An explicit CUDA request fails when CUDA is unavailable; only `device=auto` may
select CPU. PredictionBundle v3 and `run.json` retain the existing model, tool,
implementation, weight, environment, source-dataset, source-content-manifest,
benchmark-asset, selection, and plan identities. XAS execution IDs additionally
bind requested and resolved device, CPU/CUDA backend, PyTorch/CUDA/cuDNN
versions, and the digest of the bytes actually loaded. Batch runtime evidence is
retained even when every sample fails. Traditional baseline run IDs are unchanged.

## One adapter across backends

XASDenoise planning is currently expected to fail closed:

```bash
hyperspectrum run plan ... --tool-id xasdenoise --weight-file MODEL.pth --device auto --json
# unavailable: input_contract_unverified
```

When a structured, sample-bound normalization artifact becomes available,
local CPU/GPU execution calls
`hyperspectrum.adapters.xasdenoise:denoise_spectra`. OCI and
Bohr must stage the same
immutable benchmark, plan, source closure, and checkpoint and invoke that same
entrypoint; no backend-specific scientific adapter is permitted. HyperSpectrum
emits predictions and provenance only. ACE remains responsible for grouped
metrics, reports, boards, and local/Bohr parity decisions.

## Verification status and remaining gate

The deterministic test suite, static checks, source/wheel packaging, installed
wheel verification command, and installed tool-metadata query exercise the
fail-closed contract without real inference. A prior read-only 5090 audit
established the official checkpoint byte identity and exercised the upstream
architecture only. It was not an end-to-end smoke of this adapter. A fresh
current-source probe on 2026-08-09 confirmed that routing and CUDA were available
(PyTorch reported CUDA 12.8), but the exact checkpoint was absent from the
permitted/readable staging roots, the host Python lacked SciPy, and current-source
transfer was not authorized. No checkpoint was downloaded and the prior smoke
was not reused as current evidence. Therefore local-5090, OCI, and Bohr
result/parity evidence remain explicitly unverified gates; no runtime-success or
model-quality claim is made.
