# HyperSIGMA HSI Denoising Adapter Design

Date: 2026-08-10

Status: approved in conversation for DG-12

## 1. Goal and scope

Add WHU-Sigma/HyperSIGMA as the first real hyperspectral denoising tool in
HyperSpectrum. The increment adds explicit HSI axis semantics, a declarative
tool contract with two pinned denoising checkpoints, a canonical denoising
adapter, synthetic contract tests, reproducible asset verification, and local
real-data smoke evidence.

The implementation does not create an ACE Board, publish benchmark rankings,
train or fine-tune a model, connect to the HyperData Hub data plane, or add
classification, anomaly-detection, or other HyperSIGMA task families. Weights
and the research-use-only dataset remain outside Git and are not redistributed.

## 2. Fixed identities and legal boundaries

The adapter and manifest use these immutable identities:

- Source repository: `https://github.com/WHU-Sigma/HyperSIGMA.git`
- Source commit: `07e9ea24e3072fcb5c3a92a2bcb8185e43b295b9`
- Source license: `Apache-2.0`
- Weight repository: `WHU-Sigma/HyperSIGMA`
- Weight revision: `e0567395fbdfddbae994695baf5fc73358a1ec3c`
- Gaussian checkpoint:
  `Denoising_models/hypersigma_gaussian_noise_model.pth`, 2,266,408,098
  bytes, SHA-256
  `dc101cfe7d462d721eb46395b1621d82103cf4e72166d8591b5619cb2d10e806`
- Complex-noise checkpoint:
  `Denoising_models/hypersigma_complex_noise_model.pth`, 2,266,408,098
  bytes, SHA-256
  `8b1162aae6811af67d287b9271e74154db5448c5e2fd6df953dae5d7807bbe7e`
- Dataset repository: `WHU-Sigma/HyperSIGMA_Datasets`
- Dataset revision: `18ac00c7a98cae281bbdba7f2ed415c1765037d2`
- Dataset subset: `HyperSIGMA_denoising/`
- Dataset use: research evaluation only; no Git inclusion, public mirror, or
  redistribution.

The implementation does not vendor upstream model files. It loads an external,
materialized checkout only after verifying the five files it actually imports:

| File | SHA-256 at the pinned source commit |
| --- | --- |
| `ImageDenoising/models/hypersigma/model.py` | `66c2165b1e04aa7df7c995f3d9ee84a332aa189572364ffb74be8a3b89711c1f` |
| `ImageDenoising/models/hypersigma/Spatial.py` | `b9834e915333c9f7b8c48d818a0a7bf995c6c8966a1f9b1b3299e642bdb2256b` |
| `ImageDenoising/models/hypersigma/Spectral.py` | `c2fbbdcfbf75622c7ecfd88a3ff7c42295f19684e24b6022fd65e1f04050da92` |
| `ImageDenoising/models/hypersigma/Spatial_route.py` | `6b80337dfa94253504936584a85d62b5c81db380ba45ef52bc1a59cf0026883f` |
| `ImageDenoising/models/hypersigma/Spectral_route.py` | `617916efbe01fff98248f7b95bcd53bc9265b14f425aed2f90d0a8ee800931b3` |

Because no upstream source is copied into the repository,
`THIRD_PARTY_NOTICES.md` does not change. If implementation later requires
vendoring, that is a scope change and must update the notice file before merge.

## 3. Architecture

The design keeps domain semantics, manifest semantics, model execution, and
smoke orchestration separate:

1. `hyperspectrum.plugins.hsi` validates HSI physical-axis declarations and
   creates immutable rank-3 `SpectrumSample` values.
2. The tool-manifest contract represents more than one immutable weight asset
   without breaking existing single-asset v1 manifests.
3. `hyperspectrum.adapters.hypersigma` validates the selected source and weight,
   loads the fixed upstream model, translates canonical HSI input to the model
   layout, and restores the original axis order.
4. `hyperspectrum.adapters.hypersigma_smoke` loads explicitly selected MATLAB
   keys, builds denoising pairs, calls the existing unified evaluator, and emits
   JSON evidence.

The adapter never downloads assets. The caller materializes the fixed source,
one selected checkpoint, and one dataset patch before invocation.

## 4. HSI axis semantics

An HSI cube declares exactly three physical axes:

- exactly one spectral axis named `band` or `wavelength`;
- one spatial axis named `y`;
- one spatial axis named `x`.

`band` uses explicit band-index coordinates and an index unit. `wavelength`
uses explicit physical coordinates and a non-index unit. Axis names are unique,
coordinates remain immutable, and the signal shape must already agree with the
declared axes through `SpectrumSample`.

The helper accepts any permutation of the three explicitly named axes. It
derives a named permutation to the upstream `(band, y, x)` layout and records
that permutation. The output is transposed back to the exact original order.
It never identifies a spectral axis from dimension length, flattens the cube,
interpolates coordinates, fills values, crops, or silently creates patches.

HyperSIGMA additionally requires 191 spectral elements and a 64 by 64 spatial
patch. Other extents are rejected before Torch execution. Full 192 by 192 test
cases are downloaded and audited as required, but the real inference smoke uses
the supplied 64 by 64 `Patch_Cases` and performs no hidden tiling.

## 5. Multi-asset tool-manifest contract

`hyperspectrum-tool/v1` receives an additive `weights.assets` form. Each asset
contains a unique non-blank variant name plus `asset_id`, HTTPS `source_url`,
basename `filename`, positive `size_bytes`, lowercase SHA-256 `digest`, and
non-blank `license`.

Backward compatibility is explicit:

- existing v1 manifests may continue using the current singular asset fields;
- a manifest may use either the complete singular form or a non-empty
  `weights.assets` list, never both;
- `state: present` requires one complete representation;
- `not-required` and `required-missing` permit neither representation;
- asset variant names and immutable identities must be unique.

The public schema, packaged schema, typed Pydantic model, source YAML, and
packaged YAML remain semantically identical. HyperSIGMA declares the two
variants `gaussian` and `complex`. `allow_download` remains exactly `false`.

## 6. Adapter and fixed upstream construction

`HyperSIGMADenoiseAdapter` declares:

- axis ranks: `(3,)`;
- representations: `("dense",)`;
- channel counts: `(1,)`;
- required normalization: `per_spectrum_range`;
- native-unit recovery: `True`.

Both checkpoints use the same construction recovered from
`ImageDenoising/models/hypersigma/model.py`:

- input and output channels: 191;
- patch size: 64 by 64;
- spatial encoder: patch size 2, embed dimension 768, depth 12, 12 heads,
  output indices 3/5/7/11, interval 3, 8 points;
- spectral encoder: 100 tokens, embed dimension 768, depth 12, 12 heads,
  output index 11, interval 3, 8 points;
- spatial adapter: patch size 1, embed dimension 768, depth 12, output index 3;
- spectral adapter: 100 tokens, embed dimension 128, four executed blocks from
  the upstream adapter implementation, output index 3;
- reconstruction: 382-to-191 and 191-to-191 three-by-three convolutions.

The upstream constructor invokes two absolute pretraining paths. The loader
isolates the imported modules, serializes model construction with a lock,
temporarily replaces only the two `init_weights` methods with no-ops, constructs
the final architecture, and restores the original methods in a `finally`
block. It then loads the complete denoising checkpoint strictly. This avoids
unavailable private paths without copying or modifying upstream source and
without substituting random weights for the final checkpoint.

Optional Torch, timm, and einops imports are lazy. Import failure reports the
missing HyperSIGMA runtime dependency and does not affect installation or use of
the existing XAS paths.

## 7. Safe source and checkpoint loading

Source verification hashes every required file before importing any of them.
Missing files, additional path traversal, or digest drift fail closed.

Checkpoint loading performs these steps in order:

1. Resolve the requested variant to one exact `WeightAssetContract`.
2. Open the checkpoint and retain the file descriptor.
3. Stream the bytes through SHA-256 while counting the exact size.
4. Reject any mismatch before deserialization.
5. Rewind the same descriptor and call `torch.load` with
   `map_location="cpu"` and `weights_only=True`.
6. Require a mapping with the exact top-level `net` state dictionary.
7. Construct the fixed model, call `load_state_dict(..., strict=True)`, disable
   gradients, enter evaluation mode, and move to the resolved CPU or CUDA
   device.
8. Run prediction only under `torch.inference_mode()`.

The runtime checks input and output shape, mask preservation, finite valid
values, normalization-state digest, and parameter non-mutation. An explicit
CUDA request fails when CUDA is unavailable; `auto` may select CPU.

## 8. Data flow

The `hyperspectrum.adapters.hypersigma_smoke` module takes explicit paths for
source root, checkpoint, and MATLAB file, plus the explicit weight variant and
spectral-axis declaration. It loads only keys `input` and `gt`, rejects missing
or ambiguous data, and does not infer axis meaning from their shape.

For each patch it creates noisy and clean `SpectrumSample` values with identical
identity, axes, mask, unit, and split assignment. The existing evaluator then:

1. checks pair and split identity;
2. preflights model capabilities;
3. applies reversible `per_spectrum_range` normalization outside the model;
4. creates `CanonicalDenoisingInput`;
5. invokes the HyperSIGMA adapter;
6. validates the unchanged canonical output contract;
7. computes normalized RMSE/MAE and exactly inverted native RMSE/MAE;
8. emits coverage and per-sample evidence.

Case1 uses the Gaussian checkpoint. Case5 uses the complex-noise checkpoint.
Each smoke is diagnostic research evidence, not an ACE result or ranking.

## 9. Failure behavior

Rank other than three and unsupported complex or sparse representations return
the existing structured `CompatibilityIssue` codes `axis_rank` and
`representation` before normalization or Torch import.

Axis-name ambiguity, missing axes, unsupported extents, partial masks, source
drift, weight mismatch, malformed checkpoint structure, strict-load mismatch,
device failure, shape drift, non-finite output, or parameter mutation are
explicit failures. No failure path may flatten, reinterpret, interpolate,
repair, substitute assets, initialize a fallback model, or emit a successful
metric record.

## 10. Testing

Synthetic fixtures are small, deterministic, and explicitly marked synthetic.
They exercise:

- valid `band/y/x` and `wavelength/y/x` declarations;
- named axis permutations and exact output-order restoration;
- rejection of missing, duplicate, or ambiguous axes;
- structured incompatibility for rank-not-three, complex, and sparse samples;
- 191-band and 64-by-64 extent checks without shape-based axis guessing;
- both exact weight contracts and variant selection;
- streamed size/SHA verification before fake Torch loading;
- source-file verification before import;
- exact `net` extraction, strict loading, eval/inference behavior, device
  handling, and output validation through injected fake boundaries;
- old singular v1 weight manifests plus new multi-asset manifests;
- exact HyperSIGMA YAML facts and source/package resource parity;
- the existing unified evaluator with an identity fake model.

Verification runs in this order:

1. Record the clean `origin/main` pytest, Ruff, and mypy baseline.
2. Run focused HSI plugin, registry, adapter, and smoke-loader tests.
3. Run the full pytest suite and compare failures with the baseline.
4. Run Ruff, strict mypy, schema/resource parity checks, packaging tests, and a
   fresh wheel/installed-console smoke.
5. Run `git diff --check` and inspect the complete diff.
6. Run one real Case1/Gaussian patch and one real Case5/complex patch.

## 11. Asset download and evidence

Downloads use the pinned Hugging Face revisions and concurrency no greater than
four. The local cache lives outside the repository checkout. The two checkpoint
files are verified against the delegated sizes and SHA-256 values byte for
byte.

The complete `HyperSIGMA_denoising/` subset is downloaded. A generated evidence
table records every relative path, byte count, and local SHA-256, including six
`Testing/Cases/*/test.mat` files and all `Testing/Patch_Cases` files. Only the
text table, commands, and smoke output are copied into the development document.

Before commit, tracked and untracked file checks must prove that no checkpoint,
MATLAB dataset file, cache directory, or derived prediction cube is inside Git.

## 12. Documentation and handoff

`docs/dev/2026-08-10-hypersigma-hsi-denoise.md` records the goal, a one-line
description of every changed file, upstream configuration sources, decisions,
baseline/final verification, exact asset checks, smoke environment and metrics,
known limits, and generalization notes.

The PR targets `hyper-instrument/hyper-spectrum:main` from the user's fork and a
branch named `delegation/DG-12-hypersigma-hsi-denoise`. It contains DG-12 in the
title, does not use hyphenated identifiers for other delegations, and reports
the PR through the authorized DG-12 endpoint immediately after creation.

The latest packet reintroduced the literal `{{HUB_PAT}}` placeholder in generic
sections while the task-specific scope and user instruction explicitly require
no Hub data-plane access. The implementation therefore skips that chapter and
records the template inconsistency as a generalization note. A feedback ticket
is sent only if the user separately authorizes the feedback endpoint.

## 13. Rollback

The source changes are additive except for the backward-compatible manifest
extension. Reverting the HSI plugin, HyperSIGMA adapter and smoke runner, tool
resources, tests, documentation, and multi-asset fields restores the preceding
behavior. External source, weights, datasets, and smoke outputs are untracked
local cache content and can be removed independently without changing Git.

## 14. Closed decisions

- Use a thin adapter with a verified external source checkout; do not vendor or
  rewrite the upstream model.
- Support both Gaussian and complex-noise checkpoints in one v1 tool manifest
  through an additive, backward-compatible multi-asset form.
- Permit only explicit named-axis transposition and restore the input order.
- Restrict inference to 191-band, 64-by-64 patches; do not implement tiling.
- Run real diagnostic smoke for both checkpoints on Case1 and Case5 patches.
- Keep every weight and dataset byte outside Git and prohibit redistribution.
- Skip HyperData Hub login and all formal ACE Board work.

There are no open implementation decisions in this design.
