# Unified Spectral Denoising Contract Design

Date: 2026-08-09

Status: approved for the parallel M0 implementation by the user's explicit
request to implement a reusable multi-modality denoising path.

## Scope

This increment adds a small, model-independent denoising kernel for XAS/XANES/
EXAFS, Raman/IR, NMR, mass spectrometry, EELS, and hyperspectral data. It does
not train a model, stage remote data, publish an ACE board, or change the
existing XAS operational evidence.

The kernel must let one compatible denoising model run against more than one
modality suite. Modality parsing, normalization, physical-unit recovery, and
evaluation remain outside the model. Unsupported complex, sparse, multi-channel,
or multi-axis inputs are never flattened or coerced silently.

## Data contract

`SpectrumSample` is an immutable in-memory value used after a HyperData artifact
has been materialized. It complements rather than replaces `ObservationBundle`,
which remains the external manifest/reference contract.

Every sample declares:

- sample and group identity, modality, representation, and signal unit;
- ordered named axes with immutable coordinates, units, and direction;
- an immutable dense/complex/sparse signal and same-shaped validity mask;
- optional channel labels for a leading channel dimension;
- immutable JSON metadata and provenance.

Dense signals have exactly one dimension per declared axis. A leading channel
dimension is permitted only when channel labels are present. Sparse peaks use a
single coordinate axis and a one-dimensional value vector. Complex values are
accepted only for the explicit `complex` representation. A 2D EELS signal and a
3D hyperspectral cube therefore remain 2D/3D; an algorithm that supports only
rank one receives a structured incompatibility instead of a flattened array.

## Normalization

The registry exposes named `NormalizationDefinition` objects with supported
modalities/representations, fit scope, invertibility, and implementation.

- A `per_sample` transformation derives deterministic parameters from one noisy
  input and stores them in a sample-bound `NormalizationState`.
- A `dataset_fitted` transformation can be fitted only through
  `fit_normalizer(..., manifest=...)` when every supplied sample is bound to the
  manifest's train partition. Fitting a `val` or `test` assignment is rejected.
  Its state records the complete split-manifest digest, ordered training sample
  and group IDs, a canonical training-data digest, signal unit, modality, and
  fitted parameters. Evaluation requires the same manifest digest and rejects
  held-out sample/group overlap.
- Applying a state checks modality, representation, signal unit, and sample
  binding. The state, not a registry default, is the evaluation authority.
- Each definition declares `exact`, `approximate`, or `none` inverse support.
  Native/physical-space metrics require an exact inverse and original unit;
  otherwise evaluation fails closed for that metric.

The M0 registry supplies reversible range scaling for real dense spectra,
TIC scaling for sparse mass spectra, phase-preserving RMS scaling for complex
NMR, and train-fitted global standardization for compatible real arrays.

## Noise

Synthetic noise is injected before normalization in a declared native
signal/count space.
Each corruption records algorithm, numeric parameters, seed, source unit,
representation, and a digest of the clean input. Gaussian signal noise supports
finite real dense arrays. Poisson count noise supports finite non-negative real
signals with an explicit count/counts unit. Unsupported complex, sparse-Gaussian,
or malformed inputs are rejected rather than converted. Equal inputs plus equal
seed and parameters produce equal outputs and provenance.

## Reusable model and evaluator contract

`ModelCapabilities` declares accepted axis ranks, representations, channel
counts, required normalization, and whether native-unit recovery is supported.
It intentionally does not bind a model to one modality. A model receives only a
`CanonicalDenoisingInput` and returns a `CanonicalDenoisingOutput`; both retain
sample identity, shape, axis coordinates/names/units/directions, mask, and
normalization-state digest.

The evaluator owns the following flow:

1. validate digest-bound split membership plus noisy/clean identity, axes, mask,
   shape, and native unit;
2. preflight raw rank/representation/channels/normalization compatibility and
   return a typed skip before any data-dependent transform;
3. normalize outside the model and produce canonical input;
4. validate model output without reshaping or repairing it;
5. score RMSE/MAE in normalized space;
6. when exact inversion is declared, restore native units and score native RMSE/
   MAE;
7. aggregate successful samples per modality, then compute an unweighted macro
   mean across modalities for dimensionless normalized-space scores only.

Coverage reports evaluated, skipped, and failed counts. Skipped modalities never
receive synthetic zero scores and never enter the macro numerator. A modality
with no successful samples has no metric value, while its coverage remains
visible. Native-space scores remain per modality because averaging values with
different physical units is scientifically invalid; the suite report declares
cross-modality native macro aggregation not applicable.

## Testing and boundaries

Tests use literal arrays and real implementations. They prove digest-bound
train-only fit, sample/group leakage rejection, state persistence, deterministic
domain-valid corruption, exact unit recovery, cross-modality reuse by one dummy
model, equal-modality macro aggregation, empty-mask rejection, and structured
skips for complex NMR and 2D inputs. Existing XAS tests remain unchanged.

HyperSpectrum owns these contracts and predictions; ACE remains responsible for
formal task/board registration, persisted reports, and ranking.
