# XANES SPEC Benchmark Materialization Design

**Status:** Approved 2026-08-09

## Goal

Materialize the 495-file Zenodo 10606662 source tree into a deterministic,
agent-ready XANES denoising benchmark without confusing upstream dataset
identity, verified source content, or derived asset bytes. The frozen target is
a measured proxy, never physical noiseless ground truth.

## Identity model

New v3 execution and prediction contracts carry three independent SHA-256
values:

- `source_dataset_digest`: canonical digest of dataset id/code/version,
  upstream revision (including explicit `null` or `"unknown"`), and declared
  manifest digest.
- `source_content_manifest_digest`: canonical digest of sorted source-relative
  path, observed byte size, and observed SHA-256 for every source file.
- `benchmark_asset_digest`: SHA-256 of the exact canonical NPZ bytes.

No equality is required between these identities. Run-plan and prediction v2
remain readable and executable only for compatibility. New planning produces
v3 only after validating the complete benchmark manifest, recomputing both
source identity digests, and binding its declared-manifest digest to the
catalog candidate. The local executor validates NPZ bytes against
`benchmark_asset_digest` only and requires the canonical 16-key materializer
contract for v3; the five-key input contract remains v2 compatibility only.
Evidence v2 names all three identities; the validator continues to accept
frozen v1 evidence.

## Source admission

The materializer consumes a source root plus a declared JSON manifest. Every
declared SHA-256 must match. Declared/observed size disagreements are recorded
as metadata drift and do not reject a SHA-matching file. Missing, unexpected,
or SHA-mismatching source files fail the whole materialization. Each admitted
file is parsed from the exact in-memory byte snapshot that was hashed, so a
path replacement after verification cannot alter derived asset bytes.

Files are classified before parsing:

- Only `.dat` files are spectrum candidates.
- `.txt` files are condition sidecars and remain in provenance.
- Extensionless files are excluded motor scans and remain in provenance.

A `.dat` file is admitted only when it has exactly one `#S` scan, one `#L`
header, 16 uniquely named columns including case-insensitive `energy`, `i0`,
and `ketek`, and exactly 135 finite numeric rows. Energy must be strictly
increasing and have endpoints within 0.05 eV of 5693.0 and 5801.4 eV; `i0`
must be positive and `ketek` non-negative. Empty, short, multiple-scan,
malformed, or path-unknown
spectra are excluded with stable reason codes in QC. Materialization fails if
no valid spectrum remains.

## Split and grid

Path classification yields experiment and composition provenance. Experiment 1
is explicitly mapped to composition `La00`; other experiments and powder files
must encode their composition in the filename.

The fixed v1 split is applied before any noise:

- test: La08 (`Exp4`, `Exp9`, `Powder_La08`)
- validation: Gd05 (`Exp8`, `Powder_Gd05`)
- train: every other admitted composition

Composition is the leakage group. The split manifest therefore rejects any
composition crossing partitions, while experiment remains per-sample
provenance.

After splitting, paths are sorted. The first admitted train sample freezes the
real, non-uniform 135-point reference energy grid. Its relative path and file
SHA-256 are recorded. Other admitted signals are linearly interpolated onto
that grid. At most 0.05 eV of reference-grid endpoint jitter is clamped to the
nearest measured endpoint and recorded by the fixed preprocessing version;
larger overhang rejects the sample into QC.

## Target and noise

The clean-side target is named exactly `pseudo-clean frozen measurement` and is
the interpolated measured ratio `ketek / i0`. Its provenance includes
`is_physical_noiseless_ground_truth: false`.

Noise is injected in source count-channel space after the split is fixed. For
configured `0 < dose_fraction <= 1`, each raw channel uses:

`Binomial(round(count), dose_fraction) / dose_fraction`

`i0` and `ketek` are sampled independently. The noisy ratio is formed from the
thinned channels and then interpolated to the reference grid. Each sample gets
an order-independent seed derived from the global seed, sample id, source file
SHA-256, algorithm version, and dose. The default dose is 0.25, while 0.1 and
0.5 can produce separate digest-distinct bundles.

## Outputs and API

The output directory contains canonical `benchmark.npz`, `manifest.json`,
`split.json`, and `qc.json`. The NPZ carries the reference and source grids,
raw count channels, noisy and pseudo-clean ratios, ids, groups, experiments,
splits, paths, file digests, and per-sample seeds. ZIP entry order and metadata
are fixed so identical inputs produce identical bytes.

The adapter exposes a loader that constructs immutable `SpectrumSample` and
`DenoisingPair` values. Noisy and target samples share the exact reference
axis, mask, ratio unit, identity, and split assignment. CLI materialization is
JSON-first and returns the four output paths plus all three digests.
