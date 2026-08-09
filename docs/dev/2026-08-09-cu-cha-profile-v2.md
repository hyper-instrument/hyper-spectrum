# Cu-CHA operando XAS profile v2

This adapter turns a byte-verified `zenodo-10159154` Cu-CHA DAT tree into a
reproducible, agent-readable denoising benchmark profile. It does not contain,
download, or declare an official production asset pin. A production caller
must supply a declaration generated from its own approved source inventory.

## Admission and grouping

Every DAT file must declare exactly these columns:

```text
musst_enestep, mu_trans, mu_ref, I0, I1, I2
```

The adapter converts `musst_enestep` from keV to eV, rejects non-finite data,
non-increasing axes, non-positive count channels, and any scan for which
`mu_trans` does not agree with `log(I0/I1)`. Relative paths must use the public
Cu-CHA naming convention. The leakage unit is
`experiment/condition_segment`, derived by removing scan index and temperature
from the filename. A deterministic size-balanced policy assigns complete
groups to train, validation, or test; all dose variants reuse this one split.

The audited experiment-directory vocabulary is fail-closed:
`High-Cu|Low-Cu` is followed by exactly `exposure|cycles` and then a non-empty
experiment suffix. These first two tokens define `cu_loading` and
`protocol_family`; an unknown token is rejected rather than guessed. Splitting
is performed independently inside every `cu_loading × protocol_family`
stratum. Each stratum must contain at least three condition groups so train,
validation, and test are all represented. The root manifest records the two
strata keys plus per-stratum group and sample counts for every split.

## Source declaration

The command accepts a strict complete-tree declaration. `files` order is not
semantically significant; the materializer canonicalizes it before hashing.

```json
{
  "schema_version": "hyperspectrum-cu-cha-source-declaration/v1",
  "dataset": {
    "code": "zenodo-10159154",
    "version": "v1",
    "version_id": "CATALOG_VERSION_ID"
  },
  "files": [
    {
      "path": "data_txt/High-Cu_exposure_PROTOCOL/1_at_200C_CONDITION.dat",
      "size": 12345,
      "sha256": "LOWERCASE_64_CHARACTER_SHA256"
    }
  ]
}
```

The observed relative path set, byte size, and SHA-256 must all match before
the output directory is created. The example values are structural placeholders
and must never be copied into an official asset record.

Source reads use a directory-FD/openat walk with `O_NOFOLLOW`. The root, every
parent directory, and the final regular file must retain the scanned
device/inode identity; final-file size, mtime, and ctime must also remain stable
before and after reading from the same FD. A platform without these primitives
fails closed. This prevents a scan/read symlink-swap from escaping the declared
source root.

## Corruption and target semantics

The three fixed tracks are dose fractions `0.10`, `0.25`, and `0.50`. For each
sample and dose, a 64-bit seed is derived from dataset version ID, source
SHA-256, relative path, dose, corruption-contract version, and the exact NumPy
distribution ABI identity. A fixed PCG64 generator samples
`Poisson(dose * I0)` and then `Poisson(dose * I1)`, rescales both channels, and
recomputes `log(I0/I1)`. A zero sampled count rejects the materialization; no
epsilon correction is hidden in the scientific contract.

PCG64's random-bit stream compatibility does not extend to
`Generator.poisson`. Consequently, the root manifest and every track manifest
freeze the exact `numpy_version`, `numpy.random.PCG64`,
`numpy.random.Generator.poisson`, and `I0`-then-`I1` draw order under
`hyperspectrum-numpy-poisson-abi/v1`. This profile has one official
distribution runtime: `numpy==2.4.6`, matching the verified ACE XAS runtime.
The materializer, direct thinning API, and loader all fail closed before any
Poisson draw when either the declared or installed version differs. The ACE
runner image must therefore install `numpy==2.4.6`; the project's general
library dependency range is not a reproducibility claim for this profile.
Generating an alternative profile under whatever NumPy happens to be installed
is deliberately forbidden.

Use the lock's Python 3.11 environment for local materialization and verify it
before producing an official asset:

```bash
uv sync --python 3.11 --frozen
.venv/bin/python -c 'import numpy; assert numpy.__version__ == "2.4.6"'
```

The ACE Dockerfile/lock must make the equivalent exact pin and its offline
contract test must assert `numpy.__version__ == "2.4.6"` before invoking the
adapter.

The benchmark target is always:

```json
{
  "ground_truth_kind": "proxy_full_count",
  "is_physical_noiseless_ground_truth": false,
  "signal": "mu_trans_full_count"
}
```

It is a measurement-derived proxy and must not be reported as independent
clean or physical noiseless ground truth.

## Bundle and execution boundary

```text
profile/
  manifest.json
  split.json
  tracks/
    dose-0.10/
      manifest.json
      benchmark.npz
      inference.npz
    dose-0.25/...
    dose-0.50/...
```

`benchmark.npz` contains input, proxy target, count channels, sample seeds,
group/split identity, and source provenance. `inference.npz` contains only the
five fields accepted by HyperSpectrum's reusable local inference boundary:
energy, noisy signal, sample IDs, group IDs, and energy unit. The public loader
returns core `DenoisingPair` objects for quantitative evaluation.

The materialization result publishes `profile_sha256`, the SHA-256 of the exact
root `manifest.json` bytes. `load_cu_cha_denoising_pairs` requires that value as
`expected_profile_sha256` and verifies it before trusting any internal digest.
The loader then validates both NPZ schemas, exact inference/benchmark input
equality, physical count/log-ratio relationships, path-derived identities,
split strata, per-sample seed derivation, the exact NumPy distribution ABI, and
regenerated PCG64 Poisson draws.
Callers must persist the externally trusted profile digest separately; reading
a digest from the bundle being validated is not a trust root.

The execution security boundary is mandatory: a candidate model container may
receive only the selected track's `inference.npz`; `benchmark.npz`, its proxy
target, and scorer logic remain outside that container. The leaderboard scorer
publishes metrics from `split == test` only. Train and validation membership is
available to later one/few-shot policy code but must never be folded into the
formal test score. The v2 manifest records these requirements; the future ACE
adapter must enforce them when constructing mounts and score inputs.

This is deliberately `hyperspectrum-xas-denoising-profile/v2`. Current ACE v3
planning is bound to the older canonical XANES SPEC asset shape, so ACE must add
a thin profile-v2 adapter before using the target-bearing benchmark asset. No
implicit v1 conversion is allowed. Therefore this M0 materializer and loader do
not by themselves claim a completed ACE leaderboard/runtime loop.
