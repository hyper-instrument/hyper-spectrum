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
      "path": "data_txt/High-Cu_PROTOCOL/1_at_200C_CONDITION.dat",
      "size": 12345,
      "sha256": "LOWERCASE_64_CHARACTER_SHA256"
    }
  ]
}
```

The observed relative path set, byte size, and SHA-256 must all match before
the output directory is created. The example values are structural placeholders
and must never be copied into an official asset record.

## Corruption and target semantics

The three fixed tracks are dose fractions `0.10`, `0.25`, and `0.50`. For each
sample and dose, a 64-bit seed is derived from dataset version ID, source
SHA-256, relative path, dose, and corruption-contract version. A fixed PCG64
generator samples `Poisson(dose * I0)` and then `Poisson(dose * I1)`, rescales
both channels, and recomputes `log(I0/I1)`. A zero sampled count rejects the
materialization; no epsilon correction is hidden in the scientific contract.

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
