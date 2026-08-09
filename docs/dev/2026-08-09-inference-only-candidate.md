# Inference-only candidates: `inference.npz` becomes the test rows only

**Date:** 2026-08-09
**Status:** implemented on `claude/inference-only-candidate`, not yet
re-materialized against real Cu-CHA sources.
**Supersedes, in one respect:** `docs/dev/2026-08-09-cu-cha-profile-v2.md`.
That document describes `inference.npz` as carrying every admitted spectrum's
noisy input. As of this change it carries the scored test rows and nothing else.
The rest of that document — the source-integrity chain, the Poisson contract,
the split policy, the trust root — is unchanged and still current.

## Why now

ACE's dual-asset evaluation lane mounts exactly one file into a candidate
container: the selected track's `inference.npz`. Reading the two halves against
each other turned up a pair of problems that stop a dual-asset run before
anything else matters.

1. The candidate image had no entry point that could consume a single file. Its
   adapter walked its data mount looking for a directory holding
   `manifest.json`, `split.json` and `benchmark.npz`, and refused when it found
   none — which is every dual-asset mount, by construction.
2. Underneath that, a worse one. `inference.npz` carried **every** split's rows.
   A container that denoised what it was handed would answer for train and
   validation samples too. The leaderboard scorer refuses any prediction outside
   the fixed test split as leakage, and it is right to — but that refusal is
   only sound because the mount is assumed to be test-only. The candidate cannot
   filter, because the split is a scorer-only asset. ACE must not filter on the
   candidate's behalf, because then ACE would be choosing which rows the answer
   covers.

The decision taken was: **the candidate's input contains only what the candidate
is being asked to answer.** The alternative on the table was to have the scorer
score the intersection and treat extra predictions as unscored rather than as
leakage. That was rejected: it moves a frozen metric contract, changes the
scorer's code digest, and weakens a refusal that is currently correct. On the
traditional path there is no in-container training, so train and validation rows
in a candidate mount are pure leakage surface and buy nothing.

## What changed here

`materialize_cu_cha` computes the scored-row selection once, from the same split
manifest the scorer receives, on the sorted sample order both assets share — so
"the rows the candidate gets" and "the rows the scorer will accept a prediction
for" are one selection rather than two that happen to agree. Each track's
`inference.npz` is written from that selection.

`benchmark.npz` is unchanged. Every row, every count channel, every seed, the
proxy target and the `splits` array all stay in the scorer's half. Only the
candidate's view narrows.

`load_cu_cha_denoising_pairs` now checks both halves of the new sentence: the
inference arrays must be the same numbers as the benchmark's test rows *and*
must contain no other row. A materialization that quietly went back to every row
is refused by name rather than discovered later as leakage in an answer.

A new facade, `execute_ace_xas_inference_only`, is what a candidate container
calls in place of the benchmark-directory walk. It is the inference-only sibling
of `execute_ace_xas_denoising`: same tool, same parameters, same numerics.

## Which strings moved, and why those

This is a semantic change to a three-times-reviewed profile, so old and new
materializations have to be distinguishable **by contract**, not only by digest.
A digest tells an operator that bytes changed; it does not tell them that the
bytes now mean something else.

| Where | Old | New |
| --- | --- | --- |
| track manifest `inference_asset.contract` | `hyperspectrum-local-inference-npz/v1` | `hyperspectrum-local-inference-npz/v2` |
| root manifest `execution_boundary.candidate_container_asset` | `inference_only` | `inference_only_test_split` |

Both are exact-matched — by this repository's loader and by ACE's profile
adapter — so a `/v1` materialization cannot be read as a `/v2` one by either
side, and the failure names the contract rather than surfacing as a digest
mismatch of unknown cause.

### Why `PROFILE_SCHEMA_VERSION` did *not* move

It was considered. The argument for bumping
`hyperspectrum-xas-denoising-profile/v2` → `/v3` is that a consumer written
against v2 will now reject a manifest that still calls itself v2, and
"schema v2 but invalid v2 root contract" is a more confusing refusal than
"unsupported schema version".

The argument against, which won: the root manifest's *shape* is unchanged — same
keys, same types, same nesting — and the two strings above are precisely the
fields whose job is to name the candidate asset contract. Bumping the profile
schema as well would ripple through ACE's `xas_profile_v2` adapter, its track
schema, the materialization-result schema and their fixtures, for no additional
discrimination: any consumer that checks either moved string already fails
closed. Recorded here because it is a defensible call in the other direction and
a later reviewer should not have to re-derive why it went this way.

## Digests that move

Every track's `inference_asset_sha256` changes, in all three dose tracks. The
root `manifest.json` therefore changes, so `profile_sha256` changes, and every
external trust root pinned to it must be re-pinned. On the ACE side that is
`RESEARCH_XAS_PROFILE_SHA256`.

`benchmark_asset_sha256` does **not** move. Same rows, same seeds, same draws.

## The Poisson chain is untouched

No change to seed derivation, to `poisson_thin_transmission`, to the draw order,
or to the pinned NumPy 2.4.6 PCG64 ABI. The numpy-2.4.6 golden vector test
passes unmodified. What changed is which rows are copied into the candidate
asset after the draws have already happened.

The golden re-verification against real Cu-CHA sources happens at
re-materialization on the 5090 — that run is what confirms the full-count proxy
and the thinned counts are bit-identical to the reviewed profile, with only the
candidate asset's row count differing.

## How to test

```bash
uv run pytest tests/datasets/test_cu_cha.py tests/execution/ -q
```

Note that `tests/datasets/test_cu_cha.py` is gated on the exact pinned NumPy
runtime (2.4.6). On any other NumPy the Poisson-bearing tests fail closed rather
than skip, which is deliberate; install the pinned version to run them.

The two tests that carry this change specifically:

* `test_candidate_inference_asset_carries_only_the_scored_test_rows`
* `test_loader_rejects_an_inference_asset_that_readmits_untested_rows`

## Known follow-ups

* Re-materialize on the 5090 and re-pin `profile_sha256` everywhere it appears.
* The ACE side needs the matching change: its profile adapter must expect the
  new contract strings and must check the mounted inference asset against the
  profile's **test** sample count rather than its total sample count.
