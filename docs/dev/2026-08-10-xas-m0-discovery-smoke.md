# XAS M0 discovery smoke against the live volcano catalog

Date: 2026-08-10. Supersedes the connectivity half of
`docs/dev/2026-08-08-xas-m0.md`; the scientific M0 gate stays **blocked**, but
for a different and now precisely located reason.

## Status

**Connectivity blocker cleared. Scoreability blocker remains.**

The 2026-08-08 record stopped at "the Hub could not be reached". The Hub is now
reachable and authentication is server-confirmed, so discovery ran end to end
against the real catalog for the first time. It returned fifteen unique
candidates and admitted none of them. No dataset was selected, no dataset was
materialized, no prediction was run, and
`docs/evidence/xas-m0-selection.json` was deliberately **not** created — the M0
plan forbids a selection instance unless every success field is observed.

The observed facts are recorded in the redacted, machine-readable
`docs/evidence/xas-m0-discovery-smoke.json`.

## What was verified

- The active `volcano` profile answers: HTML root 200 in 0.87 s, unauthenticated
  API 401 in 0.57 s, live searches complete in 2.7–4.6 s.
- `hyd --profile volcano whoami` exits 0 with a server-confirmed identity
  (`vol_***`). Only masked CLI output was read. No profile file, authorization
  value, or endpoint secret was read or recorded.
- `hyperspectrum doctor --json` reports overall readiness: hyd client, Python,
  task contract, and both tool manifests.
- `hyperspectrum tools match --task xas-denoising --json` matches `savgol` and
  keeps `xasdenoise` blocked on its seven recorded policy/resource reasons.
- `hyperspectrum data discover --modality xas --profile volcano --json` exits 0
  with a complete search trace: XAS 7/7, XANES 3/3, EXAFS 5/5, "absorption
  edge" 0/0, one page each, `complete: true`.

## The preferred candidate is not in this catalog

`zenodo-17434349`, the candidate named in the XAS agent design, returns
`total: 0` for both `zenodo-17434349` and `17434349`. Discovery therefore had to
choose from live query evidence rather than a pre-agreed name, which is what the
M0 design asks for. The substitution decision trail is the candidate table in
the evidence file: the discovery machinery ranked and judged all fifteen live
hits, and rejected every one of them, so no substitute was adopted.

## Substring recall is not domain recall

The pinned search contract is Postgres ILIKE, i.e. plain substring matching over
dataset titles and codes. Six of the fifteen unique hits carry no X-ray
absorption content at all:

| Query | Substring coincidence | Datasets |
| --- | --- | --- |
| `XAS` | "Te**xas**" | 4 (storm flooding, coastal bridges, thunderstorm simulation, wetland hydrodynamics) |
| `XAS` | "he**xas**omes" | 1 (INO80 nucleosome remodelling) |
| `XANES` | "silo**xanes**" | 1 (siloxane protocol validation) |

A seventh category is subtler: `EXAFS` matches "N**EXAFS**" in four datasets.
Those are genuine X-ray absorption measurements, but near-edge soft X-ray
spectroscopy is not EXAFS, and one of them (`zenodo-154112`, 1211 files, 133
parsed, `.asc`) is exactly the ESRI-grid misparse trap the M0 plan already
regression-tests. They are recorded as `adjacent_modality`, not as hits.

Only five hits are true domain-term hits: `zenodo-10606662` (in-situ XANES,
`.dat`, 495 files / 491 parsed), `zenodo-15498570` (HERFD-XAS, `.dat`, 109 / 108),
`zenodo-16892323` (XANES, `.rar`, 4 / 4), `zenodo-16610131` (EXAFS, `.mp4`,
11 / 11), and `zenodo-18142209` (EELS + XAS, `.json`, 1 / 1).

This is the concrete justification for the M0 rule that a candidate is never
admitted on its name: term recall must be joined to file-content evidence.

## Why every candidate is blocked

All fifteen candidates, including the five true domain-term hits, returned the
identical verdict:

```
status: blocked
reasons: ["xas_access_unavailable", "xas_energy_axis_invalid"]
readiness_score: 0
```

The cause is a contract gap, not a data absence. A live `hyd search --ilike
--json` record contains:

```
id, dataset_code, title, visibility, status, origin, primary_format,
file_count, storage_backend, created_at, parent_dataset_id, child_count,
samples, instruments, thumbnail_file_id, parsed_file_count, owner_username,
fair, can_write
```

The discovery evidence extractor reads a different set of names —
`dataset_version`, `content_digest`, `license`, `formats`, `access_status`,
`source_kind`, `parser_status`, `axis_evidence`, `label_evidence`,
`pairing_evidence` — none of which the live endpoint emits. It was written
against a redacted fixture that the real wire format does not match.

Three of those are trivially recoverable from fields that *are* present:
`formats` from `primary_format`, `access_status` from `visibility`/`status`,
`source_kind` from `origin`. The rest are not available from this endpoint at
all. In particular **`dataset_version` and `content_digest` are absent**, and
those are two of the six M0 completion-gate conditions. No amount of retrying
the search will produce them.

A knock-on effect: because every readiness score is 0, the ranking is
degenerate. Python's stable sort preserves first-observed search order, so the
Texas flooding dataset ranks first and the real in-situ XANES corpus ranks
tenth. Ranking will only start discriminating once real evidence fields reach
the scorer.

## The 5090 could not host this run

The plan's Task 9 and the selection schema's `host_class: "5090"` both assume
the run happens on that box. It could not:

- `hyd` is installed at the user-local path but is not on the non-interactive
  `PATH`; `hyperspectrum` is not installed there at all.
- `hyd --profile volcano whoami` on the 5090 is rejected by the server: the
  profile's authorization is expired or revoked. That rejection is itself proof
  that the 5090 reaches the Hub — its 2026-08-08 network blocker is gone too.

Discovery is a read-only catalog query, so its result is host-independent and
the run was completed from the coordinating workstation instead. Restoring 5090
authorization is an operator action (it needs an interactive login) and was not
attempted.

## M0 gate evaluation

| Gate | Evidence | Result |
| --- | --- | --- |
| Local test/type/lint suite | Full suite re-run on this branch | Pass |
| Active profile search without credential reads | Live, complete, four-query trace; masked identity only | **Pass (new)** |
| Real dataset version and content digest | Live records carry neither field | Blocked |
| Explicit ground truth and split groups | No axis/label/pairing evidence in live records | Blocked |
| Complete classical PredictionBundle on 5090 | No admitted dataset; 5090 authorization expired | Not run |
| Valid redacted selection handoff | Success-only schema exists; no truthful instance exists | Not generated |

M0 remains **blocked**, with the blocker moved from "cannot reach the Hub" to
"the search contract does not carry scoreability evidence".

## Follow-ups

1. Map the live record fields the endpoint does emit (`primary_format`,
   `visibility`/`status`, `origin`) into discovery's evidence extractor, and
   pin the tests to the real wire shape rather than the fixture shape.
2. Decide how HyperSpectrum obtains a dataset version and content digest under
   the subprocess-only gateway rule. The current `hyd` surface has no read-only
   per-dataset detail command that returns them; this needs either a client
   contract extension or an explicit M0 gate amendment.
3. Restore the 5090 `volcano` authorization and install HyperSpectrum there
   before any run that must satisfy `host_class: "5090"`.
4. Consider whether `zenodo-10606662` — already integrated by the XANES SPEC
   benchmark adapter — should be admitted through the declared-source
   materialization path instead of catalog discovery, since that path already
   produces the version, digests, splits, and ground-truth roles the gate wants.
   That is a design decision, not a bug fix, and is left for the user.

## Redaction note

`process_boundary.redact()` treats `/Ce` in "Ca(II)/Ce(IV)-doped" as a POSIX
absolute path and rewrites the public title to "Ca(II)[REDACTED](IV)-doped".
The lookbehind excludes an alphanumeric predecessor, so "Pd/TiO2" survives but a
slash after a bracket does not. It is fail-safe, but it corrupts public
scientific titles. Recorded here, not fixed here.

No credential, authorization value, signed URL, endpoint address, private path,
raw response body, dataset payload, or model weight is recorded in this document
or in the evidence file.

## Delta: after the wire alignment (same day, same catalog)

Follow-up 1 above is done, and follow-up 4's redaction note is fixed. The
commands were re-run against the same live `volcano` profile after
`claude/m0-wire-alignment`; the catalog was unchanged (XAS 7/7, XANES 3/3,
EXAFS 5/5, "absorption edge" 0/0, one page each, `complete: true`), so the
difference below is entirely the reader, not the data.

### What the extractor now derives, and what it now refuses to invent

| Contract field | Source on the live wire | Result |
| --- | --- | --- |
| `formats` | `primary_format` (one value) | derived, with `formats_complete: false` recorded so a single primary format never reads as an exhaustive list |
| `access_status` | `visibility` + `status` | `admitted_catalog` when the record is `active` and `public`/`internal`; a named `restricted_catalog_*` or `catalog_status_*` otherwise |
| `source_kind` | `origin` | passed through (`zenodo` for all fifteen) |
| `dataset_version`, `content_digest` | *nothing* | stays `None`, and the verdict now says so by name |
| parser / axis / label / pairing evidence | *nothing* | `declared: false` on each aggregate; an undeclared claim is no longer scored as a declared negative one |

### Ranking, before and after

Every readiness score is still 0 — no task-truth evidence exists on this
endpoint, and none was invented. The ranking is no longer degenerate because a
second key sits *below* declared task evidence: whole-word term evidence plus
format evidence, both properties of the record rather than of its name.

| Rank | Before | After | |
| --- | --- | --- | --- |
| 1 | `zenodo-3473148` (Texas flooding, xlsx) | `zenodo-15498570` HERFD-XAS, dat, 108/109 | rel 100 |
| 2 | `zenodo-3380560` (Texas bridges, odb) | `zenodo-10606662` in-situ XANES, dat, 491/495 | rel 100 |
| 3 | `zenodo-17915983` (Texas storms, nc) | `zenodo-18142209` EELS+XAS, json | rel 85 |
| 4 | `zenodo-17483574` (hexasomes) | `zenodo-16892323` XANES, rar | rel 70 |
| 5 | `zenodo-15498570` | `zenodo-16610131` EXAFS, mp4 | rel 60 |
| 6 | `zenodo-10070373` (Texas wetlands) | `zenodo-17915983` | rel 40 |
| 7 | `zenodo-18142209` | `zenodo-3473148` | rel 35 |
| 10 | `zenodo-10606662` | — | |
| 15 | `zenodo-154112` (NEXAFS/ESRI trap) | `zenodo-17483574` (hexasomes) | rel 0 |

The five whole-word technique hits are ranks 1–5; every substring coincidence
and adjacent-modality hit is at 6 or below. The in-situ XANES corpus moved from
tenth to second, and the flooding study from first to seventh.

The separation is lexical and format-based, never nominal. `XAS` inside `Texas`
is an affix match and `XAS` in `基于EELS和XAS谱的铜氧化态预测数据集` is a whole
term, because boundaries are ASCII-alphanumeric. A regression test swaps two
records' titles and formats while leaving their dataset codes in place, and
their ranks swap with them — there is no dataset-name list in the scorer.

### Verdicts

`zenodo-10606662` moved from

```
blocked: ["xas_access_unavailable", "xas_energy_axis_invalid"]
```

to

```
blocked: ["xas_dataset_version_unavailable",
          "xas_content_digest_unavailable",
          "xas_energy_axis_evidence_unavailable"]
```

Both original codes were false. The dataset is public, active and readable, and
no energy axis had been declared and rejected — none had been declared at all.
All fifteen candidates still reach a verdict, all still blocked, and the
blocking reasons are now the true ones. The M0 gate is unchanged: a non-null
version and digest remain required for scoreable admission.

> **Superseded later the same day** — see
> `2026-08-10-inference-only-identity-absence.md`. The three reason codes above
> are unchanged and still true, but their *consequence* is not: an absent
> identity now routes to `inference_only` rather than `blocked`. These fifteen
> candidates are readable and runnable; only their scoreability was ever in
> question. `blocked` is now reserved for access and parse failures. The M0
> gate itself is still unchanged.

### Follow-up 2 is now answered, and stays open as work

A read-only probe of the hub (`hyd --profile volcano dataset show`, plus the
version, quality, files and integrity-summary endpoints) shows the pieces exist
but not on one call: `GET /datasets/{id}/versions` already serialises a version
identity, `GET /datasets/{id}/files` already serialises per-file SHA-256, and
`GET /datasets/{id}/versions/{vid}/quality` already has a `content_digest`
field — whose value is `null` for this dataset. `GET /datasets/{id}`, which
`hyd dataset show` calls, carries none of the three. So the client extension is
mostly a wrapper over endpoints that already exist, with one genuine hub-side
gap: the digest value, not the digest field.

## Test and verification delta

`.venv/bin/pytest -q` 716 passed (682 before; 34 added: 20 pinning the live
wire, 14 pinning the redaction boundary). `ruff check src tests` clean,
`mypy src` clean over 44 files, `git diff --check` clean.

The live capture is committed at
`tests/fixtures/hyperdata/hyd-search-live-volcano-2026-08-10.json`. Its
per-query SHA-256 digests equal `artifact_digests.search_response_sha256` in the
evidence file, which is what re-establishes the raw responses the smoke did not
retain; a test asserts that agreement so the fixture cannot drift away from the
run it explains.
