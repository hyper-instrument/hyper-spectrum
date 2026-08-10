# An absent identity degrades to inference-only; it does not block

## What was wrong

The live-wire smoke run earlier today ended with every one of the fifteen
candidates the volcano hub serves reading as `blocked`. The reason codes were
true — the catalog publishes no dataset version, no content digest, and no
energy-axis evidence for them — but the verdict they produced was not.

`blocked` says *nothing can be done with this*. What was actually true is
narrower: nothing can be **scored** against it. Inference needs neither a pinned
revision nor a digest; only a quantitative claim does, because only a
quantitative claim has to be reproducible later against specific bytes. Reading
the whole live corpus as blocked told the honest-sounding but wrong story that
the hub had nothing runnable at all.

## The rule now

`_blocked_verdict` became `_admission_verdict`, and it sorts gaps into two kinds
that no longer share an answer:

| gap | reason code | verdict |
|---|---|---|
| not readable through the admitted catalog | `xas_access_unavailable` | `blocked` |
| axis declared but does not validate | `xas_energy_axis_invalid` | `blocked` |
| no pinned dataset version | `xas_dataset_version_unavailable` | `inference_only` |
| no content digest | `xas_content_digest_unavailable` | `inference_only` |
| no energy-axis evidence published at all | `xas_energy_axis_evidence_unavailable` | `inference_only` |

**Failures block; absences degrade.** A claim we know to be wrong
(`..._invalid`) and bytes we cannot fetch are failures. A claim the catalog
never made (`..._unavailable`) is an absence — unknown, not known to be wrong —
and that distinction was already encoded in the reason codes; it just wasn't
reaching the verdict.

Three things deliberately did **not** change:

1. **The admission gate.** The degraded route returns *before* any task truth is
   examined, so an unidentified candidate can never come out `scoreable` — even
   one with verified noisy/clean pairing. Scoring still requires an identity.
   There is a test for exactly this
   (`test_identity_absence_still_denies_a_scoreable_verdict`).
2. **Every reason code.** They are carried verbatim onto the degraded verdict,
   plus one sentence saying why absence is not a blockage. A candidate that is
   *both* unreadable and unidentified reports all of it under the harder
   verdict; nothing gets swallowed by the softer one.
3. **Ranking.** Ranking is evidence-led and runs before any verdict exists, so
   the two large spectroscopy corpora still lead. They are now inference-only
   rank leaders, which is the intended reading: they run, they just cannot be
   scored until the catalog identifies them.

## The forward-looking seam

The hub is growing a `hd dataset detail --json` command whose `identity` block
answers "*which bytes was this?*" even for corpora no quality run ever touched
(upstream `hyper-instrument/hyper-data` PR #251). It publishes the digest
**and where the digest came from**:

- `content_digest_source: "quality_verified"` — a quality run produced it;
- `content_digest_source: "files_fingerprint"` — derived at ingest from the
  sorted `(path, sha256, size)` manifest.

The load-bearing decision, shared with the hub: **value and status are
independent axes.** A fingerprint-derived digest under
`quality_status: "unverified"` is a perfectly good content identity — it pins
which bytes — so it must not degrade a candidate. Only the *absence* of a digest
does. Refusing the fingerprint would put every hub-ingested corpus straight back
behind the gate the hub just opened.

So discovery now reads `content_digest_source` and `quality_status` off a record
when they are present, aggregates them under the same uniformity rule the digest
itself follows (two rows that disagree are not evidence of either answer), and
carries them through `DatasetCandidate` → `XASCandidateProfile` →
`ReadinessVerdict`. A consumer records all three together instead of guessing
how well the identity was established.

Today's search wire carries neither field, and absent stays absent — there is a
test pinning that a missing field never becomes an invented provenance claim.

`tests/fixtures/hyperdata/hyd-dataset-detail-identity.json` is the contract. The
two repos share no code and no import; if the hub renames a field, a test here
fails instead of the identity quietly reading as absent.

## Files

| file | change |
|---|---|
| `src/hyperspectrum/tasks/recommend.py` | `_blocked_verdict` → `_admission_verdict` + `_gap_verdict`; blocking/degrading split; `content_digest_source` + `quality_status` on `ReadinessVerdict` and `XASCandidateProfile` |
| `src/hyperspectrum/hyperdata/models.py` | `DatasetCandidate.content_digest_source` / `.quality_status` (default `None`) |
| `src/hyperspectrum/hyperdata/discovery.py` | read both fields per observation; aggregate with `_uniform_optional_string` |
| `tests/fixtures/hyperdata/hyd-dataset-detail-identity.json` | new — the upstream detail-wire contract |
| `tests/hyperdata/test_detail_wire_identity_seam.py` | new — 8 tests over that fixture |
| `tests/tasks/test_recommend.py` | +7 tests on the routing split |
| `tests/hyperdata/test_discovery_live_wire.py` | two expectations flipped to `inference_only`; +1 test that the rank leaders are runnable |
| `docs/dev/2026-08-10-xas-m0-discovery-smoke.md` | supersession note on the Verdicts section |

## How to test

```bash
.venv/bin/pytest -q          # 732 passed (716 before; 16 added)
.venv/bin/python -m ruff check src tests   # clean
.venv/bin/python -m mypy src               # clean, strict
```

## Known follow-ups

- Nothing consumes the detail wire yet. `SearchGateway` still exposes only
  `search`; wiring a `detail` call means widening that Protocol, and the
  upstream command is not deployed. The seam is shaped and pinned, not plumbed.
- `docs/evidence/xas-m0-discovery-smoke.json` still records
  `catalog_record_lacks_content_digest` against a blocked reading of that run.
  It is a dated evidence artifact, left as captured; the next capture will carry
  the new verdicts.
- `content_digest_source` is typed `str | None` here rather than a Literal. The
  hub owns that vocabulary and may extend it; validating a closed set on this
  side would turn an upstream addition into a hard failure instead of an
  unrecognised-but-recorded value.
