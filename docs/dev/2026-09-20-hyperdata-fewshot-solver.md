# HyperData few-shot XAS solver: `hyperspectrum.fewshot`

## Goal

Give HyperData's XAS sim↔exp few-shot competition a participant-side solver that
lives next to the spectroscopy models, the way the FIB solver lives next to
lumen. The organizer's stdlib kit (`apps/fewshot-solver` in hyper-data) calls it
as a subprocess with a fixed three-step contract and never imports numpy; this
package does the numeric work and writes the seven-file delivery the organizer's
`xas_evaluator.py` scores.

v1 is a baseline on purpose: a faithful port of the organizer's own
nearest-neighbour reference. A real spectral model replaces `train`/`predict`
only; the contract, delivery and checks stay.

## What changed

| File | |
|---|---|
| `src/hyperspectrum/fewshot/knn.py` | `PairIndex` (concatenated paired pools, stable order, `excluding`, `save`/`load`), `candidate_indices`, `direction_keys`, `predict_many` — the reference numerics, byte-identical. |
| `src/hyperspectrum/fewshot/delivery.py` | The seven files (`predictions.npz`, `cycle_predictions.npz`, `method_report.md`, `method_summary.json`, `data_usage.json`, `iteration_log.jsonl`, `run_manifest.json`), the evaluator's byte limits and `data_usage` rules enforced before anything counts. |
| `src/hyperspectrum/fewshot/solver.py` | `python -m hyperspectrum.fewshot.solver prepare\|train\|predict` with `--release-dir/--work-dir/--model-dir/--out-dir --opt k=v`; sha256 and energy-axis verification of the release, duplicate-id rejection, leak-free diagnostic MAE, one-line failures (`SolverError`, exit 2 for bad `--opt`, 1 otherwise). |
| `src/hyperspectrum/fewshot/__init__.py` | Public surface. |
| `tests/fewshot/test_knn.py`, `test_delivery.py`, `test_solver.py` | 55 tests: hand-computed weighted averages, tie and exact-match rules, delivery limits and `data_usage` rules, the three steps on a synthetic release, subprocess no-traceback checks. |

No new dependencies; `mypy --strict` and ruff clean.

## Key decisions

- **The numerics are the organizer's.** Candidate chain same Z+edge → same edge
  → all; mean-absolute distance; `lexsort` on ids for ties; the first exact
  match is decisive; inverse-square weights with the 1e-6 floor; float64 math,
  float32 output. A randomized comparison against the reference code is
  byte-identical. The only addition is a guard for weights that underflow to
  zero (single nearest neighbour), which the reference would have crashed on.
- **Both paired pools are `required` in a real release**, so nothing may be held
  out of the final index without failing the evaluator's `data_usage` rule.
  `train` therefore scores the pool named `validation` from an index built
  without its sample ids (a leak-free diagnostic MAE written to
  `train.json.validation`), then re-admits it to the final index.
- **The release is verified, not trusted**: pool sha256 against
  `data_manifest.json`, `queries.npz` against `queries_sha256`, every energy
  axis against `start_ev + step_ev·k`, duplicate sample ids across pools,
  `index.npz` record count against `train.json`.
- **Failures are one stderr line**, never a traceback, so the kit can relay
  them; `main()` backstops unexpected exceptions the same way.

## How to test

```bash
uv run --no-sync pytest -q tests/fewshot
uv run --no-sync ruff check src tests && uv run --no-sync mypy src
```

Real run (2026-09-20, HyperData hub on the GPU box, release frozen from the
v0.6 reference package, 45 hidden pairs, 20+20 cycle queries, `disjoint`
layout): the organizer's evaluator returned `succeeded`, total score 57.30
(sim2exp 68.23 / exp2sim 50.10), every automatic qualification passed, and the
receipt was accepted by the hub with both digests verified. Diagnostic MAE on
`validation.npz`: sim2exp 0.099, exp2sim 0.242.

## Known follow-ups

- Replace `train`/`predict` with a learned spectral model; keep the delivery.
- Content duplicates (identical spectra under different ids) across pools are
  not detected; de-duplicating by content would change the reference numerics.
- The diagnostic MAE is O(pool × index); large paired pools may want a
  deterministic subsample.
- `predict.json` is written into the delivery directory as an eighth file; the
  evaluator ignores it.
