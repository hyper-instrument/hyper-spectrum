# Unified Spectral Denoising Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-independent, leakage-safe denoising contract that lets one compatible model be evaluated across multiple spectroscopy modalities while retaining physical units and honest coverage.

**Architecture:** A focused `hyperspectrum.denoising` package owns immutable sample arrays, normalization/noise transforms, canonical model I/O, and evaluation. Existing `ObservationBundle` remains the external manifest. Models never own modality parsing, fitted normalization, inverse transforms, metrics, or aggregation.

**Tech Stack:** Python 3.10+, frozen dataclasses, NumPy, existing immutable JSON helpers, pytest, Ruff, mypy.

## Global Constraints

- Do not implement training, remote execution, share import, or ACE persistence.
- Do not modify XAS operational evidence files.
- Never infer axes or units from array shape.
- Dataset-fitted normalization may fit only the train split and must persist training identity/digest.
- Native-space metrics require declared exact inversion.
- Incompatible modalities are structured skips, never zero-valued scores.

---

### Task 1: Immutable spectrum samples

**Files:** Create `src/hyperspectrum/denoising/sample.py`; test `tests/denoising/test_sample.py`.

**Interfaces:** Produce `SpectrumAxis`, `SpectrumSample`, `SpectrumRepresentation`, and `SpectralModality`.

- [x] Write failing tests for immutable copied arrays, dense axis/shape matching, explicit channels, sparse peaks, complex NMR, and rejection of implicit complex/sparse/multi-axis coercion.
- [x] Run `pytest tests/denoising/test_sample.py -q` and verify missing imports fail.
- [x] Implement strict constructors with read-only arrays and frozen JSON metadata/provenance.
- [x] Re-run the focused tests and refactor only while green.

### Task 2: Leakage-safe normalization registry

**Files:** Create `src/hyperspectrum/denoising/normalization.py`; test `tests/denoising/test_normalization.py`.

**Interfaces:** Produce `NormalizationDefinition`, `NormalizationState`, `NormalizedSpectrum`, `NormalizationRegistry`, `default_normalization_registry`, and `fit_normalizer`/`normalize`/`denormalize` behavior.

- [x] Write failing tests that distinguish `per_sample` from `dataset_fitted`, reject val/test fitting, persist train sample IDs/digest/parameters, reject state/unit/modality mismatches, and round-trip exact transforms.
- [x] Run `pytest tests/denoising/test_normalization.py -q` and verify expected failures.
- [x] Implement range, TIC, complex RMS, and train-global-standardization definitions with explicit compatibility and inverse declarations.
- [x] Re-run focused tests and refactor only while green.

### Task 3: Native-space deterministic noise

**Files:** Create `src/hyperspectrum/denoising/noise.py`; test `tests/denoising/test_noise.py`.

**Interfaces:** Produce `NoiseSpec`, `NoiseProvenance`, `CorruptedSpectrum`, and `inject_noise`.

- [x] Write failing tests for seeded Gaussian and Poisson corruption, source digest/provenance, non-negative count validation, unit preservation, and fail-closed complex inputs.
- [x] Run `pytest tests/denoising/test_noise.py -q` and verify expected failures.
- [x] Implement minimal native-space noise injection and immutable provenance.
- [x] Re-run focused tests and refactor only while green.

### Task 4: Canonical reusable model boundary

**Files:** Create `src/hyperspectrum/denoising/model.py`; test `tests/denoising/test_model.py`.

**Interfaces:** Produce `ModelCapabilities`, `CompatibilityIssue`, `CanonicalDenoisingInput`, `CanonicalDenoisingOutput`, and `DenoisingModel` protocol.

- [x] Write failing tests for capability checks over rank, representation, channel count, required normalization, native recovery, and output identity/shape/mask/state digest.
- [x] Run `pytest tests/denoising/test_model.py -q` and verify expected failures.
- [x] Implement immutable canonical I/O and structured compatibility checks without modality-specific model code.
- [x] Re-run focused tests and refactor only while green.

### Task 5: Dual-space suite evaluator

**Files:** Create `src/hyperspectrum/denoising/evaluation.py` and `src/hyperspectrum/denoising/__init__.py`; test `tests/denoising/test_evaluation.py`.

**Interfaces:** Produce `DenoisingPair`, `SampleEvaluation`, `ModalityEvaluation`, `DenoisingSuiteReport`, and `evaluate_denoising_suite`.

- [x] Write failing tests where one identity dummy model evaluates XAS and Raman, equal modality means form the macro score despite unequal sample counts, and complex NMR/2D EELS become structured skips without zero metrics.
- [x] Write failing tests that normalized and exact-inverse native RMSE/MAE are both emitted, while malformed model output is a typed failure.
- [x] Run `pytest tests/denoising/test_evaluation.py -q` and verify expected failures.
- [x] Implement the external normalization/canonical-model/validation/inverse/metric flow and coverage aggregation.
- [x] Re-run focused tests and refactor only while green.

### Task 6: Documentation and verification

**Files:** Modify `README.md`; package exports under `src/hyperspectrum/denoising/__init__.py`.

- [x] Document the model reuse boundary, train-only normalization rule, dual metric spaces, and structured coverage semantics.
- [x] Run `.venv/bin/pytest tests/denoising -q`.
- [x] Run `.venv/bin/pytest -q`.
- [x] Run `.venv/bin/ruff check src tests`; run strict mypy on the new package/tests. Full-repository mypy was also run and retains its pre-existing errors outside this change, so unrelated files were not rewritten.
- [ ] Review `git diff --check`, inspect the complete diff, and create one local commit without pushing.
