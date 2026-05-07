---
name: rieszreg-add-learner-package
description: Contract checklist for adding a new learner package or a new backend to the rieszreg family. Covers which Protocol to implement (`Backend.fit_augmented` for augmentation-style, `MomentBackend.fit_rows` for moment-style), the directory layout, sklearn-compatibility gates, the R6 wrapper, required test suites (sklearn-conformance, reference parity, consistency), per-estimand examples, and serialization. Triggers when scaffolding a new package directory under `packages/`, adding a backend file under `packages/*/python/*/backends/`, asking "how do I add a learner / backend / package", or when the conversation mentions a not-yet-existing package name.
---

# Adding a new learner package or backend

This is the contract every implementation package must meet. The structure of the project assumes each backend implements one of the two Protocols from `rieszreg.backends.base`; everything else is determined by that choice.

For the architectural rules behind the contract (tier classification, the agnostic-orchestrator principle, sklearn-first, lazy imports), see the `rieszreg-architecture` skill.

## 1. Directory layout

```
packages/<pkg>/
├── python/<pkg>/
│   ├── backends/              # ONLY backend implementations
│   ├── <pkg>_specific.py      # things unique to the package (e.g. kernel algebra in krrr)
│   ├── convenience.py         # optional thin subclass of rieszreg.RieszEstimator with backend baked in
│   └── __init__.py            # re-exports rieszreg primitives + the package's own additions
├── r/<pkg>/                   # subclass of rieszreg's R6 base
│   ├── DESCRIPTION            # Imports: R6, reticulate, rieszreg
│   ├── NAMESPACE
│   ├── R/
│   └── tests/testthat/
├── python/tests/              # backend-specific tests + the consistency suite from rieszreg.testing
├── examples/                  # per-estimand + per-backend examples
├── pyproject.toml             # depends on rieszreg
└── CLAUDE.md                  # implementation-side notes (architecture, sharp edges)
```

`pyproject.toml` declares `Depends: rieszreg`. setuptools build, Python ≥3.10. Optional-deps groups for heavy backends (`[lgb]`, `[jax]`, `[falkon]`, `[keops]`, `[torch]`).

## 2. Pick the backend Protocol

Both Protocols live in `packages/rieszreg/python/rieszreg/backends/base.py`. Both return `FitResult(predictor, best_iteration, best_score, history)`. Pick whichever fits your learner's natural loss decomposition:

- **`Backend.fit_augmented(aug_train, aug_valid, loss, ...)`** — for learners whose loss decomposes naturally over the augmented `(a, b)` evaluation points. Use for kernel ridge, gradient boosting, anything that wants the augmented dataset pre-computed for it. Reference impls: `KernelRidgeBackend` (krrr), `XGBoostBackend` / `SklearnBackend` (rieszboost).

- **`MomentBackend.fit_rows(rows_train, rows_valid, estimand, loss, *, ys_train=None, ys_valid=None, ...)`** — for learners whose loss decomposes per original sample row. Compute per-row moments via `rieszreg.trace(estimand, row, y)` directly, avoiding the augmentation blow-up. Use for random forests, neural nets. Reference impls: `ForestRieszBackend` (forestriesz), `TorchBackend` (riesznet).

The orchestrator dispatches at fit time based on which Protocol the backend exposes. Backends implementing both default to `fit_augmented` for back-compat.

Return a `Predictor` with `predict_eta()` and `predict_alpha()` (link applied). Inherit the base interface from rieszreg; storage format is your choice.

## 3. What you reuse from rieszreg (do not redefine)

- **Estimands.** Import `Estimand`, `FiniteEvalEstimand`, and the five built-ins (`ATE`, `ATT`, `TSM`, `AdditiveShift`, `LocalShift`). New estimand factories that belong family-wide go in `rieszreg.estimands`, not your package.
- **Losses.** Import the `Loss` base and built-ins (`SquaredLoss`, `KLLoss`, `BernoulliLoss`, `BoundedSquaredLoss`). New Bregman losses contribute to `rieszreg.losses`. Document which losses your backend supports and validate at construction time — raise a clear error on unsupported losses.
- **Tracer / augmentation.** `LinearForm`, `Tracer`, `trace()`, `Estimand.augment()`, `AugmentedDataset` are inherited.
- **Diagnostics.** Inherit `Diagnostics` and `diagnose(...)`. Subclass `Diagnostics` for backend-specific extras (e.g. `KernelDiagnostics` adds λ_selected, n_support, effective_dof). Reuse the shared warnings (extreme-α̂ flag, etc.) — don't reimplement.
- **`RieszEstimator` orchestrator.** Either expose it directly with your backend baked in, or provide a thin convenience subclass. The convenience subclass is `RieszEstimator` with the backend defaulted; both routes must work and be tested.
- **Serialization helpers.** `factory_spec` registry for built-in estimands; `loss_from_spec(spec)` for losses.

## 4. Hyperparameter handling

- All learner-specific knobs (`n_estimators`, `learning_rate`, `epochs`, `kernel`, `lambda_grid`, `solver`, ...) live as constructor args on the concrete backend dataclass. **Never on `RieszEstimator`.** **Never in the Protocol method kwargs.**
- Convenience subclasses surface them as their own `__init__` args and forward via `_resolved_backend()`.
- If the backend uses a held-out slice for fit-time logic (early stopping, λ selection), expose `validation_fraction` as a constructor attribute. The orchestrator reads it via `getattr(backend, "validation_fraction", 0.0)` and produces the row-level split before augmentation. Backends that don't use the holdout for fit-time logic don't expose it.
- Tuning uses sklearn `GridSearchCV` / `HalvingGridSearchCV` / `RandomizedSearchCV`. **No bespoke `tune_riesz()`.**
- Sample-splitting and cross-fitting use `cross_val_predict`. **No bespoke `crossfit()`.**
- Heuristic resolutions (e.g. krrr's `median`, `scott`, `silverman` length-scale): document them and accept both string and numeric forms.

## 5. Numerical stability

Encapsulate all backend-specific stability tricks inside the backend itself. Examples: xgboost's `hessian_floor=2.0` for counterfactual rows; KL/Bernoulli `max_eta` clipping; per-loss link functions enforcing valid prediction ranges. The meta-package does not impose stability tricks; the `Backend` is responsible.

## 6. Fast-default + customizable-internals

Ship an automatic fast-path that "just works" at default settings. Concrete reference: krrr's solver dispatcher (`auto_choose(n_aug)`: direct ≤3k → nystrom_cg ≤50k → rff/falkon). Underneath the auto-path, expose the slower / more general / more customizable interfaces (explicit solver choice, raw kernel matrices, etc.). Document both surfaces and exercise both in tests.

## 7. Public API surface (`__init__.py`)

Re-export the rieszreg primitives a typical user needs (estimand factories, loss factories, top-level estimator class, `diagnose`, `LinearForm`, `Tracer`) plus your own backend factories and convenience class. The re-export list is invariant across the two backend Protocols — even moment-style packages re-export `LinearForm` and `Tracer`.

Lazy-import optional heavy deps (xgboost, lightgbm, JAX, falkon, keops, torch) via `__getattr__` so the package is importable without them.

## 8. Serialization

Mimic sklearn's serialization story as closely as possible — joblib-compatible pickling for the estimator object, plus a directory-format `save(path)` / `load(path)` that round-trips metadata cleanly.

Implement directory-format save/load: binary payload (booster.ubj, predictor.joblib, kernel coefficients) + `metadata.json` with loss spec, estimand factory_spec, feature_keys, base_score, best_iteration, hyperparams. `load(path, estimand=None)` accepts a re-passed custom `m()` for the non-built-in case. Custom `m()` cannot be serialized in the metadata path; document as a limitation.

## 9. R wrapper (~50 lines)

- Subclass `rieszreg::RieszEstimatorR6`. Bake your backend / defaults via `initialize()`.
- API: `<Pkg>Estimator$new(...)$fit(Z, y)$predict(Z)$score(Z)$diagnose(Z)`. **Not** functional `fit_riesz()` shims. `Z` is a predictor data.frame (treatment + covariates, in `feature_keys` order); `y` is a separate numeric outcome vector.
- Inherit `df_to_py()` for the predictor data.frame conversion.
- Estimand and loss factories are exposed from rieszreg's R package; expose your backend-specific factories on top.
- **No R-side custom `m()`.** The `LinearForm` tracer is Python-only by design. R users needing custom functionals write `m()` in Python and call from R via reticulate.
- Provide `use_python_<pkg>()` to configure the interpreter.

## 10. Required tests

- **Per-estimand smoke tests** against the five built-in factories.
- **Per-loss tests** if your backend supports non-squared losses (gradient/Hessian vs finite-diff).
- **Diagnostics output and warnings** for any extras you add.
- **sklearn-conformance subset** + documented N/A list. Reuse `rieszreg.testing.conformance`.
- **sklearn integration**: `cross_val_predict`, `Pipeline`, `get_params` round-trip.
- **Serialization round-trip** per estimand.
- **Backend equivalence on identical data** if your package has multiple internal modes (boosting backends, kernel solvers).
- **Edge cases**: ndarray vs DataFrame, single-sample, holdout-split edges (when the backend exposes `validation_fraction`).
- **Property-based (Hypothesis) tests** for tracer linearity, loss round-trips, estimand factory specs, augmentation determinism (most are inherited from rieszreg's suite; re-run if you reach into the tracer).
- **Estimator-consistency suite** — required. Use `rieszreg.testing.dgps`. On a small set of analytically tractable DGPs (linear-Gaussian ATE, binary-treatment logistic α₀, ...), with proper tuning and growing n, the learned α̂ must approach the true α₀ (or its functional).
- **R parity test** (`r/<pkg>/tests/testthat/test-parity.R`) confirming bitwise-identical Python ↔ R predictions.
- **Reference parity** *(required when a prior implementation exists)*. The reference must come from a different code path or a different mathematical formulation than your wrapper's. Either is fine: an external repo's algorithm inlined in the test file, or your own re-derivation via a different formulation. What does *not* count is hand-replicating the wrapper's own packing / call sequence and asserting bit-identity. Inline the reference in the test file rather than depending on an external repo at runtime. **A failure does not mean our implementation is worse; it means the two disagree and someone needs to investigate which one has the bug, or what algorithmic / numerical choice differs.** Document the resolution in the test file's docstring. If no prior public implementation exists, say so in the package's CLAUDE.md.

  Existing examples to model on:
  - `packages/rieszboost/python/tests/test_lee_schuler_parity.py`
  - `packages/krrr/python/tests/test_reference_parity.py`
  - `packages/forestriesz/python/tests/test_leaf_optimum.py`

  Also keep a runnable `_compare_with_reference.py` under `examples/<reference>/` that reports Pearson/RMSE on a larger DGP — useful for ad-hoc investigation.

## 11. Examples

- **Per-estimand example rule.** Every built-in estimand factory MUST have a runnable script in `examples/` exercising your backend on a realistic DGP, with EEE / one-step plug-in built around it where applicable. An estimand without an example is a hidden trap.
- **Reference-parity examples.** When a prior implementation exists, add an `examples/<reference>/` directory with synthetic DGPs and a `_compare_with_reference.py` head-to-head script.
- **`examples/README.md`** indexes scripts with a status table — subject to the living-doc rule.
- **Quickstart example rule.** When a new backend, kernel, solver, or other feature is added, add a corresponding example.
- Real-data example datasets (Lalonde NSW + CPS, NHEFS) live in or are sourced via the meta-package; reuse them rather than duplicating loaders.

## 12. Documentation

- **No separate docs site.** Contribute pages to the meta-project Quarto site at `docs/` (under Backends → `<pkg>`).
- Knitr engine; `{r}` chunks executed natively, `{python}` via reticulate. Bilingual `::: {.panel-tabset group="lang"}` panels on every page with executable code. Update both tabs together — the R wrapper is a first-class citizen.
- **Per-package CLAUDE.md** for implementation-side notes (architecture, backend internals, sharp edges). Not user-facing.
- **README per package**: status, why, install, quickstart, link to the meta-project docs. Living-doc rule: any change to public API surface (Python OR R) updates the README and the relevant docs page in the same commit.
- See the `rieszreg-notation` skill for prose conventions and doc-tone rules (enforced by `.githooks/pre-commit`).

## 13. Reference materials

Do NOT host your own arXiv reference index. Contribute to `reference/` at the repo root. Each new package adds its primary citations to the shared index with arXiv IDs and a refetch script.

## 14. Versioning

Semver pre-release (`0.0.1`) initially. Maintain a CHANGELOG. Pin a compatible `rieszreg>=X.Y` range; coordinate breaking changes through the meta-package.
