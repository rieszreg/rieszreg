---
name: rieszreg-architecture
description: Architectural rules for the rieszreg meta-package and its learner backends — tier 1/2/3 classification, the agnostic-orchestrator principle, sklearn-first, lazy imports for optional heavy deps, module separation. Triggers when editing `RieszEstimator`, the `Backend` / `MomentBackend` Protocols, any concrete backend in `packages/*/python/*/backends/`, the public re-exports in any package's `__init__.py`, or when adding a kwarg to any tier-1 object. Especially important when wondering "should this go in rieszreg or in the impl package?" or "should this kwarg live on `RieszEstimator` or on the backend?"
---

# Architectural rules for the rieszreg family

The meta-package `rieszreg` holds every cross-cutting abstraction. Each learner package (rieszboost, krrr, forestriesz, riesznet, riesztree) depends only on `rieszreg` and provides concrete backends. Misclassifying an abstraction is the most common source of design rot in this project — the rules below are the operational tests that catch it.

## 1. Tier classification (load-bearing)

Every abstraction in this family is one of three tiers. Get this wrong and `RieszEstimator` slowly accumulates backend-specific kwargs until it stops being agnostic.

**Tier 1 — universal.** Applies to *every* learner. Lives at the top of `rieszreg`'s public API. Tier-1 objects accept only args that every plausible backend can use meaningfully.

- `Estimand`, `Loss`, `Diagnostics`
- `Backend` and `MomentBackend` Protocols
- `RieszEstimator` orchestrator
- The predictor-loader registry
- `RieszEstimatorR6` base R6 class

**Tier 2 — shared utility.** Used by *multiple* learners but not all. Lives in `rieszreg`, but tier-1 objects invoke them as opt-in services — never bake them in. Tier-1 dispatch may select between tier-2 utilities based on backend type; tier-2 must not appear in any tier-1 object's constructor signature or method-kwarg list.

- `Estimand.augment(features)` / `AugmentedDataset` (consumed by augmentation-style backends)
- `trace(estimand, row)` helper (consumed by moment-style backends)
- `rieszreg.testing.dgps` canonical DGPs

**Tier 3 — learner-specific.** Lives on the concrete backend dataclass in the implementation package. Examples: `n_estimators`, `learning_rate`, `epochs`, `batch_size`, `early_stopping_rounds`, `kernel`, `lambda_grid`, `riesz_feature_fns`, `hessian_floor`, `solver`, `n_landmarks`.

## 2. The agnostic-orchestrator rule

`RieszEstimator.__init__` accepts only tier-1 args:

- `estimand`, `backend`, `loss` — required structural inputs.
- `init`, `random_state` — universal fit-time config.

The `Backend` / `MomentBackend` Protocol method signatures pass only:

- data (`AugmentedDataset` for augmentation-style; `rows + estimand` for moment-style)
- the `loss` spec
- `base_score` (η-space init, computed by the orchestrator from `init` + `loss.alpha_to_eta`)
- `random_state`
- `hyperparams` — a dict for backend-specific passthrough

Backend-specific knobs live as constructor args on the concrete backend dataclass. Convenience subclasses of `RieszEstimator` surface them as their own `__init__` args and forward via `_resolved_backend()`. Example: `RieszBooster(n_estimators=200, learning_rate=0.05)` builds `XGBoostBackend(n_estimators=200, learning_rate=0.05)` in `_resolved_backend()`. The orchestrator does not see `n_estimators`.

`validation_fraction` is per-backend, not tier-1. Backends that use a held-out slice for fit-time logic expose `validation_fraction` as a constructor attribute. The orchestrator reads it via `getattr(backend, "validation_fraction", 0.0)` and produces the row-level split before augmentation. Backends that don't use a holdout for fit-time logic don't expose it; users wanting held-out loss reporting on those backends pass `eval_set=` at fit time.

## 3. The "would-be-ignored" lint test

Before adding a kwarg to a tier-1 object, ask: **"would any plausible backend ignore this kwarg?"** If yes, it is tier-2 or tier-3 and must move out.

Examples the rule catches:

- ❌ `RieszEstimator(n_estimators=...)` — kernel ridge and forests ignore. Tier 3.
- ❌ `RieszEstimator(learning_rate=...)` — kernel ridge ignores. Tier 3.
- ❌ `RieszEstimator(early_stopping_rounds=...)` — kernel ridge and forests ignore. Tier 3.
- ❌ `RieszEstimator(epochs=...)` — only neural backends. Tier 3.
- ❌ `diagnose(booster=...)` — name claims every estimator is a booster. Use `estimator=`.
- ❌ `RieszEstimator(validation_fraction=...)` — forest backends don't use the held-out slice for fit-time logic. Tier 3 → on backends that need it; the orchestrator reads via `getattr` for the split.
- ✅ `RieszEstimator(random_state=...)` — every backend seeds randomness somewhere.
- ✅ `RieszEstimator(init=...)` — every loss has `best_constant_init(m_bar)`; every backend gets `base_score` from it.

A correct tier-1 abstraction reads cleanly under any plausible new backend you might add — Bayesian net, monotonic spline, deep kernel, transformer. If imagining a hypothetical backend forces you to add a sentinel default or a `del kwarg_i_will_ignore` line in the new backend's `fit_*`, the abstraction is wrong. Move the kwarg down a tier.

## 4. The two backend Protocols

The orchestrator dispatches at fit time based on which Protocol the backend implements:

```python
# inside RieszEstimator.fit
if hasattr(backend, "fit_rows") and not hasattr(backend, "fit_augmented"):
    result = backend.fit_rows(rows_train, rows_valid, self.estimand, loss, ...)
else:
    aug_train = self.estimand.augment(feats_train, ys=ys_train)        # tier-2 service
    aug_valid = self.estimand.augment(feats_valid, ys=ys_valid) if feats_valid is not None else None
    result = backend.fit_augmented(aug_train, aug_valid, loss, ...)
```

- **Augmentation-style** (`Backend.fit_augmented`) — for learners whose loss decomposes naturally over the augmented `(a, b)` evaluation points. Examples: `KernelRidgeBackend` (krrr), `XGBoostBackend` / `SklearnBackend` (rieszboost).
- **Moment-style** (`MomentBackend.fit_rows`) — for learners whose loss decomposes per original sample row. Backend computes per-row moments via `rieszreg.trace(estimand, row, y)` directly, avoiding the augmentation blow-up. Examples: `ForestRieszBackend` (forestriesz), `TorchBackend` (riesznet).

Backends implementing both default to `fit_augmented` for back-compat.

## 5. Where shared abstractions live

| What | Goes in | Not in |
|---|---|---|
| New `Estimand` factory | `packages/rieszreg/python/rieszreg/estimands/` | impl package |
| New `Loss` subclass | `packages/rieszreg/python/rieszreg/losses/` | impl package |
| Tracer / `LinearForm` / `AugmentedDataset` | `packages/rieszreg/python/rieszreg/` (already there) | impl package — reuse |
| Backend `fit_augmented` or `fit_rows` impl | `packages/<pkg>/python/<pkg>/backends/` | rieszreg |
| Backend hyperparameter (e.g. `kernel`, `n_estimators`) | concrete backend dataclass | `RieszEstimator` |
| Convenience subclass (`RieszBooster`, `KernelRieszRegressor`) | `packages/<pkg>/python/<pkg>/` | rieszreg |
| Per-package R wrapper subclassing `RieszEstimatorR6` | `packages/<pkg>/r/<pkg>/R/` | rieszreg |

Implementation packages depend on `rieszreg`. They never depend on each other. If you find yourself importing from a sibling impl package, the abstraction belongs in `rieszreg`.

## 6. sklearn-first

Before writing any procedural code with loops, splits, grids, or folds, ask **"is there an sklearn way?"**. If yes, use it.

- Cross-fitting → `cross_val_predict` / `cross_validate` / `KFold` / `StratifiedKFold`
- Hyperparameter tuning → `GridSearchCV` / `HalvingGridSearchCV` / `RandomizedSearchCV`
- Train/test split → `train_test_split`
- Composition → `Pipeline` / `ColumnTransformer` / `FunctionTransformer`
- Custom scorer → `make_scorer`
- Parallelism → `n_jobs=`

Hand-rolled fold loops are a code smell. Bespoke is reserved for things sklearn genuinely doesn't cover — the `LinearForm` tracer, the custom xgboost objective, the Bregman `Loss`. **No bespoke `crossfit()` or `tune_riesz()`.**

`RieszEstimator` inherits from `BaseEstimator`. `get_params` / `set_params`, `clone`, `Pipeline`, `GridSearchCV`, `cross_val_predict` must all compose. Anything that breaks composition is a regression.

## 7. Public-API rules

- **No input-schema args on `fit` / `predict`.** No `feature_keys=` or similar. The `Estimand` owns its input schema; if a new estimand needs different inputs, that's a property of the estimand object.
- **`fit(Z, y=None)`** — `Z` is the predictor matrix (treatment + covariates), ndarray (columns matched to `estimand.feature_keys`) or DataFrame (matched by name). `y` is the per-row outcome vector (sklearn-style); built-in estimands ignore it, custom Y-dependent estimands read it.
- **`predict(Z)`** returns shape `(n,)` array of α̂.
- **`score(Z, y=None)`** returns `−mean(Riesz loss)` (sklearn higher-is-better convention).
- **Public re-exports.** Each impl package's `__init__.py` re-exports the rieszreg primitives users typically need (estimand factories, loss factories, top-level estimator, `diagnose`, `LinearForm`, `Tracer`) plus its own backend factories and convenience class. The re-export list is invariant across the two backend Protocols — even moment-style packages re-export `LinearForm` and `Tracer`.

## 8. Lazy imports for optional heavy deps

Optional heavy deps (xgboost, lightgbm, JAX, falkon, keops, torch) must lazy-load via `__getattr__` so the package is importable without them. Reference pattern: [packages/rieszboost/python/rieszboost/__init__.py](../../packages/rieszboost/python/rieszboost/__init__.py) (the `__getattr__` block near the bottom).

## 9. Module separation of concerns

Keep this seam structure:

- `estimands/` — schema + functional
- `losses/` — Bregman link / grad / Hessian
- `tracer.py` + `augmentation.py` — symbolic linear-form algebra; used by augmentation-style backends. Moment-style backends call `trace` directly.
- `backends/` — algorithm-specific `fit_augmented` *or* `fit_rows`
- `estimator.py` — sklearn wrapper that orchestrates and dispatches between the two backend paths
- `diagnostics.py` — health checks
- `serialization.py` — save/load + factory_spec
- `testing/` — DGPs and conformance helpers

In an impl package, `backends/` is what you actually own. Backend-specific code lives in `backends/<backend>.py`. Everything else comes from `rieszreg`.

## 10. What NOT to do

- Don't redefine `Estimand`, `FiniteEvalEstimand`, `Loss`, `AugmentedDataset`, `Diagnostics`, `RieszEstimator`, `LinearForm`, `Tracer`, factory-spec registries, or testing DGPs in an impl package — import from `rieszreg`.
- Don't depend on another impl package (rieszboost, krrr, ...). Depend on `rieszreg`.
- Don't add a custom-`m()` R entry point. The `LinearForm` tracer is Python-only by design; R users needing custom functionals write `m()` in Python and call from R via reticulate.
- Don't reinvent `cross_val_predict`, `GridSearchCV`, or any sklearn primitive.
- Don't put `feature_keys=` (or any input-schema arg) on `fit` / `predict`.
- Don't put learner-specific knobs (`n_estimators`, `learning_rate`, `epochs`, `batch_size`, `early_stopping_rounds`, `kernel`, ...) on `RieszEstimator` or in the Protocol method kwargs. They live on the backend dataclass; convenience subclasses forward via `_resolved_backend()`.
- Don't add backend-specific framing to tier-1 docs ("the booster does X", "the kernel matrix is Y"). Use neutral language ("the backend produces η").
- Don't introduce a bespoke `crossfit()` or `tune_riesz()`.

## 11. Data flow

Construct estimand + loss + backend → `RieszEstimator` (or a convenience subclass) → `.fit(Z)` calls `estimand.augment(features)` (built-in subclasses emit augmented `(a, b)` rows in vectorised numpy; custom estimands trace `m` row-wise via a `LinearForm` and emit the same shape), or `.fit(Z)` calls `backend.fit_rows(...)` directly for moment-style backends → backend consumes → predictor returned → `.predict(Z)` applies link → α̂.

`m()` is JAX-style opaque. The tracer enforces linearity; any non-linear op raises. Fast path = augmentation + closed-form-friendly fitting. The slow general path (Friedman gradient boosting against arbitrary base learners for non-finite-point `m`) is on the roadmap; do not block on it.
