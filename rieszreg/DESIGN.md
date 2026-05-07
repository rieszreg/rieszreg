# RieszReg meta-package design + learner-package guide

This document does two things:

1. **Part A** designs a shared meta-package — `rieszreg`, in the spirit of sklearn — that holds every cross-cutting abstraction so each implementation package depends on it instead of duplicating code or cross-importing.
2. **Part B** is the detailed checklist + design directive that a new "learner" implementation package (rieszboost, krrr, future) MUST follow to plug into the ecosystem.

Four learner packages exist today: `rieszboost` (gradient boosting), `krrr` (kernel ridge), `forestriesz` (random forests), and `riesznet` (neural networks). Each depends only on `rieszreg`.

Reference paper: Lee & Schuler 2025 ([arXiv:2501.04871](https://arxiv.org/abs/2501.04871)). Part A moves the shared `reference/` to the meta-project level.

Tags used in Part B:
- **[from rieszreg]** — import or inherit; do not redefine.
- **[your package]** — implement in the learner package.
- **[design rule]** — a philosophy the learner package must follow.

**Notation.** Math, prose, and code-naming conventions for every package, doc, docstring, comment, and example in the family live in the [`rieszreg-notation` skill](../.claude/skills/rieszreg-notation/SKILL.md). Headlines: outcome regression is $\mu$, estimand is $\psi$, functional notation is $m(\mu)(Z)$, data tuple is $Z = (A, X)$, algorithms are called "learners". Read the skill before introducing or revising any of these.

---

# Part A — Meta-package design: `rieszreg`

## A.1 Goals

- A single Python package + R package pair that holds **every shared abstraction** so implementation packages (rieszboost, krrr, future) depend on it instead of duplicating or cross-importing.
- Mirror sklearn's structure: meta-package provides the framework (BaseEstimator-equivalent, protocols, primitives); implementation packages provide concrete backends and convenience wrappers.
- Single docs site at the meta-package level. Single `reference/`. Shared base R6 class.
- Eliminate the current code duplication: krrr today imports `Estimand`, `LinearForm`, `Loss`, `AugmentedDataset`, `Diagnostics`, and `RieszBooster` from rieszboost. Those imports should redirect to `rieszreg`.

## A.2 Layering: tier-1, tier-2, tier-3

Three tiers determine where an abstraction lives. Misclassifying an abstraction is the most common source of design rot in this project — the rule below is the operational test that catches it.

**Tier 1 — universal.** Applies to *every* learner. Lives at the top of `rieszreg`'s public API: `Estimand`, `Loss`, `Diagnostics`, the `Backend` / `MomentBackend` Protocols, `RieszEstimator` orchestrator, the predictor-loader registry, `RieszEstimatorR6` base R6 class. Tier-1 objects accept only args that every plausible backend can use meaningfully.

**Tier 2 — shared utility.** Used by *multiple* learners but not all. `Estimand.augment(features)` / `AugmentedDataset` (consumed by augmentation-style backends), `trace(estimand, row)` helper (consumed by moment-style backends), the `rieszreg.testing.dgps` canonical DGPs (used by every package's consistency suite, but optional). Lives in `rieszreg`, but tier-1 objects invoke them as opt-in services — never bake them in. Tier-1 dispatch may select between tier-2 utilities based on backend type; tier-2 must not appear in any tier-1 object's constructor signature or method-kwarg list.

**Tier 3 — learner-specific.** `n_estimators`, `learning_rate`, `epochs`, `batch_size`, `early_stopping_rounds`, `kernel`, `lambda_grid`, `riesz_feature_fns`, `hessian_floor`, `solver`, `n_landmarks`, etc. Lives in implementation packages, on the concrete backend dataclass.

### The agnostic-orchestrator rule

`RieszEstimator.__init__` accepts only tier-1 args:
- `estimand`, `backend`, `loss` — required structural inputs.
- `init`, `random_state` — universal fit-time config.

The `Backend` / `MomentBackend` Protocol method signatures pass only:
- data (`AugmentedDataset` for augmentation-style, `rows + estimand` for moment-style).
- the `loss` spec.
- `base_score` (η-space init, computed by the orchestrator from `init` + `loss.alpha_to_eta`).
- `random_state`.
- `hyperparams` — a dict for backend-specific passthrough (e.g. xgboost's `max_depth`, `reg_lambda`).

Backend-specific knobs live as constructor args on the concrete backend dataclass. Convenience subclasses of `RieszEstimator` surface them as their own `__init__` args and forward via `_resolved_backend()`. For example: `RieszBooster(n_estimators=200, learning_rate=0.05, early_stopping_rounds=10)` builds `XGBoostBackend(n_estimators=200, learning_rate=0.05, early_stopping_rounds=10)` in `_resolved_backend()`. The orchestrator does not see `n_estimators`.

`validation_fraction` is per-package, not tier-1. Backends that use a held-out slice for fit-time logic (early stopping in `XGBoostBackend` / `SklearnBackend` / `TorchBackend`, λ selection in `KernelRidgeBackend`) expose `validation_fraction` as a constructor attribute. The orchestrator reads it via `getattr(backend, "validation_fraction", 0.0)` and produces the row-level split before augmentation. Backends that don't use a holdout for fit-time logic (`ForestRieszBackend`, `AugForestRieszBackend`) don't expose it; users wanting held-out loss reporting on a forest pass `eval_set=` at fit time.

### The "would-be-ignored" lint test

Before adding a kwarg to a tier-1 object, ask: **"would any plausible backend ignore this kwarg?"** If yes, it is tier-2 or tier-3 and must move out. Examples the rule catches:

- ❌ `RieszEstimator(n_estimators=...)` — kernel ridge and forests ignore. Tier 3 → on `XGBoostBackend`, `SklearnBackend`, `TorchBackend`.
- ❌ `RieszEstimator(learning_rate=...)` — kernel ridge ignores; the optimizer owns it for neural backends. Tier 3.
- ❌ `RieszEstimator(early_stopping_rounds=...)` — kernel ridge and forests ignore. Tier 3.
- ❌ `RieszEstimator(epochs=...)` — only neural backends. Tier 3.
- ❌ `diagnose(booster=...)` — the kwarg name claims every estimator is a booster. Use `estimator=`.
- ❌ `RieszEstimator(validation_fraction=...)` — forest backends don't use the held-out slice for fit-time logic, only reporting. Tier 3 → on the backends that use it (XGBoost, Sklearn, KernelRidge, Torch); read by the orchestrator via `getattr` for the split.
- ✅ `RieszEstimator(random_state=...)` — every backend seeds randomness somewhere.
- ✅ `RieszEstimator(init=...)` — every loss has `best_constant_init(m_bar)`; every backend gets `base_score` from it.

### What tier-2 utilities look like in code

The orchestrator's `fit` chooses between two tier-2 services based on which Protocol the backend implements:

```python
# tier-1 dispatch logic inside RieszEstimator.fit
if hasattr(backend, "fit_rows") and not hasattr(backend, "fit_augmented"):
    result = backend.fit_rows(
        rows_train, rows_valid, self.estimand, loss,
        ys_train=ys_train, ys_valid=ys_valid, **common_kwargs,
    )
else:
    aug_train = self.estimand.augment(feats_train, ys=ys_train)        # tier-2 service
    aug_valid = self.estimand.augment(feats_valid, ys=ys_valid) if feats_valid is not None else None
    result = backend.fit_augmented(aug_train, aug_valid, loss, **common_kwargs)
```

Neither `Estimand.augment` nor `trace` appears in `RieszEstimator.__init__`. The orchestrator selects between them internally; the user-facing surface stays uniform.

### When in doubt

A correct tier-1 abstraction reads cleanly under any plausible new backend you might add — Bayesian neural network, monotonic spline, deep kernel learner, transformer, anything. If imagining a hypothetical backend forces you to add a sentinel default or a `del kwarg_i_will_ignore` line in the new backend's `fit_*`, the abstraction is wrong. Move the kwarg down a tier.


## A.3 Python module layout (`rieszreg/`)

```
rieszreg/
├── estimands/
│   ├── __init__.py        # ATE, ATT, TSM, AdditiveShift, LocalShift
│   ├── base.py            # Estimand base + ATE/ATT/TSM/AdditiveShift/LocalShift subclasses + factory_spec registry
│   └── tracer.py          # LinearForm, Tracer, trace
├── losses/
│   ├── __init__.py
│   ├── base.py            # Loss base class, loss_from_spec
│   ├── squared.py
│   ├── kl.py
│   ├── bernoulli.py
│   └── bounded_squared.py
├── augmentation.py        # AugmentedDataset packaging
├── backends/
│   └── base.py            # Backend Protocol, FitResult, Predictor base
├── diagnostics.py         # Diagnostics dataclass, diagnose function
├── estimator.py           # RieszEstimator (sklearn BaseEstimator orchestrator)
├── serialization.py       # directory save/load helpers, factory_spec round-trip
├── testing/
│   ├── dgps.py            # canonical DGPs for consistency tests (linear-Gaussian ATE, etc.)
│   ├── conformance.py     # sklearn-conformance check helpers
│   └── parity.py          # reference-parity utilities
└── __init__.py            # public re-exports (sklearn-style)
```

`RieszEstimator` is the orchestrator — it takes `(estimand, loss, backend)` and implements `fit/predict/score/diagnose/save/load`. It is the *only* sklearn-compatible class in the meta-package. Implementation packages either expose it directly with their backend baked in, or provide a thin convenience subclass (`RieszBooster` in rieszboost, `KernelRieszRegressor` in krrr).

## A.4 Implementation-package layout (each learner package)

```
<pkg>/
├── python/<pkg>/
│   ├── backends/          # ONLY backend implementations (XGBoostBackend, KernelRidgeBackend, etc.)
│   ├── <pkg>_specific.py  # things unique to the package (e.g., kernel algebra in krrr)
│   ├── convenience.py     # optional thin subclass of rieszreg.RieszEstimator with backend baked in
│   └── __init__.py        # re-exports rieszreg primitives + the package's own additions
├── r/<pkg>/               # subclass of rieszreg's R6 base
├── examples/              # per-estimand + per-backend examples
├── tests/                 # backend-specific tests + the consistency suite from rieszreg.testing
└── pyproject.toml         # depends on rieszreg
```

## A.5 R-side: shared base R6 class

- `rieszreg/r/rieszreg/R/rieszreg.R` defines `RieszEstimatorR6` as the base R6 class with `$fit/$predict/$score/$diagnose/$save/$load`, `df_to_py()` helper, and base estimand/loss factories.
- Per-package R wrappers: `R6::R6Class("KernelRieszRegressor", inherit = rieszreg::RieszEstimatorR6, public = list(initialize = function(...) { super$initialize(backend = krrr_pkg$KernelRidgeBackend(...)) }))`.
- A new package's R wrapper should be ~50 lines (subclass, initialize, expose backend factory) instead of ~300.

## A.6 Single meta-project docs site

- Quarto site at `RieszReg/docs/` (or `rieszreg/docs/`). Sklearn-style sectioning: Concepts → Estimands → Losses → Backends → [Boosting | Kernel | …] → API reference → R interface → References.
- Backend pages either authored directly inside the meta-site or pulled from each implementation package's `docs-fragment/` directory at build time.
- Single `.github/workflows/docs.yml` builds and deploys the unified site.

## A.7 Shared `reference/` and shared CI templates

- Move arXiv-paper index to `RieszReg/reference/` (top level). Each package's existing `reference/` is removed.
- Shared `.github/workflows/` templates for testing and docs build live in the meta-package.
- Shared `.githooks/pre-commit` template (living-doc rule + doc-tone rules) sourced from the meta-package.

## A.8 Dependency graph

```
rieszreg (no deps on impl packages)
   ↑
   ├── rieszboost      (provides XGBoostBackend, SklearnBackend, optional convenience RieszBooster)
   ├── krrr            (provides KernelRidgeBackend, kernels, solvers, optional KernelRieszRegressor)
   └── <future-pkg>    (provides its backend(s) + thin convenience class)
```

Users compose either way — both are first-class:

```python
# Compose explicitly: pick a backend from any impl package and pass to RieszEstimator
from rieszreg import RieszEstimator, ATE, SquaredLoss
from rieszboost.backends import XGBoostBackend
est = RieszEstimator(estimand=ATE(), loss=SquaredLoss(), backend=XGBoostBackend(n_estimators=200))
est.fit(Z)
```

```python
# Or use the per-package convenience class with smart defaults baked in
from rieszboost import RieszBooster
booster = RieszBooster(estimand=ATE(), n_estimators=200)
booster.fit(Z)
```

Both routes must be tested and documented. The convenience class is just `RieszEstimator` with the backend defaulted.

## A.9 Migration sequencing

1. Create `rieszreg/` skeleton: estimands, losses, augmentation, backends/base, diagnostics, RieszEstimator orchestrator, serialization, testing utilities. Re-host the shared modules currently in rieszboost.
2. Update rieszboost to depend on rieszreg; reduce rieszboost to (a) backend implementations, (b) optional convenience class. Tests still pass.
3. Update krrr to depend on rieszreg (it currently depends on rieszboost). Remove rieszreg-equivalent cross-imports.
4. Move `reference/` to top level. Delete per-package copies.
5. Build the unified Quarto docs site at the meta level. Migrate the rieszboost docs pages in.
6. Build the shared R6 base class + helpers in `rieszreg`'s R package; update both rieszboost's and krrr's R wrappers to subclass it.
7. Add the meta-package's testing utilities (DGPs, consistency tests) and have each implementation package import + run them in CI.
8. Wire shared pre-commit hook + CI workflow templates.
9. Publish version pins so impl packages declare a compatible `rieszreg>=X.Y` range.

---

# Part B — Learner-package guide (detailed checklist + directives)

This is the contract every implementation package must meet. Section structure follows the categories an implementation team thinks in. File-path references point to the existing rieszboost and krrr code where the pattern is concretely instantiated.

## 1. Theoretical / statistical capabilities

### 1.1 Estimands
- **[from rieszreg]** Import the abstract `Estimand` base, the concrete `FiniteEvalEstimand` subclass, and the five built-in subclasses (`ATE`, `ATT`, `TSM(level)`, `AdditiveShift(delta)`, `LocalShift(delta, threshold)`). All built-ins are subclasses of `FiniteEvalEstimand` with vectorised `augment()` overrides. Reference: [base.py](rieszreg/python/rieszreg/estimands/base.py). `StochasticIntervention` previously appeared here; it is currently stubbed (raises `NotImplementedError`) and will be reintroduced.
- **[from rieszreg]** Support custom estimands via user-supplied `m(alpha)(z, y) -> LinearForm` wrapped in a `FiniteEvalEstimand`. Do not bypass the tracer; linearity violations must raise. The per-row outcome `y` flows in sklearn-style: separate from `Z` everywhere. `m`'s inner closure declares `def inner(z, y=None)`; built-ins ignore the second arg.
- **[from rieszreg]** `trace()`, `Estimand.augment()`, and `RieszEstimator.fit()` accept only `FiniteEvalEstimand`. The base `Estimand` class is reserved for future subclasses outside the finite-evaluation algebra.
- **[from rieszreg]** Honor the partial-parameter distinction (ATT and LocalShift fit partial representers; full ATT/LASE require delta-method downstream). Document this in any examples.
- **[design rule]** If a new estimand factory belongs in the family at large, contribute it back to `rieszreg.estimands`, not to your package. A learner package never owns an `Estimand` factory.

### 1.2 Losses
- **[from rieszreg]** Import the `Loss` base class and the built-in losses (`SquaredLoss`, `KLLoss`, `BernoulliLoss`, `BoundedSquaredLoss`). Reference: [losses.py](rieszboost/python/rieszboost/losses.py).
- **[design rule]** New Bregman losses contribute to `rieszreg.losses`, not to your package. The learner package's job is to consume `Loss` instances, not to define them.
- **[your package]** Document which losses your backend supports (e.g., krrr's `KernelRidgeBackend` is `SquaredLoss`-only today). Validate at construction time and raise a clear error if an unsupported loss is passed.

### 1.3 Identification / debiasing
- **[from rieszreg]** `LinearForm` tracer ([tracer.py](rieszboost/python/rieszboost/tracer.py)), the `Estimand.augment(features, ys=None)` method, and `AugmentedDataset` ([augmentation.py](rieszboost/python/rieszboost/augmentation.py)) are inherited. Do not reimplement.
- **[design rule]** Sample-splitting and cross-fitting use sklearn's `cross_val_predict`. Do not introduce a bespoke `crossfit()` function.
- **[design rule]** `score(Z)` returns `−mean(per-row Riesz loss)` (sklearn "higher is better").

### 1.4 Diagnostics
- **[from rieszreg]** Inherit the base `Diagnostics` dataclass and the `diagnose(...)` function. Reference: [diagnostics.py](rieszboost/python/rieszboost/diagnostics.py).
- **[your package]** Subclass `Diagnostics` with backend-specific extras when warranted (krrr's `KernelDiagnostics` adds λ_selected, n_support, effective_dof, condition_number, ill-conditioning warnings). Use the same pattern.
- **[from rieszreg]** Reuse the shared warnings (extreme-α̂ flag, etc.) — don't reimplement.

---

## 2. Computational architecture

### 2.1 Backend Protocol
- **[your package]** Implement *at least one* of two Protocols from `rieszreg.backends.base`. Both return `FitResult(predictor, best_iteration, best_score, history)`. Pick whichever fits your learner's natural loss decomposition:
  - `Backend.fit_augmented(aug_train, aug_valid, loss, ...)` — for learners whose loss decomposes naturally over the augmented `(a, b)` evaluation points (kernel ridge, gradient boosting). Implementations: `KernelRidgeBackend` (krrr), `XGBoostBackend` / `SklearnBackend` (rieszboost).
  - `MomentBackend.fit_rows(rows_train, rows_valid, estimand, loss, *, ys_train=None, ys_valid=None, ...)` — for learners whose loss decomposes per original sample row (random forests, neural nets). Such backends compute per-row moments via `rieszreg.trace(estimand, row, y)` directly, avoiding the augmentation blow-up. The `estimand` argument is a `FiniteEvalEstimand`. `ys_train` / `ys_valid` carry the per-row outcome (sklearn-style) for estimands whose `m` reads it; they are `None` otherwise. Implementations: `ForestRieszBackend` (forestriesz).
- **[design rule]** The orchestrator dispatches at fit time: if the backend exposes `fit_rows` and not `fit_augmented`, the moment path is used; otherwise the augmented path. Backends implementing both default to `fit_augmented` for back-compat.
- **[your package]** Return a `Predictor` with `predict_eta()` and `predict_alpha()` (link applied). Inherit base interface from rieszreg; storage format is your choice.
- **[your package]** `FitResult` shape must match the protocol so `RieszEstimator` can orchestrate uniformly.

### 2.2 Backend implementations
- **[your package]** Concrete backends live in `<pkg>/backends/`. Examples in the wild: `XGBoostBackend(hessian_floor=2.0, gradient_only=False)` ([backends/xgboost.py:93](rieszboost/python/rieszboost/backends/xgboost.py:93)), `SklearnBackend(base_learner_factory)` ([backends/sklearn.py:93](rieszboost/python/rieszboost/backends/sklearn.py:93)), `KernelRidgeBackend` (krrr), `ForestRieszBackend` (forestriesz; moment-style, satisfies `MomentBackend.fit_rows`).
- **[design rule]** Lazy-import optional heavy deps (xgboost, lightgbm, JAX, falkon, keops, torch) via `__getattr__` so the package is importable without them. Reference: [__init__.py:52-64](rieszboost/python/rieszboost/__init__.py:52).

### 2.3 Hyperparameter tuning
- **[design rule]** Tuning uses sklearn `GridSearchCV` / `HalvingGridSearchCV` / `RandomizedSearchCV`. No bespoke `tune_riesz()`.
- **[from rieszreg]** The orchestrator performs the row-level holdout split. With explicit `eval_set=` at fit time it uses that; otherwise it reads `validation_fraction` off the backend via `getattr` and splits before augmentation. Backends that need the holdout for fit-time logic (early stopping, λ selection) expose `validation_fraction` as a constructor attribute.
- **[your package]** Expose backend-specific hyperparameters as constructor args (boosting: `n_estimators`, `learning_rate`, `max_depth`, `reg_lambda`, `subsample`, `early_stopping_rounds`, `validation_fraction`; kernel: `lambda_grid`, `validation_fraction`, `solver`, `n_landmarks`, `n_features`, `cg_tol`). Forest-style backends, which do not use the holdout for fit-time logic, do not expose `validation_fraction`.
- **[your package]** If your backend has heuristic resolutions (krrr's `median`, `scott`, `silverman` length-scale), document them and accept both string and numeric forms.

### 2.4 Numerical stability
- **[your package]** Encapsulate all backend-specific stability tricks inside the backend itself. Examples: xgboost's `hessian_floor=2.0` for counterfactual rows; KL/Bernoulli `max_eta` clipping; per-loss link functions enforcing valid prediction ranges.
- **[design rule]** The meta-package does not impose stability tricks; the `Backend` is responsible.

### 2.5 Fast-default + customizable-internals
- **[design rule]** Ship an automatic fast-path that "just works" at default settings. Concrete reference: krrr's solver dispatcher (`auto_choose(n_aug)`: direct ≤3k → nystrom_cg ≤50k → rff/falkon).
- **[design rule]** Underneath the auto-path, expose the slower / more general / more customizable interfaces (explicit solver choice, raw kernel matrices, etc.). Users with unusual problems should not be locked out.
- **[your package]** Document both surfaces in docs and exercise both in tests.

---

## 3. API & organizational design

### 3.1 sklearn-compatibility (load-bearing)
- **[design rule]** Inherit `BaseEstimator`. `get_params`/`set_params`, `clone`, `Pipeline`, `GridSearchCV`, `cross_val_predict` must all compose. Anything that breaks composition is a regression.
- **[design rule]** `.fit(Z, y=None)` accepts ndarray (columns matched to `estimand.feature_keys`) or DataFrame (columns matched by name). The predictor matrix is named `Z` (treatment + covariates); see notation rule 4. `y` is a separate per-row outcome vector (sklearn convention); the orchestrator plumbs it into `m(alpha)(z, y)` and the augmentation / moment paths. Built-in estimands ignore `y`; custom Y-dependent estimands read it.
- **[design rule]** `.predict(Z)` returns shape `(n,)` array of α̂.
- **[design rule]** `.score(Z, y=None)` returns `−mean(Riesz loss)`.
- **[your package]** Acceptance gates in tests for `clone`, `GridSearchCV`, `cross_val_predict` (re-use `rieszreg.testing.conformance`).

### 3.2 Public-API rules
- **[design rule]** ngboost/sklearn-style: object-oriented factories that bake configuration in at construction (estimand, loss, backend, hyperparameters); fit/predict/score/diagnose at use time.
- **[design rule]** No `feature_keys` (or any input-schema arg) on `fit()` / `predict()`. The Estimand owns its input schema. If a new estimand needs different inputs, that's a property of the estimand object.
- **[design rule]** Cross-fitting == `cross_val_predict`. Tuning == `GridSearchCV`. Don't reinvent.
- **[design rule]** Swappable orthogonal components: backend, loss, estimand are independent.
- **[design rule: agnostic orchestrator]** All learner-specific knobs (`n_estimators`, `learning_rate`, `epochs`, `batch_size`, `early_stopping_rounds`, `kernel`, `lambda_grid`, `riesz_feature_fns`, …) live as constructor args on the concrete backend dataclass — never on `RieszEstimator` and never in the `Backend`/`MomentBackend` Protocol method kwargs. Convenience subclasses surface them as their own ctor args and forward via `_resolved_backend()`. See §A.2 for the layering principle and the would-be-ignored lint test that catches violations.
- **[design rule: sklearn-first, every feature]** Before writing any procedural code with loops, splits, grids, or folds, ask *"is there an sklearn way?"*. If yes, use it (`cross_val_predict`, `cross_validate`, `KFold`, `train_test_split`, `StratifiedKFold`, `GridSearchCV`, `HalvingGridSearchCV`, `RandomizedSearchCV`, `Pipeline`, `ColumnTransformer`, `FunctionTransformer`, `make_scorer`, `n_jobs=`). Hand-rolled fold loops are a code smell. Bespoke is reserved for things sklearn genuinely doesn't cover (the `LinearForm` tracer, the custom xgboost objective, the Bregman `Loss`).

### 3.3 Module separation of concerns
- **[design rule]** Keep the seam structure that already works: `estimands/` (schema + functional), `losses/` (Bregman link/grad/Hessian), `tracer.py` + `augmentation.py` (symbolic linear-form algebra and dataset assembly — used by augmentation-style backends; moment-style backends call `trace` directly), `backends/` (algorithm-specific `fit_augmented` *or* `fit_rows`), `estimator.py` (sklearn wrapper that orchestrates and dispatches between the two backend paths), `diagnostics.py` (health checks), `serialization.py` (save/load + factory_spec), `testing/` (DGPs and conformance helpers).
- **[your package]** Most of these come from `rieszreg`; in your package, `backends/` is what you actually own. Backend-specific code lives in `backends/<backend>.py`.

### 3.4 Public API surface (entry points in `__init__.py`)
- **[your package]** Re-export the rieszreg primitives a typical user needs (estimand factories, loss factories, top-level estimator class, `diagnose`, `LinearForm`, `Tracer`) plus your own backend factories and convenience class. Pattern: [__init__.py:14-49](rieszboost/python/rieszboost/__init__.py:14).
- **[design rule]** The re-export list is invariant across the two backend Protocols: even moment-style packages re-export `LinearForm` and `Tracer` so users can author custom `m()`s the same way (the backend just consumes them through `trace(estimand, row)` instead of `Estimand.augment`). Backend choice is an internal implementation detail; the user-facing surface is one `RieszEstimator` subclass with `fit / predict / score / diagnose`.
- **[design rule]** Lazy `__getattr__` for any symbol that pulls in optional heavy deps.

### 3.5 Serialization & persistence
- **[design rule]** Mimic sklearn's serialization story as closely as possible — joblib-compatible pickling for the estimator object, plus a directory-format `save(path)` / `load(path)` that round-trips metadata cleanly.
- **[from rieszreg]** `factory_spec` registry for built-in estimands and `loss_from_spec(spec)` for losses are inherited.
- **[your package]** Implement directory-format save/load: binary payload (booster.ubj, predictor.joblib, kernel coefficients) + `metadata.json` with loss spec, estimand factory_spec, feature_keys, base_score, best_iteration, hyperparams. `load(path, estimand=None)` accepts a re-passed custom `m()` for the non-built-in case.
- **[design rule]** Custom `m()` cannot be serialized in the metadata path; document as a limitation.

### 3.6 Data-flow conventions
- **[design rule]** Honor the canonical data flow: construct estimand + loss + backend → orchestrator estimator → `.fit(Z)` calls `estimand.augment(features)` → built-in subclasses emit augmented `(a, b)` rows in vectorised numpy; custom estimands trace `m` row-wise via a LinearForm and emit the same shape → backend consumes → predictor returned → `.predict(Z)` applies link → α̂.
- **[design rule]** `m()` is JAX-style opaque; the tracer enforces linearity. Any non-linear op raises and signals slow-path dispatch.
- **[design rule]** Fast path = augmentation + closed-form-friendly fitting. The slow general path (Friedman gradient boosting against arbitrary base learners for non-finite-point m) is on the roadmap; do not block on it.

---

## 4. R / Python parity

- **[design rule]** Python is primary; R is a thin reticulate wrapper.
- **[design rule]** R6 class API: `<Pkg>Estimator$new(...)$fit(Z, y)$predict(Z)$score(Z)$diagnose(Z)`. NOT functional `fit_riesz()` shims. `Z` is a predictor data.frame (treatment + covariates, in `feature_keys` order); `y` is a separate numeric outcome vector (sklearn convention).
- **[from rieszreg]** Subclass `rieszreg::RieszEstimatorR6`. Bake your backend / defaults via `initialize()`. Goal: ~50 lines per package R wrapper.
- **[from rieszreg]** Inherit `df_to_py()` to convert the predictor data.frame `Z` to a pandas DataFrame.
- **[from rieszreg]** Estimand and loss factories are exposed from rieszreg's R package; expose your backend-specific factories on top.
- **[design rule]** No R-side custom `m()`. The `LinearForm` tracer is Python-only by design. R users needing custom functionals write `m()` in Python and call from R via reticulate.
- **[your package]** Add `r/<pkg>/tests/testthat/test-parity.R` confirming bitwise-identical predictions Python ↔ R.
- **[your package]** Provide a `setup_python_<pkg>()` / `use_python_<pkg>()` helper to configure the interpreter.

---

## 5. Testing infrastructure

### 5.1 Frameworks
- pytest (Python), testthat (R).

### 5.2 Required test suites
- **Per-estimand smoke tests** against the five built-in factories.
- **Per-loss tests** if your backend supports non-squared losses (gradient/Hessian vs finite-diff).
- **Tracer algebra and dedup** — usually inherited from rieszreg's test suite; re-run if you reach into the tracer.
- **Diagnostics output and warnings** for any extras you add.
- **[from rieszreg]** **sklearn-conformance subset** + documented N/A list (re-use `rieszreg.testing.conformance`).
- **sklearn integration**: `cross_val_predict`, `Pipeline`, `get_params` round-trip.
- **Serialization round-trip** per estimand.
- **Backend equivalence on identical data** if your package has multiple internal modes (boosting backends, kernel solvers).
- **Edge cases**: ndarray vs DataFrame, single-sample, holdout-split edges (when the backend exposes `validation_fraction`).
- **Reference parity** *(required when a prior implementation exists)*: every implementation package must include at least one test that cross-checks its predictions against a *self-contained* re-derivation of any prior implementation of the same algorithm.

  **What counts as a parity test**: the reference must come from a different code path or a different mathematical formulation than your wrapper's. Either is fine: an external repo's algorithm inlined in the test file, or your own re-derivation of the algorithm via a different formulation (closed-form leaf solve, dual problem, alternative basis). What does *not* count is hand-replicating the wrapper's own packing / call sequence and asserting bit-identity — that just tests the wrapper against a copy of itself.

  Existing examples:
  - `rieszboost/python/tests/test_lee_schuler_parity.py` — inlines `ATE_ES_stochastic.fit_internal` and `ATT_ES_stochastic.fit_internal` from `kaitlynjlee/boosting_for_rr` verbatim and asserts Pearson > 0.95 / 0.85.
  - `krrr/python/tests/test_reference_parity.py` — re-derives the dml-tmle krrr.R closed form in NumPy and matches at 1e-8.
  - `forestriesz/python/tests/test_leaf_optimum.py` — pins predicted α to the closed-form per-leaf solution (`θ* = (Σφφ')⁻¹ (Σm)`) when no splits occur.
  - `forestriesz/python/tests/test_locally_linear.py` — confirms ATE on the `[1{T=0}, 1{T=1}]` sieve recovers inverse-propensity weights as n grows.

  Inline the reference algorithm in the test file rather than depending on an external repo at runtime — keeps CI reproducible. When you add a new learner package, look up every existing implementation of the same algorithm and add a parity test for each. **A failure does not mean our implementation is worse; it means the two disagree and somebody needs to investigate which one has the bug, or what algorithmic / numerical choice differs.** Document the resolution in the test file's docstring. If no prior public implementation exists (e.g., forestriesz vs EconML's `BaseGRF`, where `BaseGRF` is the only stable external interface and we use it directly), say so in the package's CLAUDE.md so future maintainers don't re-search for one.

  Also keep a runnable `_compare_with_reference.py` under `examples/<reference>/` that reports Pearson/RMSE on a larger DGP — useful for ad-hoc investigation when the test fires.
- **Property-based (Hypothesis) tests** for tracer linearity, loss round-trips, estimand factory specs, augmentation determinism.
- **[from rieszreg] Estimator-consistency suite** — required. Use `rieszreg.testing.dgps`. On a small set of analytically tractable DGPs (linear-Gaussian ATE, binary-treatment logistic α₀, etc.), with proper tuning and growing n, the learned α̂ must approach the true α₀ (or its functional). The DGPs live in rieszreg so all packages share them; each package's CI runs them against its own backend.
- **R parity test** confirming bitwise-identical Python ↔ R predictions.

---

## 6. Documentation

### 6.1 Single meta-project docs site
- **[design rule]** Do NOT host a separate docs site. Contribute pages to the meta-project Quarto site (under Backends → `<pkg>`).
- **[design rule]** Knitr engine; `{r}` chunks executed natively, `{python}` via reticulate. Bilingual `::: {.panel-tabset group="lang"}` panels on every page with executable code. Update both tabs together — the R wrapper is a first-class citizen.

### 6.2 Per-package CLAUDE.md
- **[design rule]** Each package keeps its own `CLAUDE.md` for implementation-side notes (architecture, backend internals, sharp edges, layout). NOT user-facing.
- **[from rieszreg]** A meta-project `CLAUDE.md` documents cross-cutting design rules (sklearn-first, where shared abstractions live, where to add new estimands).

### 6.3 README per package
- Status, why, install, quickstart, link to the meta-project docs.
- **[design rule] Living-doc rule**: any change to public API surface (Python OR R) must update the README and the relevant meta-site docs page in the same commit.
- **[design rule]** When a roadmap item ships, move it to "What works today" in the same commit. When scope shifts, update the roadmap with the rationale.

### 6.4 Doc-tone rules (enforced by pre-commit)
- **[design rule]** No design-decision metacommentary. Don't explain negative space ("intentionally no X", "by design", "we chose Y over Z"). The user only cares what they can call. Just describe what the function does and how to use it.
- **[design rule]** No AI-flavored hedging or editorial framing ("the workhorse", "almost never", "the natural way", "rather than reinvent"). Avoid em-dashes peppered through prose; short active-voice sentences (8–15 words on average).

### 6.5 Docstrings
- Module-level docstrings on every `.py` file explaining semantics. Class/function docstrings on all public types.

---

## 7. Examples

### 7.1 Per-estimand example rule
- **[design rule]** Every built-in estimand factory MUST have a runnable script in `examples/` exercising your backend on a realistic DGP, with EEE / one-step plug-in built around it where applicable. An estimand without an example is a hidden trap.
- **[from rieszreg]** Real-data example datasets (Lalonde NSW + CPS, NHEFS) live in or are sourced via the meta-package; reuse them rather than duplicating loaders.
- **[design rule]** `examples/README.md` indexes scripts with a status table — also subject to the living-doc rule.

### 7.2 Reference-parity examples
- **[your package]** When a prior implementation exists, add an `examples/<reference>/` directory with synthetic DGPs and a `_compare_with_reference.py` head-to-head script that reports Pearson/RMSE.

### 7.3 Quickstart example rule
- **[design rule]** When a new backend, kernel, solver, or other feature is added, add a corresponding example.

---

## 8. Tooling / dev experience

### 8.1 Package config
- `pyproject.toml` per package. setuptools build, Python ≥3.10. Depend on `rieszreg>=X.Y`.
- Optional-deps groups (`[test]`, `[lgb]`, `[jax]`, `[falkon]`, `[keops]`).
- No lockfile; `.venv/` per package, gitignored.
- R: `DESCRIPTION` with Roxygen2; `Imports: R6, reticulate, rieszreg`.

### 8.2 Pre-commit hooks (`.githooks/pre-commit`)
- **[from rieszreg]** Use the shared pre-commit hook template from the meta-project. Activate with `git config core.hooksPath .githooks` once per clone.
- The hook blocks commits that touch public-API modules without updating docs / README, and greps for the doc-tone rule violations in §6.4.
- Bypass with `--no-verify` only for genuinely doc-irrelevant changes (internal refactor, tests, comments).

### 8.3 CI / GitHub Actions
- **[from rieszreg]** Use the shared workflow templates: `docs.yml` (render meta-site, deploy to gh-pages), `test.yml` (per-package pytest + R parity).
- pytest CI is currently NOT wired in either rieszboost or krrr — your package should add it from day one.

### 8.4 Local commands
- `.venv/bin/python -m pytest python/tests -v`
- `pkgload::load_all("r/<pkg>")` + `testthat::test_dir(...)`
- `quarto preview docs/` / `quarto render docs/` (run from the meta-project root).
- macOS prerequisite: `brew install libomp` for xgboost (rieszboost-only).

### 8.5 Versioning
- Semver pre-release (`0.0.1`) initially.
- Maintain a CHANGELOG.
- Pin a compatible `rieszreg>=X.Y` range; coordinate breaking changes through the meta-package.

---

## 9. Reference materials

- **[design rule]** Do NOT host your own arXiv reference index. Contribute to `RieszReg/reference/` at the meta-project top level. Each new package adds its primary citations to the shared index with arXiv IDs and a refetch script.
- Existing core papers: Lee & Schuler 2025 (2501.04871), Chernozhukov et al. (2104.14737, 2110.03031), Singh (2102.11076), Hines & Miles (2510.16127), Kato (2601.07752), van der Laan et al. (2501.11868). Papers themselves are gitignored; the index + refetch script are committed.

---

## 10. What NOT to do (concise summary)

- Don't redefine `Estimand`, `FiniteEvalEstimand`, `Loss`, `AugmentedDataset`, `Diagnostics`, `RieszEstimator`, `LinearForm`, `Tracer`, factory-spec registries, or testing DGPs — import from `rieszreg`.
- Don't depend on another implementation package (rieszboost, krrr) — depend on `rieszreg`.
- Don't add a custom-`m()` R entry point.
- Don't reinvent `cross_val_predict`, `GridSearchCV`, or any sklearn primitive.
- Don't put `feature_keys=` (or any input-schema arg) on `fit/predict`.
- Don't put learner-specific knobs (`n_estimators`, `learning_rate`, `epochs`, `batch_size`, `early_stopping_rounds`, `kernel`, …) on `RieszEstimator` or in the Protocol method kwargs. They live on the backend dataclass; convenience subclasses forward via `_resolved_backend()`. See §A.2.
- Don't add backend-specific framing to tier-1 docs ("the booster does X", "the kernel matrix is Y"). Use neutral language ("the backend produces η").
- Don't add design-decision metacommentary or AI hedging to user docs.
- Don't host your own docs site or `reference/` directory.
- Don't introduce a bespoke `crossfit()` or `tune_riesz()`.
- Don't ship without the per-estimand examples and the consistency suite.
