# forestriesz

Random-forest backends for the [RieszReg meta-package](../README.md). Two flavors:

- An augmentation-style ensemble of `riesztree.RieszTreeBackend` instances over block-bootstrapped augmented rows, with sklearn `RandomForestRegressor`-style hyperparameters and loss-aware splits. Works on every estimand without per-estimand configuration.
- A moment-style implementation of [Chernozhukov, Newey, Quintas-Martínez, Syrgkanis (2022)](https://proceedings.mlr.press/v162/chernozhukov22a/chernozhukov22a.pdf) on top of EconML's GRF, with honest-split confidence intervals on `ATE` / `ATT` / `TSM`.

This package depends on `rieszreg` for the shared abstractions (`Estimand`, `Loss`, `Backend` / `MomentBackend` Protocols, `Diagnostics`, `RieszEstimator` orchestrator, `trace`, `AugmentedDataset`) and on `riesztree` for the per-tree augmented learner. See [`../rieszreg/DESIGN.md`](../rieszreg/DESIGN.md) for the meta-package design and the contract every implementation package follows. `forestriesz` contributes:

- `AugForestRieszBackend` — `Backend.fit_augmented` Protocol implementation. Ensemble of `riesztree.RieszTreeBackend` instances over block-bootstrapped augmented rows; each tree fits a loss-aware splitter directly on the augmented dataset. Works on every estimand without per-estimand configuration. Supports all four built-in Bregman losses. When `splitter='hist'` and the config is "simple" (no categoricals, default `max_features`, no `ccp_alpha`, no leaf cap), the bin mapper is fitted once on the full augmented data and shared across joblib workers — `~2×` speedup at shallow depths. The exact-splitter path is unchanged from the per-tree default.
- `AugForestRieszRegressor` — convenience subclass of `rieszreg.RieszEstimator` with sklearn `RandomForestRegressor`-style hyperparameters (`n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_features`, `bootstrap`, `max_samples`, `n_jobs`, `splitter`, `max_bins`, `categorical_features`, ...).
- `ForestRieszBackend` — `MomentBackend.fit_rows` Protocol implementation. Computes per-row moments via `rieszreg.trace` and packs them into EconML's linear-moment GRF.
- `ForestRieszRegressor` — convenience subclass of `rieszreg.RieszEstimator` with forest-specific hyperparameters (`n_estimators`, `max_depth`, `min_samples_leaf`, `honest`, `inference`, `l2`, `riesz_feature_fns`, ...) on the constructor.
- `default_riesz_features(estimand)` — defaults for the moment-style backend's `riesz_feature_fns` for built-in estimands (treatment indicators).
- `predict_interval(X, alpha)` — honest-split confidence intervals on the moment-style backend for the locally constant / single-basis case.
- R6 wrapper subclassing `rieszreg::RieszEstimatorR6` (moment-style only).

## Living-doc rule (README + meta-project docs)

`README.md` is a living document — update it in the same edit whenever a change touches the public API surface (new sieve helper, new option on `ForestRieszRegressor`, change to defaults). If a change makes any line in the README false or outdated, the change is not done until the README is fixed.

The user guide is the unified Quarto site at [`../docs/`](../docs/). The forest-specific page is [`../docs/backends/forest.qmd`](../docs/backends/forest.qmd). Any change to the forest backend that affects user-facing behavior must update that page in the same edit. On bilingual pages, update BOTH the `{python}` and `{r}` tabs.

The pre-commit hook at the monorepo root (`../../.githooks/pre-commit`) enforces this; activate it once per clone with `bash ../../scripts/setup-hooks.sh`. The same lint runs in CI via the `lint-docs` job.

## Per-estimand example rule

Estimand factories live in `rieszreg`. When a *forest-side* feature is added (new sieve helper, new diagnostic, new inference path), add a corresponding example in `examples/` that exercises it on a realistic DGP.

## R wrapper scope

The R6 wrapper exposes single-basis fits only (constant basis or single-basis sieve like TSM's `[1{T=level}]`). Multi-basis sieves (e.g. ATE's `[1{T=0}, 1{T=1}]`) are Python-callable lambdas and not yet marshalled through reticulate. R users who need ATE/ATT should call into Python via reticulate.

## API design rule

The public API mirrors **ngboost / sklearn**:

- Object-oriented factory `ForestRieszRegressor(estimand=, n_estimators=, max_depth=, ...)`. `BaseEstimator`-compatible `fit / predict / score / diagnose`. Anything that can't compose with `sklearn.model_selection` (`GridSearchCV`, `cross_val_predict`, `Pipeline`) is a regression and should be fixed.
- **No `feature_keys` (or other input-schema args) on `fit()` / `predict()`.** The estimand owns its input schema.
- Cross-fitting is `sklearn.model_selection.cross_val_predict`. No bespoke `crossfit()`.
- Hyperparameter tuning is `sklearn.model_selection.GridSearchCV`. No `tune_riesz()`.

R-side mirrors this: R6 classes (`ForestRieszRegressor$new(estimand=, n_estimators=, ...)$fit(df)$predict(df)$predict_interval(df)`).

## Layout

- `python/forestriesz/` — `AugForestRieszBackend` / `AugForestRieszRegressor` / `AugForestPredictor` (augmentation-style; built on `riesztree.RieszTreeBackend`), `ForestRieszBackend` / `ForestRieszRegressor` / `ForestPredictor` / `_RieszGRF` (moment-style; built on EconML's BaseGRF), `default_riesz_features`. `pyproject.toml` declares `econml>=0.15`, `rieszreg>=0.0.1`, `riesztree>=0.0.1` as dependencies.
- `r/forestriesz/` — R6 wrapper via reticulate. `ForestRieszRegressor` subclasses `rieszreg::RieszEstimatorR6` (~80 lines locally). Estimand and loss factories are re-exported from `rieszreg` via NAMESPACE.
- `examples/` — runnable demonstrations of each feature (ATE, TSM with intervals; AdditiveShift on the augmented backend).

## Run tests

```sh
.venv/bin/python -m pytest python/tests -v
```

R parity:

```sh
Rscript -e '
  Sys.setenv(RETICULATE_PYTHON = file.path(getwd(), ".venv/bin/python"))
  pkgload::load_all("../rieszreg/r/rieszreg")
  pkgload::load_all("r/forestriesz")
  testthat::test_dir("r/forestriesz/tests/testthat")
'
```

## Architecture notes

### Dependency on rieszreg and riesztree

`forestriesz` depends on `rieszreg` and reuses, without modification:

- `Estimand`, `Tracer`/`LinearForm`, `trace`, `AugmentedDataset` — moment-functional / augmentation abstractions. The moment-style backend uses `trace` directly to compute per-row moments. The augmentation-style backend consumes the precomputed `AugmentedDataset` from the orchestrator.
- `Loss`, `SquaredLoss`, `KLLoss`, `BernoulliLoss`, `BoundedSquaredLoss` — the Bregman-Riesz loss framework. The augmentation-style backend supports all four; the moment-style backend supports `SquaredLoss` only (v2 will extend it).
- `Diagnostics`, `diagnose` — base diagnostics (`ForestDiagnostics` extends with feature importance, leaf-size summary).
- `RieszEstimator` — orchestration; `ForestRieszRegressor` and `AugForestRieszRegressor` are thin subclasses with their respective backends defaulted.

The augmentation-style backend additionally depends on `riesztree`'s `RieszTreeBackend` (consumed via the `Backend.fit_augmented` Protocol) — each forest tree is one riesztree fit on a block-bootstrapped subsample.

The integration points are `rieszreg`'s `Backend` and `MomentBackend` Protocols (`rieszreg/backends/base.py`). `AugForestRieszBackend.fit_augmented(...)` consumes the precomputed `AugmentedDataset`; `ForestRieszBackend.fit_rows(...)` consumes raw rows + the estimand. Both return a `FitResult`. The respective predictors register themselves on import via `register_predictor_loader("aug-forestriesz", ...)` and `register_predictor_loader("forestriesz", ...)`.

### Moment-path packing for EconML's BaseGRF

EconML's `LinearMomentGRFCriterionMSE` (the criterion the paper authors used) solves `E[J θ - A | X = x] = 0` per leaf. We pack the per-row J and A into the EconML "treatment" array T:

```
X       : (n, n_split)        split features (estimand.feature_keys minus sieve-handled cols)
T       : (n, p² + p)          first p² cols = J flattened, next p cols = A
y       : (n, 1)               dummy zeros (LinearMomentGRFCriterion wants scalar y)
```

with

```
A[i, j] = Σ_(coef, point) ∈ trace(W_i)  coef · φ_j(point)         (per-row moment vector)
J[i]    = φ(W_i) φ(W_i)'                                           (per-row Jacobian)
```

`_get_alpha_and_pointJ` unpacks T and returns `(A, J)` to the criterion. `_get_n_outputs_decomposition` declares all p outputs are relevant. Per-leaf solve is closed-form `θ_ℓ = (Σ J_i)^{-1} Σ A_i`; predictions are `α(z) = θ(z_split) · φ(z) + base_score`.

For TSM with `default_riesz_features([1{T=level}])`, A_i = 1 (constant), J_i = T_i (varies). Per-leaf θ = 1 / P̂(T=level | X-region) — exactly the IPW representer. The forest splits on covariates only (the sieve resolves treatment).

### Constant-basis degeneracy

For all built-in estimands, `m(W; 1) = Σ coef` in the trace doesn't depend on W (e.g., ATE → 0, TSM → 1, AdditiveShift → 0). Under a constant basis both A and J are row-constant, so splits learn nothing. The backend detects this and raises with a hint to use `riesz_feature_fns="auto"` or a custom sieve. The default is `"auto"`, which resolves to `default_riesz_features(estimand)` automatically.

### What's lazy-imported

`econml` and `riesztree` are imported eagerly. There are no truly optional heavy deps in v1.

## What works today (v0.0.1)

### Augmentation-style (`AugForestRieszRegressor`)

- sklearn `RandomForestRegressor`-style hyperparameters. Composes with `GridSearchCV`, `cross_val_predict`, `clone`, `Pipeline`.
- Works on every built-in estimand (`ATE`, `ATT`, `TSM`, `AdditiveShift`, `LocalShift`) and any user-defined `FiniteEvalEstimand` without per-estimand configuration.
- Supports all four built-in Bregman losses (`SquaredLoss`, `KLLoss`, `BernoulliLoss`, `BoundedSquaredLoss`) via riesztree's loss-aware splitter dispatch.
- Save / load: per-tree subdir format. Round-trips without any extra arguments.
- No CIs in v1.

### Moment-style (`ForestRieszRegressor`)

- sklearn-compatible. Composes with `GridSearchCV`, `cross_val_predict`, `clone`, `Pipeline`. Same `fit / predict / score / diagnose` surface as `RieszBooster` and `KernelRieszRegressor`.
- All five built-in estimands. Custom `FiniteEvalEstimand`s also work; for difference-style functionals supply your own `riesz_feature_fns`. `StochasticIntervention` is currently stubbed in rieszreg and will be reintroduced.
- Default `riesz_feature_fns` resolution: `riesz_feature_fns="auto"` (the default) picks `[1{A=0}, 1{A=1}]` for ATE/ATT, `[1{A=level}]` for TSM, falling back to constant for custom estimands.
- Honest-split confidence intervals via `predict_interval(X, alpha)` for single-basis fits.
- Loss: `SquaredLoss` only (closed-form per-leaf solve). KLLoss / BernoulliLoss / BoundedSquaredLoss raise `NotImplementedError`; planned for v2.
- Save / load: directory format with JSON metadata + joblib forest. Built-in estimands round-trip automatically; custom `riesz_feature_fns` require repassing callables.
- R wrapper: R6 mirror via reticulate. Single-basis fits only.

### Shared

- `ForestDiagnostics` extends `rieszreg.Diagnostics` with feature importance, mean leaf size, leaf count per tree (moment-style).

## Known sharp edges

See README's `## Known sharp edges` section. Headlines: SquaredLoss only, multi-basis intervals raise (workaround: fit per-arm), honest+inference requires n_estimators % 4 == 0, constant basis raises for built-ins.

## What's next

See README's `## On the roadmap`. Headlines: non-quadratic Bregman losses via per-leaf Newton; delta-method intervals for multi-basis sieves; locally linear from R.
