# forestriesz

Random-forest backend for the [RieszReg meta-package](../README.md), implementing [Chernozhukov, Newey, Quintas-Martínez, Syrgkanis (2022)](https://proceedings.mlr.press/v162/chernozhukov22a/chernozhukov22a.pdf) for the full set of estimands the rieszreg framework supports.

This package depends on `rieszreg` for the shared abstractions (`Estimand`, `LossSpec`, `MomentBackend` Protocol, `Diagnostics`, `RieszEstimator` orchestrator, `trace`). See [`../rieszreg/DESIGN.md`](../rieszreg/DESIGN.md) for the meta-package design and the contract every implementation package follows. `forestriesz` contributes:

- `ForestRieszBackend` — `MomentBackend` Protocol implementation (the moment-style entry point added in the rieszreg refactor that ships with this package). Computes per-row moments via `rieszreg.trace` and packs them into EconML's linear-moment GRF.
- `ForestRieszRegressor` — convenience subclass of `rieszreg.RieszEstimator` with forest-specific hyperparameters (`n_estimators`, `max_depth`, `min_samples_leaf`, `honest`, `inference`, `l2`, `riesz_feature_fns`, ...) on the constructor.
- `default_riesz_features(estimand)` — sensible sieves for built-in estimands (treatment indicators).
- `predict_interval(X, alpha)` — honest-split confidence intervals for the locally constant / single-basis sieve case.
- R6 wrapper subclassing `rieszreg::RieszEstimatorR6`.

## Living-doc rule (README + meta-project docs)

`README.md` is a living document — update it in the same edit whenever a change touches the public API surface (new sieve helper, new option on `ForestRieszRegressor`, change to defaults). If a change makes any line in the README false or outdated, the change is not done until the README is fixed.

The user guide is the unified Quarto site at [`../docs/`](../docs/). The forest-specific page is [`../docs/backends/forest.qmd`](../docs/backends/forest.qmd). Any change to the forest backend that affects user-facing behavior must update that page in the same edit. On bilingual pages, update BOTH the `{python}` and `{r}` tabs.

The pre-commit hook at `.githooks/pre-commit` enforces this. Activate per clone with `git config core.hooksPath .githooks`.

## Per-estimand example rule

Estimand factories live in `rieszreg`. When a *forest-side* feature is added (new sieve helper, new diagnostic, new inference path), add a corresponding example in `examples/` that exercises it on a realistic DGP.

## R wrapper scope

The R6 wrapper exposes single-basis fits only (constant basis or single-basis sieve like TSM's `[1{T=level}]`). Multi-basis sieves (e.g. ATE's `[1{T=0}, 1{T=1}]`) are Python-callable lambdas and not yet marshalled through reticulate. R users who need ATE/ATT should call into Python via reticulate. This matches the krrr R-wrapper-scope decision.

## API design rule

The public API mirrors **ngboost / sklearn**:

- Object-oriented factory `ForestRieszRegressor(estimand=, n_estimators=, max_depth=, ...)`. `BaseEstimator`-compatible `fit / predict / score / diagnose`. Anything that can't compose with `sklearn.model_selection` (`GridSearchCV`, `cross_val_predict`, `Pipeline`) is a regression and should be fixed.
- **No `feature_keys` (or other input-schema args) on `fit()` / `predict()`.** The estimand owns its input schema.
- Cross-fitting is `sklearn.model_selection.cross_val_predict`. No bespoke `crossfit()`.
- Hyperparameter tuning is `sklearn.model_selection.GridSearchCV`. No `tune_riesz()`.

R-side mirrors this: R6 classes (`ForestRieszRegressor$new(estimand=, n_estimators=, ...)$fit(df)$predict(df)$predict_interval(df)`).

## Layout

- `python/forestriesz/` — `ForestRieszBackend`, `_RieszGRF` (BaseGRF subclass), `ForestRieszRegressor` convenience class, `ForestPredictor`, `default_riesz_features`. `pyproject.toml` declares `econml>=0.15`, `rieszreg>=0.0.1` as dependencies.
- `r/forestriesz/` — R6 wrapper via reticulate. `ForestRieszRegressor` subclasses `rieszreg::RieszEstimatorR6` (~80 lines locally). Estimand and loss factories are re-exported from `rieszreg` via NAMESPACE.
- `examples/` — runnable demonstrations of each feature (ATE, TSM with intervals).
- `.githooks/pre-commit` — copy of the meta-project canonical hook (`../.githooks/pre-commit`). Activate per clone.

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

### Dependency on rieszreg

`forestriesz` depends on `rieszreg` and reuses, without modification:

- `Estimand`, `Tracer`/`LinearForm`, `trace` — the moment-functional abstraction. Forest backends use `trace` directly to compute per-row moments without going through `build_augmented`.
- `LossSpec`, `SquaredLoss` — the Bregman-Riesz loss framework. (KLLoss / Bernoulli / BoundedSquared are NOT yet supported by the forest backend; v2.)
- `Diagnostics`, `diagnose` — base diagnostics (`ForestDiagnostics` extends with feature importance, leaf-size summary).
- `RieszEstimator` — orchestration; `ForestRieszRegressor` is a thin subclass with the forest backend defaulted.

The integration point is `rieszreg`'s `MomentBackend` Protocol (`rieszreg/backends/base.py`) — the moment-style alternative to `Backend`. `ForestRieszBackend.fit_rows(...)` consumes raw rows + the estimand and returns a `FitResult`. `ForestPredictor` registers itself for the registry-based save/load path on import via `register_predictor_loader("forestriesz", ...)`.

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

`econml` is imported eagerly because the BaseGRF subclass is constructed at fit time. There are no truly optional heavy deps in v1.

## What works today (v0.0.1)

- **`ForestRieszRegressor(BaseEstimator)`** — sklearn-compatible. Composes with `GridSearchCV`, `cross_val_predict`, `clone`, `Pipeline`. Same `fit / predict / score / diagnose` surface as `RieszBooster` and `KernelRieszRegressor`.
- **All five built-in estimands** via the rieszreg re-exports: `ATE`, `ATT`, `TSM`, `AdditiveShift`, `LocalShift`. Custom `FiniteEvalEstimand`s also work; for difference-style functionals supply your own `riesz_feature_fns`. `StochasticIntervention` is currently stubbed in rieszreg and will be reintroduced.
- **Default sieve auto-resolution**: `riesz_feature_fns="auto"` (the default) picks `[1{T=0}, 1{T=1}]` for ATE/ATT, `[1{T=level}]` for TSM, falling back to constant for custom estimands.
- **Honest-split confidence intervals** via `predict_interval(X, alpha)` for single-basis fits.
- **Loss**: `SquaredLoss` only (closed-form per-leaf solve). KLLoss / BernoulliLoss / BoundedSquaredLoss raise `NotImplementedError`; planned for v2.
- **Save / load**: directory format with JSON metadata + joblib forest. Built-in estimands round-trip automatically (sieve re-resolves on load); custom sieves require repassing callables.
- **Diagnostics**: `ForestDiagnostics` extends `rieszreg.Diagnostics` with feature importance, mean leaf size, leaf count per tree.
- **R wrapper**: R6 mirror via reticulate. Single-basis fits only.
- **34 Python tests** covering: Protocol satisfaction, all six built-in estimands, locally linear sieve correctness, single-leaf closed-form check, sklearn integration (clone, GridSearchCV, cross_val_predict), save/load round-trip, predict_interval (TSM single-basis works; ATE multi-basis raises), constant-basis degeneracy detection, NotImplementedError for non-squared losses.

## Known sharp edges

See README's `## Known sharp edges` section. Headlines: SquaredLoss only, multi-basis intervals raise (workaround: fit per-arm), honest+inference requires n_estimators % 4 == 0, constant basis raises for built-ins.

## What's next

See README's `## On the roadmap`. Headlines: non-quadratic Bregman losses via per-leaf Newton; delta-method intervals for multi-basis sieves; locally linear from R.
