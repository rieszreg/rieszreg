# rieszboost

> **Read the family design doc first.** It lives in the rieszreg meta-package
> at `rieszreg/DESIGN.md` (clone [rieszreg/rieszreg](https://github.com/rieszreg/rieszreg) as a sibling, then it's at
> [`../rieszreg/DESIGN.md`](../rieszreg/DESIGN.md)). Part B is the contract this package implements —
> anything in this CLAUDE.md is rieszboost-specific notes layered on top.

Gradient-boosting backend for the [RieszReg meta-package](../README.md), implementing Lee & Schuler ([arXiv:2501.04871](https://arxiv.org/abs/2501.04871)).

This package depends on `rieszreg` for the shared abstractions (`Estimand`, `Loss`, `AugmentedDataset`, `Diagnostics`, `Backend` Protocol, `RieszEstimator` orchestrator). `rieszboost` contributes:

- `XGBoostBackend` (default) and `SklearnBackend` — concrete `Backend` Protocol implementations.
- `RieszBooster` — convenience subclass of `rieszreg.RieszEstimator` with `XGBoostBackend` defaulted and boosting hyperparameters (`max_depth`, `reg_lambda`, `subsample`) on the constructor.
- R6 wrapper subclassing `rieszreg::RieszEstimatorR6`.

The shared `python/rieszboost/{estimand,losses,tracer,augmentation,diagnostics}.py` modules are now thin re-export shims pointing at `rieszreg`. Don't add functionality to the shims; contribute to `rieszreg` and re-export here if needed.

## Living-doc rule (README + meta-project docs)

`README.md` is a living document — update it in the same edit whenever a change touches the public API surface (new backend, new convenience-class arg). If a change makes any line in the README false or outdated, the change is not done until the README is fixed.

The user guide moved to the unified Quarto site at [`../docs/`](../docs/). The boosting-specific page is [`../docs/backends/boosting.qmd`](../docs/backends/boosting.qmd). Any change to the boosting backend that affects user-facing behavior must update that page in the same edit. On bilingual pages, update BOTH the `{python}` and `{r}` tabs.

The pre-commit hook at `.githooks/pre-commit` enforces this — a public-API change with no `README.md` or `docs/*.qmd` change in the same commit is rejected. Activate the hook once per clone with `git config core.hooksPath .githooks`. Bypass only for genuinely doc-irrelevant changes (internal refactor, tests, comments) with `--no-verify`.

The original `docs/` directory in this package is deprecated; see `docs/DEPRECATED.md`.

### Doc tone rules

User-facing docs (`docs/*.qmd`, `README.md`) describe what's currently in the package, in plain instructive prose matching the [ngboost user guide](https://stanfordmlgroup.github.io/ngboost/intro.html). Two failure modes the pre-commit hook also checks for:

1. **No design-decision metacommentary.** Don't explain the API's negative space — what we removed, intentionally didn't build, or chose between. Examples to avoid: "there is intentionally no bespoke `crossfit()` function", "no separate `feature_keys=` argument", "the design rule is to lean on…", "we chose X over Y", "intentionally out of scope", "by design". The user only cares what they CAN call. Just describe what the function does and how to use it; if a sentence describes what isn't there, delete it. The API design rule below is for the maintainer; it doesn't belong in user docs.

2. **No AI-flavored hedge or editorial framing.** Avoid phrases like "the workhorse", "the right choice for almost every", "almost never needs tuning", "the natural way/API", "rather than reinvent". Avoid em-dashes peppered through prose; use periods or rewrite. Sentences should be short (8-15 words on average), active voice, no excessive parentheticals.

The hook greps the staged diff (added lines only) for the worst offenders and blocks the commit; bypass with `--no-verify` if it's a false positive.

## Per-estimand example rule

**Every built-in estimand factory must have a worked example** in `examples/`. When you add a new factory (e.g. a new shift variant, a new IPSI form, a longitudinal helper), the change is not done until there's a runnable script demonstrating it on a realistic DGP, with the EEE/one-step plug-in built around it where applicable.

This is non-negotiable: an estimand without an example is a hidden trap — users find it in `dir(rieszboost)`, guess at the API, and silently misuse it. Worked examples are the documentation.

## R wrapper scope

The R wrapper exposes *only the built-in estimands* (`ATE`, `ATT`, `TSM`, `AdditiveShift`, `LocalShift`, `StochasticIntervention`). **Do not** add an R-side custom-m() entry point — the `LinearForm` tracer is Python-only and porting it is more trouble than the use case warrants. R users who need a brand-new functional write the m() in Python (as an `Estimand`) and call into it from R via reticulate; that path already works today.

If a new estimand factory is added on the Python side, it should also be exposed in the R `NAMESPACE` and have an integration check in `r/rieszboost/tests/testthat/`.

## API design rule

The public API should feel like **ngboost / sklearn**:

- Object-oriented factories that bake in configuration (estimand, loss, backend, hyperparameters) at construction; `BaseEstimator`-compatible `fit / predict / score` on every fittable thing; swappable orthogonal components (backend, loss, estimand). Anything that can't compose with `sklearn.model_selection` (`GridSearchCV`, `cross_val_predict`, `Pipeline`) is a regression and should be fixed.
- **No `feature_keys` (or other input-schema args) on `fit()` / `predict()`.** The estimand owns its input schema — `feature_keys`, `extra_keys`, anything else. If a new estimand needs different inputs, that's a property of the estimand object, not a separate argument the user repeats every call.
- Cross-fitting is `sklearn.model_selection.cross_val_predict`. Don't reintroduce a bespoke `crossfit()` function.
- Hyperparameter tuning is `sklearn.model_selection.GridSearchCV` (or `HalvingGridSearchCV`, etc.). Don't introduce a `tune_riesz()`.

**Apply this rule to every new feature, not just the public API surface.** Before writing any new code — especially anything procedural with a loop, a split, a grid, or a fold — ask: *is there an sklearn way to do this?* If yes, use it. The answer is yes more often than feels intuitive: cross-fitting (`cross_val_predict`), CV scoring (`cross_validate`), splits (`KFold`, `train_test_split`, `StratifiedKFold`), tuning (`GridSearchCV`, `HalvingGridSearchCV`, `RandomizedSearchCV`), composition (`Pipeline`, `ColumnTransformer`, `FunctionTransformer`), scoring (`make_scorer`), parallelism (`n_jobs=`). This applies to library code, examples, docs, and tests. A hand-rolled `for tr_idx, te_idx in KFold(...).split(X):` loop is a code smell — when you find one (yours or pre-existing), replace it with `cross_val_predict` unless you can articulate why sklearn's version is wrong for the task. Bespoke is reserved for things sklearn genuinely doesn't cover (the `LinearForm` tracer, the custom xgboost objective, the Bregman `Loss`).

R-side mirrors this: R6 classes (`RieszBooster$new(estimand=, loss=, ...)$fit(df)$predict(df)`) rather than functional `fit_riesz()` shims.

## Layout

- `python/rieszboost/` — backend implementations (`XGBoostBackend`, `SklearnBackend`) and the `RieszBooster` convenience class. Shared modules (`estimand.py`, `losses.py`, `tracer.py`, `augmentation.py`, `diagnostics.py`, `backends/base.py`) are now thin re-export shims pointing at `rieszreg`. `pyproject.toml` declares `rieszreg>=0.0.1` as a dependency.
- `r/rieszboost/` — R6 wrapper via reticulate. `RieszBooster` subclasses `rieszreg::RieszEstimatorR6` (~50 lines locally). Estimand and loss factories are re-exported from `rieszreg` via NAMESPACE. Run R tests by dev-loading both packages: `pkgload::load_all("../rieszreg/r/rieszreg"); pkgload::load_all("r/rieszboost"); testthat::test_dir(...)`.
- `reference/` — moved to the meta-project top level at `../reference/` (the local copy is deprecated; will be removed).
- `docs/` — deprecated; see `docs/DEPRECATED.md`. The unified Quarto site lives at `../docs/`.
- `.githooks/pre-commit` — copy of the meta-project canonical hook (`../.githooks/pre-commit`). Activate per clone.
- `.venv/` — local Python venv (gitignored).

## Run tests

```sh
.venv/bin/python -m pytest python/tests -v
```

`xgboost` requires `libomp` on macOS — install with `brew install libomp` once.

## Architecture notes

- **m() is opaque, JAX-style.** Users write `m(alpha)` as an operator that returns a function of `z`, built with `+`, `-`, scalar `*`, and calls to `alpha(**kwargs)`. The `Tracer` (`rieszboost/tracer.py`) records each call as a `LinearTerm` and composes them into a `LinearForm`. Anything that leaves the linear-form algebra raises (signal to dispatch to slow path, when the slow path lands).
- **Fast path = data augmentation + xgboost custom objective.** `Estimand.augment` emits per-row (a, b) coefficients so the loss is `Σ a_j α(z̃_j)² + b_j α(z̃_j)`. xgboost's custom objective consumes gradient `2aF + b` and Hessian `2a` directly. Augmented rows with `a=0` get a small `eps` floor in the Hessian to keep leaf-weight optimization stable.
- **Slow general path** (not yet implemented): Friedman-style gradient boosting against arbitrary base learners (sklearn, JAX, etc.) for non-finite-point m (integrals, derivatives).
- **xgboost is lazy-imported** so the rest of the rieszboost / rieszreg API is usable without xgboost or libomp.

## What's done (v0.0.1)

- **`RieszBooster`** subclasses `rieszreg.RieszEstimator`, defaulting backend to `XGBoostBackend()` and adding `max_depth`, `reg_lambda`, `subsample` constructor args. Composes with `GridSearchCV`, `cross_val_predict`, `clone`, `Pipeline`.
- **Backends**: `XGBoostBackend(hessian_floor=2.0, gradient_only=False)` and `SklearnBackend(base_learner_factory)`. Both register their `Predictor` loader for the registry-based save/load path on import.
- All shared abstractions (`Estimand`, `Loss`, `LinearForm`, `AugmentedDataset`, `Diagnostics`) live in `rieszreg`; this package re-exports them.
- 109 Python tests + 11 R parity tests passing. Includes acceptance gates for `clone`, `GridSearchCV`, `cross_val_predict`.

## Longitudinal / LMTP

Full LMTP support requires multi-stage orchestration (one Riesz fit per time-stage in the nested g-formula). That belongs in a downstream wrapper, not in this library. The single-stage `fit(rows, m, ...)` API IS the upstream that an LMTP wrapper calls; do not add a half-built `Longitudinal` factory that only handles the no-time-varying-confounding case.

## Known sharp edges

- Boosting can extrapolate aggressively in low-overlap regions of α̂. Use shallow trees (`max_depth=3`), early stopping, and look at `diagnose(...)` warnings — it flags when max |α̂| dwarfs the 99th percentile (a sign of a single outlier extrapolating).
- `_make_objective` floors the Hessian at `hessian_floor=2.0` (matching the natural Hessian of original a=1 rows). Earlier we used `eps=1e-6`, which made counterfactual leaves degenerate in xgboost's leaf-weight Newton step (`-G/(H+λ)` with H≈0) and required `reg_lambda=100` to stabilize. The new floor mimics the row-uniform weighting that first-order gradient boosting (Friedman 2001) uses by construction; standard xgboost-style hyperparameters now work without sledgehammer regularization.
- `gradient_only=True` on `fit(...)` short-circuits the Loss hessian and sends `hess=ones_like(grad)` to xgboost — exactly first-order gradient boosting (Lee-Schuler Algorithm 2). Empirically on the binary-DGP example it's *worse* than the floored second-order path at matched hyperparameters (ATE α-RMSE ~2.0 vs ~1.2; ATT ~1.0 vs ~0.8). Even though the second-order Hessian is artificial (it's just our floor), the leaf-weight balancing seems to help relative to first-order.
- Cross-check vs Kaitlyn Lee's reference implementation (`kaitlynjlee/boosting_for_rr`): our `gradient_only=True, learning_rate=lr_ref/2, reg_lambda=0` reproduces `ATE_ES_stochastic` / `ATT_ES_stochastic` to Pearson 0.998 / 0.986 on identical data. The factor of 2 is real: their per-row residual is `f - (2a-1)` (drops a factor of 2 from the natural Riesz loss gradient `2(2aF + b)`), ours is `2aF + b`. Their `fit_internal` (no early stopping path) has a shape bug — `A.reshape(-1,1)` then `A == 1` for indexing 1D arrays — only `fit_internal_early_stopping` works.

## What's next

See README's `## On the roadmap` section. Headlines: serialization, more example datasets (Lalonde / NHEFS / two-stage longitudinal), R-side custom m(), lightgbm backend, more Bregman losses, packaging.
