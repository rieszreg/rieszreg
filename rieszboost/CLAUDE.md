# rieszboost

General-purpose gradient-boosting library for Riesz representers, implementing Lee & Schuler ([arXiv:2501.04871](https://arxiv.org/abs/2501.04871)).

## Living-doc rule

`README.md` is a living document — update it in the same edit whenever a change touches the public API surface (new estimand factory, new function exported from `rieszboost.__init__`, new engine), the supported feature list, the install/run instructions, or the quickstart example. If a change makes any line in the README false or outdated, the change is not done until the README is fixed. The README is the user-facing contract; CLAUDE.md is the implementation-side notes.

The README's `## Status` and `## Roadmap` sections must stay current too — when a roadmap item ships, move it to "What works today" (or remove if mentioned elsewhere) **in the same commit**. When scope shifts (an item is dropped, deferred, or replaced by something new), update the roadmap with the rationale. Don't let either section drift behind reality. Same applies to any analogous status table in `examples/README.md`.

## API design rule

The public API should feel like **ngboost / sklearn**:

- Object-oriented factories that bake in configuration (estimand, loss, backend, hyperparameters) at construction; `BaseEstimator`-compatible `fit / predict / score` on every fittable thing; swappable orthogonal components (backend, loss, estimand). Anything that can't compose with `sklearn.model_selection` (`GridSearchCV`, `cross_val_predict`, `Pipeline`) is a regression and should be fixed.
- **No `feature_keys` (or other input-schema args) on `fit()` / `predict()`.** The estimand owns its input schema — `feature_keys`, `extra_keys`, anything else. If a new estimand needs different inputs, that's a property of the estimand object, not a separate argument the user repeats every call.
- Cross-fitting is `sklearn.model_selection.cross_val_predict`. Don't reintroduce a bespoke `crossfit()` function.
- Hyperparameter tuning is `sklearn.model_selection.GridSearchCV` (or `HalvingGridSearchCV`, etc.). Don't introduce a `tune_riesz()`.

R-side mirrors this: R6 classes (`RieszBooster$new(estimand=, loss=, ...)$fit(df)$predict(df)`) rather than functional `fit_riesz()` shims.

## Layout

- `python/` — primary implementation. Library is `rieszboost/`; tests in `tests/`. Build/dependency config in `pyproject.toml`.
- `reference/` — arXiv source for the relevant papers (gitignored). See `reference/README.md` for the index and refetch script.
- `.venv/` — local Python venv (gitignored once we add a top-level `.gitignore`).
- `r/rieszboost/` — R6 wrapper via reticulate. Construct with `RieszBooster$new(estimand=, loss=, ...)`; `$fit(df)`, `$predict(df)`, `$score(df)`, `$diagnose(df)`. All estimand factories (`ATE()`, `ATT()`, `TSM()`, `AdditiveShift()`, `LocalShift()`, `StochasticIntervention()`) and loss specs (`SquaredLoss()`, `KLLoss()`) and backends (`XGBoostBackend()`, `SklearnBackend()`) are exposed. Custom user-supplied m() must currently be written in Python — the LinearForm tracer is Python-only. Run R tests via `pkgload::load_all` + `testthat::test_dir`; the parity test confirms R/Python predictions are bitwise-identical.

## Run tests

```sh
.venv/bin/python -m pytest python/tests -v
```

`xgboost` requires `libomp` on macOS — install with `brew install libomp` once.

## Architecture notes

- **m() is opaque, JAX-style.** Users write `m(z, alpha) -> LinearForm` using `+`, `-`, scalar `*`, and calls to `alpha(**kwargs)`. The `Tracer` (`rieszboost/tracer.py`) records each call as a `LinearTerm` and composes them into a `LinearForm`. Anything that leaves the linear-form algebra raises (signal to dispatch to slow path, when the slow path lands).
- **Fast path = data augmentation + xgboost custom objective.** `engine.build_augmented` traces m on each row and assembles per-row (a, b) coefficients so the loss is `Σ a_j α(z̃_j)² + b_j α(z̃_j)`. xgboost's custom objective consumes gradient `2aF + b` and Hessian `2a` directly. Augmented rows with `a=0` get a small `eps` floor in the Hessian to keep leaf-weight optimization stable.
- **Slow general path** (not yet implemented): Friedman-style gradient boosting against arbitrary base learners (sklearn, JAX, etc.) for non-finite-point m (integrals, derivatives).
- **xgboost is lazy-imported** so the tracer and estimand factories are usable without xgboost or libomp.

## What's done (v0.0.1)

- **sklearn-compatible `RieszBooster(BaseEstimator)`** in `python/rieszboost/estimator.py`. Composes with `GridSearchCV`, `cross_val_predict`, `clone`, `Pipeline`. Configuration objects (estimand, loss, backend) baked in at construction; `.fit / .predict / .score / .diagnose` at use time. Accepts ndarray (columns matched to `estimand.feature_keys`) or DataFrame (columns matched by name; `extra_keys` payload pulled through).
- **`Estimand` class** owns its input schema (`feature_keys`, `extra_keys`, m). No more `feature_keys=` arg on `fit()`. Factories: `ATE / ATT / LocalShift / TSM / AdditiveShift / StochasticIntervention`. ATT and LocalShift fit *partial-parameter* representers (full ATT/LASE require delta-method downstream).
- **Backends** (swappable): `XGBoostBackend(hessian_floor=2.0, gradient_only=False)` and `SklearnBackend(base_learner_factory)`. New backends slot in without touching `RieszBooster`.
- **Bregman-Riesz losses**: `LossSpec` protocol with `link_to_alpha` / `alpha_to_eta`. `SquaredLoss` (identity link, default), `KLLoss` (exp link, for density-ratio targets; requires non-negative m-coefficients).
- `LinearForm` tracer + linearity enforcement; `build_augmented` extracted to `augmentation.py`.
- `init={float, 'm1', None}` in α-space; loss spec converts to η for the booster.
- Early stopping via `validation_fraction` (auto internal split) or explicit `eval_set=`; `best_iteration_` + predict-with-best-iteration baked in.
- Cross-fitting via `sklearn.model_selection.cross_val_predict` (no bespoke `crossfit()` function).
- Diagnostics: `booster.diagnose(X)` or top-level `rieszboost.diagnose(...)`.
- R wrapper is R6-style: `RieszBooster$new(estimand=, loss=, ...)` with `$fit/$predict/$score/$diagnose`. Bitwise-identical predictions vs Python.
- 51 Python tests + R parity test passing. Includes acceptance gates for `clone`, `GridSearchCV`, `cross_val_predict`.

## Longitudinal / LMTP

Full LMTP support requires multi-stage orchestration (one Riesz fit per time-stage in the nested g-formula). That belongs in a downstream wrapper, not in this library. The single-stage `fit(rows, m, ...)` API IS the upstream that an LMTP wrapper calls; do not add a half-built `Longitudinal` factory that only handles the no-time-varying-confounding case.

## Known sharp edges

- Boosting can extrapolate aggressively in low-overlap regions of α̂. Use shallow trees (`max_depth=3`), early stopping, and look at `diagnose(...)` warnings — it flags when max |α̂| dwarfs the 99th percentile (a sign of a single outlier extrapolating).
- `_make_objective` floors the Hessian at `hessian_floor=2.0` (matching the natural Hessian of original a=1 rows). Earlier we used `eps=1e-6`, which made counterfactual leaves degenerate in xgboost's leaf-weight Newton step (`-G/(H+λ)` with H≈0) and required `reg_lambda=100` to stabilize. The new floor mimics the row-uniform weighting that first-order gradient boosting (Friedman 2001) uses by construction; standard xgboost-style hyperparameters now work without sledgehammer regularization.
- `gradient_only=True` on `fit(...)` short-circuits the LossSpec hessian and sends `hess=ones_like(grad)` to xgboost — exactly first-order gradient boosting (Lee-Schuler Algorithm 2). Empirically on the binary-DGP example it's *worse* than the floored second-order path at matched hyperparameters (ATE α-RMSE ~2.0 vs ~1.2; ATT ~1.0 vs ~0.8). Even though the second-order Hessian is artificial (it's just our floor), the leaf-weight balancing seems to help relative to first-order.
- Cross-check vs Kaitlyn Lee's reference implementation (`kaitlynjlee/boosting_for_rr`): our `gradient_only=True, learning_rate=lr_ref/2, reg_lambda=0` reproduces `ATE_ES_stochastic` / `ATT_ES_stochastic` to Pearson 0.998 / 0.986 on identical data. The factor of 2 is real: their per-row residual is `f - (2a-1)` (drops a factor of 2 from the natural Riesz loss gradient `2(2aF + b)`), ours is `2aF + b`. Their `fit_internal` (no early stopping path) has a shape bug — `A.reshape(-1,1)` then `A == 1` for indexing 1D arrays — only `fit_internal_early_stopping` works.

## What's next

See README's `## On the roadmap` section. Headlines: serialization, more example datasets (Lalonde / NHEFS / two-stage longitudinal), R-side custom m(), lightgbm backend, more Bregman losses, packaging.
