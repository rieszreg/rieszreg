# rieszboost

General-purpose gradient-boosting library for Riesz representers, implementing Lee & Schuler ([arXiv:2501.04871](https://arxiv.org/abs/2501.04871)).

## Living-doc rule

`README.md` is a living document — update it in the same edit whenever a change touches the public API surface (new estimand factory, new function exported from `rieszboost.__init__`, new engine), the supported feature list, the install/run instructions, or the quickstart example. If a change makes any line in the README false or outdated, the change is not done until the README is fixed. The README is the user-facing contract; CLAUDE.md is the implementation-side notes.

## Layout

- `python/` — primary implementation. Library is `rieszboost/`; tests in `tests/`. Build/dependency config in `pyproject.toml`.
- `reference/` — arXiv source for the relevant papers (gitignored). See `reference/README.md` for the index and refetch script.
- `.venv/` — local Python venv (gitignored once we add a top-level `.gitignore`).
- `r/rieszboost/` — R wrapper via reticulate. Same API as Python: `fit_riesz`, `predict.RieszBooster`, `crossfit`, `diagnose_alpha`, and the `ATE`/`ATT`/`TSM`/`AdditiveShift` factories. Custom user-supplied m() must currently be written in Python — the LinearForm tracer is Python-only. Run R tests via `pkgload::load_all` + `testthat::test_dir`; the parity test confirms R/Python predictions are bitwise-identical.

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

- `LinearForm` tracer + linearity enforcement via algebra.
- `build_augmented` + `fit` + `RieszBooster.predict` on the xgboost fast path.
- `general_fit` slow path: first-order gradient boosting (Friedman 2001) on the augmented dataset with arbitrary sklearn-compatible base learners (line-searched step size each round).
- ATE / ATT / TSM / AdditiveShift / StochasticIntervention estimand factories. StochasticIntervention is Monte Carlo: user pre-samples K treatment values per row from the intervention density and stores them under `samples_key`; m averages alpha over those samples (1/K each). Tracer sees finite-point linear combination → fast path works without engine changes.
- `init={0, float, 'm1'}` initialization; m1 traces m on a constant alpha=1.
- Early stopping on held-out Riesz loss in both engines; `best_iteration` + predict-with-best-iteration baked in.
- K-fold cross-fitting (`crossfit.crossfit`) with optional inner-split early stopping.
- Diagnostics (`diagnostics.diagnose`) — RMS, |α| quantiles, extreme-row count, near-positivity and outlier-extrapolation warnings, held-out Riesz loss.
- 36 Python tests + 13 R tests passing.
- R wrapper via reticulate — same surface as Python.
- Bregman-Riesz framework: `LossSpec` protocol with `link_to_alpha` / `alpha_to_eta` so xgboost boosts in η-space and predictions live in α-space. Concrete: `SquaredLoss` (identity link), `KLLoss` (exp link, requires non-negative m-coefficients). Note: KL on the Lee-Schuler-style augmented dataset is structurally less stable than squared because pure-counterfactual rows (a=0, b<0) have no minimum in η, so leaf weights at low-overlap points can extrapolate. Document this when promoting KL.

## Longitudinal / LMTP

Full LMTP support requires multi-stage orchestration (one Riesz fit per time-stage in the nested g-formula). That belongs in a downstream wrapper, not in this library. The single-stage `fit(rows, m, ...)` API IS the upstream that an LMTP wrapper calls; do not add a half-built `Longitudinal` factory that only handles the no-time-varying-confounding case.

## Known sharp edges

- Boosting can extrapolate aggressively in low-overlap regions of α̂. Use shallow trees (`max_depth=3`), early stopping, and look at `diagnose(...)` warnings — it flags when max |α̂| dwarfs the 99th percentile (a sign of a single outlier extrapolating).
- `_make_objective` floors the Hessian at `hessian_floor=2.0` (matching the natural Hessian of original a=1 rows). Earlier we used `eps=1e-6`, which made counterfactual leaves degenerate in xgboost's leaf-weight Newton step (`-G/(H+λ)` with H≈0) and required `reg_lambda=100` to stabilize. The new floor mimics the row-uniform weighting that first-order gradient boosting (Friedman 2001) uses by construction; standard xgboost-style hyperparameters now work without sledgehammer regularization.
- `gradient_only=True` on `fit(...)` short-circuits the LossSpec hessian and sends `hess=ones_like(grad)` to xgboost — exactly first-order gradient boosting (Lee-Schuler Algorithm 2). Empirically on the binary-DGP example it's *worse* than the floored second-order path at matched hyperparameters (ATE α-RMSE ~2.0 vs ~1.2; ATT ~1.0 vs ~0.8). Even though the second-order Hessian is artificial (it's just our floor), the leaf-weight balancing seems to help relative to first-order.
- Cross-check vs Kaitlyn Lee's reference implementation (`kaitlynjlee/boosting_for_rr`): our `gradient_only=True, learning_rate=lr_ref/2, reg_lambda=0` reproduces `ATE_ES_stochastic` / `ATT_ES_stochastic` to Pearson 0.998 / 0.986 on identical data. The factor of 2 is real: their per-row residual is `f - (2a-1)` (drops a factor of 2 from the natural Riesz loss gradient `2(2aF + b)`), ours is `2aF + b`. Their `fit_internal` (no early stopping path) has a shape bug — `A.reshape(-1,1)` then `A == 1` for indexing 1D arrays — only `fit_internal_early_stopping` works.

## What's next (per `~/.claude/plans/i-d-like-to-write-crystalline-raven.md`)

- lightgbm engine adapter.
- Slow general path with sklearn/JAX base learners.
- Longitudinal/LMTP estimand factory and ATT factory.
- R wrapper via reticulate.
- Bregman extension (v2; design `LossSpec` abstraction now so v2 is additive).
