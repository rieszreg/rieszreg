# rieszboost

General-purpose gradient-boosting library for Riesz representers, implementing Lee & Schuler ([arXiv:2501.04871](https://arxiv.org/abs/2501.04871)).

## Living-doc rule

`README.md` is a living document — update it in the same edit whenever a change touches the public API surface (new estimand factory, new function exported from `rieszboost.__init__`, new engine), the supported feature list, the install/run instructions, or the quickstart example. If a change makes any line in the README false or outdated, the change is not done until the README is fixed. The README is the user-facing contract; CLAUDE.md is the implementation-side notes.

## Layout

- `python/` — primary implementation. Library is `rieszboost/`; tests in `tests/`. Build/dependency config in `pyproject.toml`.
- `reference/` — arXiv source for the relevant papers (gitignored). See `reference/README.md` for the index and refetch script.
- `.venv/` — local Python venv (gitignored once we add a top-level `.gitignore`).
- `r/` — R wrapper via reticulate (not yet built; reuse the Python core when added).

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
- ATE / TSM / AdditiveShift estimand factories.
- `init={0, float, 'm1'}` initialization; m1 traces m on a constant alpha=1.
- 12 tests passing, including end-to-end ATE Riesz recovery on the Lee-Schuler binary-treatment DGP (RMSE 0.64 at n=4000 — better than Lee-Schuler's reported 0.92 at n=500, as expected).

## What's next (per `~/.claude/plans/i-d-like-to-write-crystalline-raven.md`)

- Cross-fitting, diagnostics (‖α̂‖, overlap warnings), early-stopping on validation Riesz loss.
- lightgbm engine adapter.
- Slow general path with sklearn/JAX base learners.
- Longitudinal/LMTP estimand factory.
- R wrapper via reticulate.
- Bregman extension (v2; design `LossSpec` abstraction now so v2 is additive).
