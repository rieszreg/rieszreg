# riesztree

Single-tree backend for the [RieszReg meta-package](../README.md). Greedy splits on the augmented Bregman-Riesz loss; closed-form per-leaf α* = -C/D (loss-projected to the link's α-domain).

This package depends on `rieszreg` for shared abstractions (`Estimand`, `Loss`, `Backend` Protocol, `Diagnostics`, `RieszEstimator` orchestrator, `AugmentedDataset`). See [`../rieszreg/DESIGN.md`](../rieszreg/DESIGN.md) for the meta-package design and the contract every implementation package follows. `riesztree` contributes:

- `RieszTreeBackend` — `Backend` Protocol implementation (the augmentation-style entry point). Consumes the precomputed `AugmentedDataset` and grows / prunes a single decision tree with loss-aware splits.
- `RieszTreeRegressor` — convenience subclass of `rieszreg.RieszEstimator` with tree-specific hyperparameters mirroring `sklearn.tree.DecisionTreeRegressor` where the augmented Bregman-Riesz setting allows: `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_weight_fraction_leaf`, `max_leaf_nodes`, `max_features`, `growth_policy`, `min_impurity_decrease`, `ccp_alpha`, `early_stopping_rounds`, `validation_fraction`, `categorical_features`. The v0.0.1 names `max_leaves`/`pruning_alpha` remain as deprecated aliases (emit `FutureWarning`).
- `TreeDiagnostics` — extends `rieszreg.Diagnostics` with `n_leaves`, `max_depth_actual`, `mean_leaf_size`, `feature_importances`.
- `RieszTreePredictor` — walks the tree to predict α; registers itself for `RieszEstimator.load` via `register_predictor_loader("riesztree", ...)`. Internally builds a Cython-backed flat-array companion (`riesztree.fast.FlatTree`) on first predict for a C-speed tight loop; the `Node` tree remains the source of truth for diagnostics, pruning, and serialization.
- `riesztree.fast` — compiled extensions:
  - `FlatTree` + Cython `predict_alpha` (`fast/_tree_c.pyx`).
  - `_loss_kernels.pyx` — built-in leaf-loss + alpha-at-opt kernels for the four Bregman losses.
  - `_splitter_c.pyx` — Cython continuous-feature best-split sweep, used when `splitter="exact"` (the default). Includes the random-threshold variant for `splitter="random"`.
  - `_splitter_hist.pyx` — Cython histogram splitter, used when `splitter="hist"`. Per-leaf histogram accumulation + sweep over bin boundaries; quantile pre-binning via `_binner.py`.
  - `_binner.py` — quantile `BinMapper` for the histogram splitter (sklearn HGB-style; default 255 bins).
  - `_splitter.py` — Python facade. Maps a `Loss` to a loss-kind integer + bounded clip parameters; dispatches to exact / hist / random Cython kernels; provides `register_fast_leaf_solver(LossClass, leaf_loss_cfunc, alpha_at_opt)` for users to plug a Numba `@cfunc` into the splitter for any custom `Loss`.
- R6 wrapper subclassing `rieszreg::RieszEstimatorR6`.

## Living-doc rule (README + meta-project docs)

`README.md` is a living document — update it in the same edit whenever a change touches the public API surface (new hyperparameter, new growth policy, new diagnostic). If a change makes any line in the README false, the change is not done until the README is fixed.

The user guide is the unified Quarto site at [`../docs/`](../docs/). The tree-specific page is [`../docs/backends/tree.qmd`](../docs/backends/tree.qmd). Any change to the tree backend that affects user-facing behaviour must update that page in the same edit.

## API design rule

Mirrors **ngboost / sklearn**:

- Object-oriented factory `RieszTreeRegressor(estimand=, max_depth=, ...)`. `BaseEstimator`-compatible `fit / predict / score / diagnose`.
- **No `feature_keys` (or other input-schema args) on `fit()` / `predict()`.** The estimand owns its input schema.
- Cross-fitting is `sklearn.model_selection.cross_val_predict`. No bespoke `crossfit()`.
- Hyperparameter tuning is `sklearn.model_selection.GridSearchCV`. No `tune_riesz()`.

R-side mirrors this: R6 class `RieszTreeRegressor$new(estimand=, max_depth=, ...)$fit(df)$predict(df)`.

## Layout

- `python/riesztree/` — `splitter.py` (per-loss leaf-loss + best-split sweep), `tree.py` (Node + traversal + serialisation), `grow.py` (depthwise + leafwise), `pruning.py` (cost-complexity), `backend.py` (`RieszTreeBackend`), `predictor.py` (`RieszTreePredictor` + loader registration), `estimator.py` (`RieszTreeRegressor` convenience subclass), `diagnostics.py` (`TreeDiagnostics`), `fast/` (`FlatTree` + Cython `predict` extension).
- `r/riesztree/` — R6 wrapper via reticulate. `RieszTreeRegressor` subclasses `rieszreg::RieszEstimatorR6`.
- `examples/` — runnable demonstrations of each built-in estimand (ATE, ATT, TSM, AdditiveShift, LocalShift).
- `python/tests/` — 122 tests covering decoupling, Backend Protocol, growth policies, pruning, early stopping, categorical, sklearn integration, save/load round-trip per estimand, KL on TSM, BoundedSquared clipping, leaf-self-parity, sklearn-style hyperparameter parity, flat-tree predict parity, Cython↔Python splitter parity, user-loss registration, histogram splitter parity, random splitter, deprecation of the python splitter.
- `python/benchmarks/` — `bench_fit.py` (locked perf grid; baseline in `BENCH_BASELINE.md`) and `bench_compare.py` (sklearn DTR / HGB, LightGBM, XGBoost comparison).

## Run tests

```sh
cd /Users/aschuler/Desktop/RieszReg/riesztree && \
  /Users/aschuler/Desktop/RieszReg/.venv/bin/python -m pytest python/tests -v
```

R parity:

```sh
Rscript -e '
  Sys.setenv(RETICULATE_PYTHON = file.path(getwd(), "../.venv/bin/python"))
  pkgload::load_all("../rieszreg/r/rieszreg")
  pkgload::load_all("r/riesztree")
  testthat::test_dir("r/riesztree/tests/testthat")
'
```

## Architecture notes

### Dependency on rieszreg

`riesztree` depends on `rieszreg` and reuses, without modification:

- `Estimand`, `Tracer`/`LinearForm`, `trace`, `Estimand.augment`, `AugmentedDataset` — the augmentation engine.
- `Loss`, `SquaredLoss`, `KLLoss`, `BernoulliLoss`, `BoundedSquaredLoss` — the Bregman-Riesz loss framework.
- `Diagnostics`, `diagnose` — base diagnostics (`TreeDiagnostics` extends with tree-specific extras).
- `RieszEstimator` — orchestration; `RieszTreeRegressor` is a thin subclass with the tree backend defaulted.
- `register_predictor_loader` — registry-based save/load.

The integration point is `rieszreg`'s `Backend` Protocol (`rieszreg/backends/base.py`). `RieszTreeBackend.fit_augmented(...)` consumes the precomputed `AugmentedDataset` and returns a `FitResult`.

### Loss-aware splits

The per-leaf optimum α* = -C/D is universal across the four built-in Bregman losses, but the **leaf-loss-at-optimum** (and therefore the split-gain) is loss-specific. Each loss has its own analytic form:

- `SquaredLoss`: L(α*) = -C²/D.
- `KLLoss`: L(α*) = -C + C·log(-C/D) when C < 0; +∞ when C > 0 (infeasible — disqualifies the split).
- `BernoulliLoss`: L(α*) = D·log D - (D+C)·log(D+C) + C·log(-C) when -D < C < 0; +∞ otherwise.
- `BoundedSquaredLoss(lo, hi)`: project α* into [lo, hi] and evaluate D·α*² + 2C·α*.

The dispatcher lives in `splitter.make_leaf_solvers`. Custom Loss subclasses raise `NotImplementedError` with a clear message — extend the dispatcher there.

### Backend Protocol choice

We implement `Backend.fit_augmented` (augmentation-style), not `MomentBackend.fit_rows`. The augmentation-style splitter handles every built-in *and* custom estimand without requiring a sieve, mirroring `forestriesz/AugForestRieszBackend`. The trade-off (M ≈ k·n training rows vs n) is irrelevant for a single tree at the dataset sizes a single tree is sensible for.

### Predictor representation

The tree is a Python class hierarchy (`tree.Node`); prediction walks the tree in pure Python. For n=10⁵ predictions × depth ~10 the dispatch overhead is tens of milliseconds, well below the splitter's cost. A flat-array representation could replace this once profiling justifies it.

### Reference parity

No prior implementation of this method exists (per `RIESZTREE_DESIGN.md` §1, the algorithm is new), so per `rieszreg/DESIGN.md` §5.2 the package documents the absence and ships a self-parity test (`tests/test_self_parity.py`) that verifies the splitter's leaf values agree with a hand-applied closed-form on the same final partition.

## What works today (v0.0.1)

See [`README.md` § What works today](README.md#what-works-today-v001).

## Known sharp edges

See [`README.md` § Known sharp edges](README.md#known-sharp-edges).

## What's next

See [`README.md` § On the roadmap](README.md#on-the-roadmap). Headlines: honest splits + CIs, model-tree leaves, treatment-dimension sieves.
