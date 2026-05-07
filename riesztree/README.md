# riesztree

Single-tree backend for the [RieszReg meta-package](../README.md). Fits a single decision tree to estimate the Riesz representer of a linear functional via greedy splits on the augmented Bregman-Riesz loss.

Each leaf stores the closed-form per-leaf optimum

$$\alpha_\ell^* = -\,C_\ell / D_\ell,$$

projected to the loss's α-domain. The same per-leaf formula is universal across `SquaredLoss`, `KLLoss`, `BernoulliLoss`, and `BoundedSquaredLoss` — only the split-gain function depends on the loss.

## Why a single tree?

Forests (`forestriesz`) and gradient boosting (`rieszboost`) are the right learners for headline accuracy. A single tree fills a different niche:

- **Interpretability.** A depth-3 or depth-4 tree is a readable rule. Each leaf's α* is a subgroup-specific IPW-like weight you can stare at.
- **No external heavy dependencies.** Pure NumPy + SciPy + scikit-learn; no EconML, XGBoost, JAX, or Torch.
- **Custom estimands without a sieve.** Augmentation-style splitter handles `AdditiveShift`, `LocalShift`, and any user `FiniteEvalEstimand` without configuration. (See `forestriesz/AugForestRieszRegressor` for the related forest variant.)

## Install

```sh
pip install -e python/   # from this directory
```

`riesztree` ships a small Cython extension (`riesztree.fast._tree_c`) that
backs the prediction tight loop. `pip install -e python/` builds it
automatically — you need a C compiler on the build machine
(gcc / clang / MSVC). Editing a `.pyx` file requires re-running
`pip install -e python/` to recompile.

R:

```r
pkgload::load_all("../rieszreg/r/rieszreg")
pkgload::load_all("r/riesztree")
```

## Quickstart

```python
import numpy as np, pandas as pd
from riesztree import RieszTreeRegressor, ATE

rng = np.random.default_rng(0)
n = 1500
x = rng.uniform(0, 1, n)
pi = 1 / (1 + np.exp(-(-0.02*x - x**2 + 4*np.log(x + 0.3) + 1.5)))
a = rng.binomial(1, pi).astype(float)
df = pd.DataFrame({"a": a, "x": x})

est = RieszTreeRegressor(
    estimand=ATE(treatment="a", covariates=("x",)),
    max_depth=4,
    random_state=0,
)
est.fit(df)
alpha_hat = est.predict(df)
```

## What works today (v0.0.1)

- **`RieszTreeRegressor(BaseEstimator)`** — sklearn-compatible. Composes with `GridSearchCV`, `cross_val_predict`, `clone`, `Pipeline`.
- **All five built-in estimands** via the rieszreg re-exports: `ATE`, `ATT`, `TSM`, `AdditiveShift`, `LocalShift`. Custom `FiniteEvalEstimand`s also work — the augmentation-style splitter handles them without a sieve.
- **All four built-in losses**: `SquaredLoss` (default), `KLLoss`, `BernoulliLoss`, `BoundedSquaredLoss`. Splits are loss-aware: each loss has its own analytic per-leaf objective and the splitter optimises the corresponding gain.
- **Two growth policies**: `growth_policy="depthwise"` (default; recursive depth-first) and `"leafwise"` (best-first growth, capped by `max_leaves`).
- **Cost-complexity pruning** via `pruning_alpha > 0`. Default off.
- **Early stopping** via `early_stopping_rounds` + `validation_fraction`. Default off.
- **Categorical predictors** via `categorical_features=(col_idx, ...)`. Splits use the standard CART trick: order levels by within-level α* and sweep contiguous splits.
- **Save / load**: directory format with JSON predictor + JSON metadata. Built-in estimands round-trip automatically.
- **Diagnostics**: `TreeDiagnostics` extends `rieszreg.Diagnostics` with `n_leaves`, `max_depth_actual`, `mean_leaf_size`, `feature_importances` (per-feature normalised split-gain).
- **R wrapper**: R6 mirror via reticulate.
- **Cython prediction**: `predict` walks a flat-array tree (built once per fit) at C speed. The `Node` tree continues to back diagnostics, pruning, and serialization.
- **Three Cython splitter paths**: `splitter="exact"` (default; per-feature threshold sweep), `splitter="hist"` (quantile-binned histogram, fastest at large `n`), `splitter="random"` (sklearn ExtraTrees-style; one uniform threshold per feature). `splitter="python"` keeps the legacy pure-Python path (deprecated, scheduled for removal in v0.0.3).
- **Custom-loss extension hook**: `riesztree.fast.register_fast_leaf_solver(LossClass, leaf_loss_cfunc, alpha_at_opt)` plugs a Numba `@cfunc` (signature `float64(float64, float64)`) into the Cython splitter for any user `Loss` subclass.
- **122 Python tests** covering decoupling, Backend Protocol, growth policies, pruning, early stopping, categorical, sklearn integration, save/load round-trip per estimand, KL on TSM, BoundedSquared clipping, leaf-self-parity, sklearn-style hyperparameter parity, flat-tree predict parity, Cython↔Python splitter parity, user-loss registration, histogram splitter parity, random splitter, deprecation of the python splitter.

## Hyperparameters

| Knob | Default | Notes |
|---|---|---|
| `max_depth` | 8 | Cap on tree depth. |
| `min_samples_split` | 20 | Minimum count of original (D > 0) augmented rows in a node before considering a split. |
| `min_samples_leaf` | 10 | Minimum count of original rows in each child. |
| `min_weight_fraction_leaf` | 0.0 | Sklearn parity. With unit weights, leaves must hold ≥ `ceil(min_weight_fraction_leaf · n_original)` original rows (combined with `min_samples_leaf` via `max(...)`). |
| `max_leaf_nodes` | 31 | Cap for leafwise growth. Ignored when `growth_policy="depthwise"`. |
| `max_features` | None | Per-split feature subsample. `None`, `"sqrt"`, `"log2"`, an int, or a float in `(0, 1]`. Sklearn convention. |
| `growth_policy` | `"depthwise"` | Or `"leafwise"`. |
| `min_impurity_decrease` | 0.0 | Reject splits with gain ≤ this threshold. |
| `ccp_alpha` | 0.0 | Cost-complexity pruning penalty. Sklearn name. |
| `early_stopping_rounds` | None | Stop when held-out augmented loss has not improved for that many splits. |
| `validation_fraction` | 0.1 | Held-out fraction for early stopping. Ignored when not needed. |
| `categorical_features` | None | Sequence of column indices treated as integer category labels. |
| `loss` | `SquaredLoss()` | Bregman-Riesz loss. |
| `random_state` | 0 | Seeds the per-split feature subsample under `max_features`. |
| `splitter` | `"exact"` | One of `"exact"`, `"hist"`, `"random"`, `"python"`. See the Splitter modes section below. |
| `max_bins` | 255 | Bins per feature when `splitter="hist"`. Sklearn HGB convention; fits in `uint8`. |

The v0.0.1 names `max_leaves` and `pruning_alpha` are accepted as deprecated aliases for `max_leaf_nodes` and `ccp_alpha`; passing them emits a `FutureWarning` and behaves identically. `splitter="python"` is also deprecated and will be removed in v0.0.3.

## Splitter modes

| `splitter=` | When to use | Implementation |
|---|---|---|
| `"exact"` (default) | Most fits. Best partition; per-feature linear scan over distinct values. | Cython per-feature sweep with C-call dispatch into the Bregman leaf-loss kernels in `riesztree.fast._loss_kernels`. |
| `"hist"` | Large `n` (≥ 10⁵), or inside a forest where slight discretization is fine. | Quantile pre-binning (`max_bins`, default 255) once per fit; per-leaf histogram accumulation + sweep, all in Cython. |
| `"random"` | ExtraTrees-style forests; cheap baseline. | One uniform threshold per feature per leaf; single Cython pass. |
| `"python"` | Legacy / debugging. Deprecated, removed in v0.0.3. | Pure-Python sweep over distinct values. |

Custom user `Loss` subclasses are routed to the Cython exact / hist / random paths once registered via [`riesztree.fast.register_fast_leaf_solver`](#custom-loss-extension-hook); unregistered ones fall back to the Python path with a one-time warning.

## Custom loss extension hook

```python
import numba
from rieszreg import Loss
from riesztree.fast import register_fast_leaf_solver

class MyLoss(Loss):
    ...   # implement the Loss interface

@numba.cfunc("float64(float64, float64)", cache=True, nopython=True)
def my_leaf_loss(D, C):
    if D <= 0.0:
        return 0.0
    return -C * C / D                      # SquaredLoss-equivalent here

def my_alpha(D, C):
    return 0.0 if D <= 0 else -C / D

register_fast_leaf_solver(MyLoss, my_leaf_loss, my_alpha)

# Subsequent fits with loss=MyLoss() use the C-speed Cython splitter.
```

The cfunc is called from the Cython splitter's tight loop without the GIL — same per-evaluation cost as the four built-in losses.

## Speed

`bench_fit.py` and `bench_compare.py` (under `python/benchmarks/`) document where `riesztree` sits versus state-of-the-art tree libraries on equivalent `(n_aug, p, max_depth)` workloads. Headline cell `(n_aug=100k, p=20, depth=16)` with `splitter="hist"`:

| Library | Fit time | vs riesztree |
|---|---|---|
| **riesztree-hist** | **0.62 s** | 1.0× |
| sklearn `HistGradientBoostingRegressor` (max_iter=1) | 0.64 s | parity |
| sklearn `DecisionTreeRegressor` (exact) | 2.44 s | we're 4× faster |
| XGBoost (n_estimators=1, hist) | 0.36 s | 1.7× behind |
| LightGBM (n_estimators=1) | 5.67 s | we're 9× faster |

We're at parity with sklearn HGB, faster than sklearn-DTR, and within ~1.7× of XGBoost. Concrete attribution for the remaining XGBoost gap (Python-level growth loop, no parent-minus-sibling histogram trick, no presort reuse, per-split numpy index allocation) is left as documented future work in [`BENCH_BASELINE.md`](BENCH_BASELINE.md). See that file for the full benchmark protocol and locked baselines.

## Known sharp edges

- **High variance.** A single tree has higher RMSE than a forest or booster on the same DGP — cf. `forestriesz` / `rieszboost` for accuracy-first work.
- **No confidence intervals in v1.** Honest splits are not implemented.
- **Constant per-leaf α only.** Linear-in-X leaves (model trees) and treatment-dimension sieves are documented as future work but not implemented.
- **KL / Bernoulli + difference-of-evaluations estimands.** Splits that produce a leaf with `C > 0` (KL) or `C` outside `(-D, 0)` (Bernoulli) are disqualified. KL is intended for density-ratio estimands like `TSM`; pairing it with `ATE` will produce many disqualified splits.

## On the roadmap

- Honest splits + confidence intervals.
- Linear-in-X leaves (model trees).
- Treatment-dimension sieves (the `forestriesz` `riesz_feature_fns="auto"` trick).
- Loss-aware splits for additional Bregman losses beyond the four built-ins.
- Reference-parity test once a second implementation exists.
