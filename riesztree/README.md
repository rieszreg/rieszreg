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
- **38 Python tests** covering decoupling, Backend Protocol, growth policies, pruning, early stopping, categorical, sklearn integration, save/load round-trip per estimand, KL on TSM, BoundedSquared clipping, leaf-self-parity.

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

The v0.0.1 names `max_leaves` and `pruning_alpha` are accepted as deprecated aliases for `max_leaf_nodes` and `ccp_alpha`; passing them emits a `FutureWarning` and behaves identically.

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
