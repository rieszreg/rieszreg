# forestriesz

Random-forest Riesz regression, in **Python** and **R**. A learner package in the [RieszReg family](https://github.com/rieszreg/rieszreg): estimates the Riesz representer α of a linear estimand ψ = E[m(μ)(Z)] using a generalized random forest.

Two backends ship side-by-side:

- **`AugForestRieszRegressor`** — augmentation-style. An ensemble of single-tree Riesz regressors (built on `riesztree.RieszTreeBackend`) fit on the augmented dataset of evaluation points $z_r$ with weights $(D_r, C_r)$ that `Estimand.augment` produces. sklearn `RandomForestRegressor`-style hyperparameters; loss-aware splits handle every built-in Bregman loss directly. **Works on every estimand without per-estimand configuration.** No CIs in v1 (the augmented training set has correlated blocks per original row).
- **`ForestRieszRegressor`** — moment-style. Implements [Chernozhukov, Newey, Quintas-Martínez, Syrgkanis (ICML 2022)](https://proceedings.mlr.press/v162/chernozhukov22a/chernozhukov22a.pdf) on top of EconML's GRF. Trains on $n$ original rows; the user supplies a list of basis functions of the data (`riesz_feature_fns`; auto-resolved for ATE/ATT/TSM). Supports honest-split `predict_interval` for single-basis fits.

See the [forest backend docs page](https://rieszreg.github.io/rieszreg/backends/forest.html) for the full comparison.

## Status

v0.0.2 — feature-complete for single-stage Riesz regression. Both backends are **sklearn-compatible** (compose with `GridSearchCV`, `cross_val_predict`, `clone`, `Pipeline`), expose six built-in estimand factories (re-exported from rieszreg), and round-trip via `save`/`load`. R6 wrapper covers the moment-style backend.

## Why

Forests are non-parametric, scale to many covariates without a kernel choice or boosting hyperparameter sweep, and (in the moment-style backend) provide asymptotic confidence intervals on $\alpha(z)$. The augmentation-style backend trades the CIs for working on every estimand without per-estimand configuration.

## Install

`forestriesz` depends on `rieszreg` (shared abstractions) and `riesztree` (per-tree augmented learner). The simplest install is from the rieszreg-monorepo workspace root:

```sh
git clone https://github.com/rieszreg/rieszreg.git
cd rieszreg
uv sync --all-packages --all-extras
```

This editable-installs every package in the workspace, including `forestriesz`. `econml` and `riesztree` are pulled in as dependencies; both ship prebuilt wheels.

## Quickstart (Python) — ATE

```python
import numpy as np
import pandas as pd
from forestriesz import ForestRieszRegressor, ATE

# Synthetic binary-treatment data
rng = np.random.default_rng(0)
n = 1500
x = rng.uniform(0, 1, n)
pi = 1 / (1 + np.exp(-(-0.02*x - x**2 + 4*np.log(x + 0.3) + 1.5)))
a = rng.binomial(1, pi).astype(float)
df = pd.DataFrame({"a": a, "x": x})

fr = ForestRieszRegressor(
    estimand=ATE(treatment="a", covariates=("x",)),
    n_estimators=500,
    min_samples_leaf=10,
    random_state=0,
)
fr.fit(df)
alpha_hat = fr.predict(df)
# alpha_hat ≈ A/π(X) - (1-A)/(1-π(X)) — without ever estimating π(X)
```

The default `riesz_feature_fns="auto"` resolves to a sensible sieve for built-in estimands (treatment indicators for ATE/ATT/TSM). Pass an explicit list of basis callables to override.

## Quickstart (Python) — AdditiveShift with the augmentation-style backend

For estimands without a canonical list of basis functions — `AdditiveShift`, `LocalShift`, custom user moments — `ForestRieszRegressor` raises a row-constant degeneracy error. `AugForestRieszRegressor` handles them with no extra configuration:

```python
from forestriesz import AugForestRieszRegressor, AdditiveShift

# Continuous-treatment data
a_cont = rng.normal(0.5*x, 1.0)
df_cont = pd.DataFrame({"a": a_cont, "x": x})

fr = AugForestRieszRegressor(
    estimand=AdditiveShift(delta=0.5),
    n_estimators=500,
    min_samples_leaf=10,
    random_state=0,
)
fr.fit(df_cont)
alpha_hat = fr.predict(df_cont)
# Forest learns a local density-ratio estimator directly from the augmented data.
```

It also works on every built-in estimand (ATE, ATT, TSM, …) without any extra arguments. Trade-off: no `predict_interval` in v1.

## Quickstart (Python) — TSM with confidence intervals

```python
from forestriesz import ForestRieszRegressor, TSM

fr = ForestRieszRegressor(
    estimand=TSM(level=1, treatment="a", covariates=("x",)),
    n_estimators=500,
    honest=True,
    inference=True,
    random_state=0,
)
fr.fit(df)
alpha_hat = fr.predict(df)
lb, ub = fr.predict_interval(df, alpha=0.05)  # 95% CI per row
```

`honest=True` enables the GRF half-sample honest-split scheme so the CIs are asymptotically valid. `inference=True` retains the per-tree subsample structure needed for variance estimation (requires `n_estimators` divisible by `subforest_size=4`).

## Quickstart (R) — TSM

```r
library(forestriesz)
use_python_forestriesz(".venv/bin/python")

fr <- ForestRieszRegressor$new(
  estimand = TSM(level = 1L, treatment = "a", covariates = "x"),
  n_estimators = 500L,
  honest = TRUE,
  inference = TRUE
)
fr$fit(df)
alpha_hat <- fr$predict(df)
ci <- fr$predict_interval(df, alpha = 0.05)
```

The R wrapper exposes locally constant fits (single-basis sieve under the hood for built-in estimands). For ATE/ATT and other multi-basis sieves, call into Python via reticulate.

## Built-in estimands

Re-exported from rieszreg — same API, same semantics:

| Factory | $m(\mu)(z, y)$ | Default `riesz_feature_fns` (moment-style) |
|---|---|---|
| `ATE(treatment, covariates)` | μ(1, x) − μ(0, x) | `[1{A=0}, 1{A=1}]` |
| `ATT(treatment, covariates)` | a · (μ(1, x) − μ(0, x)) | `[1{A=0}, 1{A=1}]` |
| `TSM(level, treatment, covariates)` | μ(level, x) | `[1{A=level}]` |
| `AdditiveShift(delta, ...)` | μ(a + δ, x) − μ(a, x) | none — use `AugForestRieszRegressor` |
| `LocalShift(delta, threshold, ...)` | 1(a < threshold) · (μ(a + δ, x) − μ(a, x)) | none — use `AugForestRieszRegressor` |

`StochasticIntervention` previously appeared here; it is currently being rewritten and will return.

`AugForestRieszRegressor` works on every row in this table (and any user-defined `Estimand`) without any extra arguments.

## Architecture

### Augmentation-style: `AugForestRieszBackend` (`Backend.fit_augmented`)

An ensemble of `riesztree.RieszTreeBackend` instances. For each tree, original-row indices are sampled (with or without replacement per `bootstrap`) and expanded to the corresponding block of augmented rows; the tree is grown by riesztree's loss-aware splitter directly on the augmented dataset. The forest averages the per-tree $\hat\alpha$ predictions.

Augmented rows carry weights $D_r$ (1 if $z_r$ is the original observation, 0 otherwise) and $C_r$ (the trace coefficient at $z_r$). The empirical Bregman-Riesz loss decomposes as $\sum_r [D_r \tilde h(\alpha(z_r)) + C_r h'(\alpha(z_r))]$, so each leaf has a closed-form per-loss optimum. For ATE the per-leaf solve recovers $1/\hat P(A=a \mid X\text{-leaf})$; for AdditiveShift it recovers a local density-ratio estimator. No user-supplied basis functions are needed — the augmented row weights already vary per row.

When `splitter="hist"` on "simple" configurations (no categoricals, default `max_features`, no `ccp_alpha`, no leaf cap, built-in loss), the bin mapper is fitted once on the full augmented training data and reused across joblib workers — `sklearn.ensemble.HistGradientBoostingRegressor` convention. Saves `n_estimators - 1` repeats of `fit_bin_mapper + transform`. The win is largest at shallow depths where per-tree binning dominates tree-build cost (~2× faster at `max_depth=8`).

### Moment-style: `ForestRieszBackend` (`MomentBackend.fit_rows`)

Per-row moments $m(\varphi)(Z_i, Y_i)$ are computed from `rieszreg.trace(estimand, Z_i)` for each user-supplied basis function $\varphi_j$ and packed into EconML's linear-moment slot:

```
A[i, j] = Σ_(coef, point) ∈ trace(Z_i)  coef · φ_j(point)         (per-row moment vector)
J[i]     = φ(Z_i) φ(Z_i)'                                          (per-row Jacobian)
```

In each leaf the closed-form solve is `θ_ℓ = (Σ_i J_i)^{-1} Σ_i A_i`. The MSE splitting criterion picks splits that minimize sum of in-leaf residuals against this leaf optimum — exactly what the paper's reference implementation uses.

## Sklearn integration

```python
from sklearn.model_selection import GridSearchCV, cross_val_predict

# Hyperparameter sweep
gs = GridSearchCV(
    ForestRieszRegressor(estimand=ATE()),
    param_grid={"max_depth": [3, 5, 8], "min_samples_leaf": [5, 15, 50]},
    cv=3,
)
gs.fit(df)

# Cross-fit predictions (debiased ML)
alpha_hat_cv = cross_val_predict(
    ForestRieszRegressor(estimand=ATE(), n_estimators=500),
    df, cv=5,
)
```

## Save / load

```python
fr.fit(df)
fr.save("path/to/dir")

# Reloads with default sieve auto-resolved from estimand metadata
loaded = ForestRieszRegressor.load("path/to/dir")
loaded.predict(df)
```

User-supplied callables in `riesz_feature_fns` are not pickled (lambdas, closures don't round-trip reliably across processes). For custom sieves, repass them at load:

```python
loaded = ForestRieszRegressor.load("path/to/dir", riesz_feature_fns=my_basis)
```

## Diagnostics

```python
from forestriesz import diagnose_forest

diagnose_forest(fr, df).summary()
# -> RMS magnitude, alpha quantiles, extreme-value warnings,
#    feature_importances, mean leaf size, mean leaf count per tree.
```

## Picking a backend

| | `AugForestRieszRegressor` (aug-style) | `ForestRieszRegressor` (moment-style) |
|---|---|---|
| Per-estimand setup | none — works on every estimand directly | user supplies `riesz_feature_fns` (auto-resolved for ATE/ATT/TSM) |
| Loss support | `SquaredLoss`, `KLLoss`, `BernoulliLoss`, `BoundedSquaredLoss` | `SquaredLoss` only |
| `predict_interval` | not in v1 (augmented training rows are correlated within blocks) | yes (single-basis fits, honest splits) |
| Splits on | full feature space | covariates only when the basis handles treatment |
| Backend Protocol | `Backend.fit_augmented` | `MomentBackend.fit_rows` |

Reach for `AugForestRieszRegressor` for general use, especially shift-style or custom estimands. Default to `ForestRieszRegressor` when you need CIs on ATE/ATT/TSM.

## Bregman losses

`AugForestRieszRegressor` supports all four built-in losses: `SquaredLoss` (default), `KLLoss` (density-ratio estimands; α̂ in the non-negative reals), `BernoulliLoss` (α̂ in [0, 1]), and `BoundedSquaredLoss(lo, hi)` (α̂ in [lo, hi]). riesztree's per-tree splitter dispatches on the loss directly — splits are chosen against the loss the user supplied.

```python
from forestriesz import AugForestRieszRegressor, TSM
from rieszreg import KLLoss

# Density-ratio fit on TSM. KLLoss requires non-negative m-coefficients
# (TSM, IPSI, and similar density-ratio estimands satisfy this).
est = AugForestRieszRegressor(estimand=TSM(level=1), loss=KLLoss())
est.fit(df)
alpha_hat = est.predict(df)
```

`ForestRieszRegressor` (moment-style) is `SquaredLoss`-only.

## Known sharp edges

- **`predict_interval` is moment-style + single-basis only.** For multi-basis fits (e.g. ATE's `[1{A=0}, 1{A=1}]`), CIs need a delta-method on θ' φ(x). For `AugForestRieszRegressor`, CIs need cluster-robust variance with `origin_index` as the cluster id. Both planned for v2.
- **`ForestRieszRegressor` is squared-only.** Use `AugForestRieszRegressor` for the other Bregman losses.
- **Honest splits + inference require `n_estimators % subforest_size == 0`** (EconML constraint, moment-style only). Default `subforest_size=4`, so `n_estimators=100, 200, 500, ...` are safe values.
- **`riesz_feature_fns` callables don't auto-save.** Save persists the forest; load needs the callables repassed for custom bases. Built-in estimands round-trip fine via `riesz_feature_fns="auto"` (the default).
- **R wrapper exposes `ForestRieszRegressor` only.** `AugForestRieszRegressor` works from Python; call it from R via reticulate if needed.
- **Constant basis with `ForestRieszRegressor` is degenerate for built-in estimands.** Forcing `riesz_feature_fns=None` raises a row-constant check error. The default `"auto"` does the right thing. `AugForestRieszRegressor` is unaffected.
- **`KLLoss` and `BernoulliLoss` require non-negative C-coefficients in the augmented dataset** (density-ratio-style estimands like TSM and IPSI). They reject ATE / ATT / shift-style data at fit time with a clear error.
- **Forest backends don't take `validation_fraction`.** Forests don't use a held-out slice for fit-time logic; validation loss is reported only when you pass `eval_set=` explicitly to `fit()`. (See rieszreg `DESIGN.md §A.2` for the agnostic-orchestrator rule that puts `validation_fraction` on the backends that actually use it: boosting, kernel ridge, neural.)

## On the roadmap

- v3: Bregman losses for the moment-style `ForestRieszRegressor`.
- v2: delta-method `predict_interval` for multi-basis moment-style fits.
- v2: cluster-robust `predict_interval` for `AugForestRieszRegressor`.
- R wrapper for `AugForestRieszRegressor`.

## Living-doc rule

`README.md` is a living document — update it in the same edit whenever a change touches the public API surface. The user guide is the unified Quarto site at [rieszreg.github.io/rieszreg](https://rieszreg.github.io/rieszreg/); the forest-specific page is [forest backend](https://rieszreg.github.io/rieszreg/backends/forest.html).

## License

MIT.
