# forestriesz

Random-forest Riesz regression, in **Python** and **R**. Sister package to [krrr](https://github.com/rieszreg/krrr) and [rieszboost](https://github.com/rieszreg/rieszboost) in the [RieszReg family](https://github.com/rieszreg/rieszreg): same scope (Riesz representers for linear functionals; θ(P) = E[m(Z, g₀)]), same user-facing API, but the fitter is a generalized random forest.

Two backends ship side-by-side, both built on EconML's `BaseGRF`:

- **`ForestRieszRegressor`** — moment-style. Implements [Chernozhukov, Newey, Quintas-Martínez, Syrgkanis (ICML 2022)](https://proceedings.mlr.press/v162/chernozhukov22a/chernozhukov22a.pdf). Trains on n original rows; needs a sieve (auto-resolved for ATE/ATT/TSM via `default_riesz_features`). Supports honest-split `predict_interval` for single-basis fits.
- **`AugForestRieszRegressor`** — augmentation-style. Trains on the M = k·n augmented `(a, b)` dataset. **Estimand-agnostic** — works on every built-in estimand and any custom user `Estimand` without a sieve. No CIs in v1 (the augmented training set has correlated blocks per original row, breaking GRF's iid variance assumption).

The augmentation-style variant is novel — see `forestriesz/python/tests/test_aug_vs_moment.py` for a benchmark and the [forest backend docs page](https://rieszreg.github.io/rieszreg/backends/forest.html) for the full comparison.

## Status

v0.0.2 — feature-complete for single-stage Riesz regression. Both backends are **sklearn-compatible** (compose with `GridSearchCV`, `cross_val_predict`, `clone`, `Pipeline`), expose six built-in estimand factories (re-exported from rieszreg), and round-trip via `save`/`load`. R6 wrapper covers the moment-style backend.

## Why

Forests are the natural fit for problems where (a) you want a non-parametric Riesz estimator, (b) you have many covariates and don't want to commit to a kernel or boosting hyperparameter sweep, and (c) you care about asymptotic confidence intervals on α(x). `forestriesz` gives you all three through one estimator that defaults to the right configuration for the standard estimands.

## Install

`forestriesz` depends on the [rieszreg](https://github.com/rieszreg/rieszreg) meta-package. From the [RieszReg](https://github.com/rieszreg/rieszreg) repo root:

```sh
git clone https://github.com/rieszreg/forestriesz.git
git clone https://github.com/rieszreg/rieszreg.git
cd forestriesz
python3 -m venv .venv
.venv/bin/pip install -e ../rieszreg/python
.venv/bin/pip install -e python/
```

(Once both packages publish to PyPI, this collapses to `pip install forestriesz`.) `econml` is pulled in transitively and ships prebuilt wheels for macOS arm64 and x86_64 — no compiler toolchain required.

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

For estimands without a canonical sieve — `AdditiveShift`, `LocalShift`, custom user moments — `ForestRieszRegressor` raises a row-constant degeneracy error (the moment doesn't depend on W under a constant basis). `AugForestRieszRegressor` handles them with no extra configuration:

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
# Forest learns a local density-ratio estimator without needing a basis.
```

It also works on every built-in estimand (ATE, ATT, TSM, …) at near-identical RMSE and fit time. Trade-off: no `predict_interval` in v1 because the augmented training set has correlated blocks per original row.

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

| Factory | m(z, α) | Default sieve |
|---|---|---|
| `ATE(treatment, covariates)` | α(1, x) − α(0, x) | `[1{T=0}, 1{T=1}]` |
| `ATT(treatment, covariates)` | a · (α(1, x) − α(0, x)) | `[1{T=0}, 1{T=1}]` |
| `TSM(level, treatment, covariates)` | α(level, x) | `[1{T=level}]` |
| `AdditiveShift(delta, ...)` | α(a + δ, x) − α(a, x) | none — use `AugForestRieszRegressor` |
| `LocalShift(delta, threshold, ...)` | 1(a < threshold) · (α(a + δ, x) − α(a, x)) | none — use `AugForestRieszRegressor` |
| `StochasticIntervention(samples_key, ...)` | (1/K) Σₖ α(a'ₖ, x) | none |

For estimands without a built-in sieve, the moment varies in W naturally if your evaluation points (the (coef, point) pairs from `trace`) depend on the row data — `StochasticIntervention` is the canonical example. For estimands where it doesn't (`AdditiveShift`, `LocalShift`, custom user moments), use `AugForestRieszRegressor`.

## Architecture

### Moment-style: `ForestRieszBackend` (`MomentBackend.fit_rows`)

Per-row moments `m(W_i; φ_j)` are computed from `rieszreg.trace(estimand, W_i)` and packed into EconML's linear-moment slot:

```
A[i, j] = Σ_(coef, point) ∈ trace(W_i)  coef · φ_j(point)         (per-row moment vector)
J[i]     = φ(W_i) φ(W_i)'                                          (per-row Jacobian)
```

In each leaf the closed-form solve is `θ_ℓ = (Σ_i J_i)^{-1} Σ_i A_i`. The MSE splitting criterion picks splits that minimize sum of in-leaf residuals against this leaf optimum — exactly what the paper's reference implementation uses.

### Augmentation-style: `AugForestRieszBackend` (`Backend.fit_augmented`)

Trained on the augmented dataset of M = k·n evaluation points produced by `rieszreg.build_augmented`. Per *augmented* row:

```
J_k = 2 a_k · φ(z_k) φ(z_k)'        # zero for counterfactual eval points (a_k=0)
A_k = -b_k · φ(z_k)                  # nonzero where the trace placed mass
```

Even with a constant basis the per-row J and A vary across augmented rows (originals contribute J, counterfactual eval points contribute A), so the forest can split on the full feature space without a sieve. For ATE the leaf solve recovers `1/P̂(T=t | X-leaf)`; for AdditiveShift it recovers a local density-ratio estimator. The combination of GRF + augmented training data appears to be novel — see [test_aug_vs_moment.py](python/tests/test_aug_vs_moment.py) for a benchmark.

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

| | `ForestRieszRegressor` (moment-style) | `AugForestRieszRegressor` (aug-style) |
|---|---|---|
| Estimand support | ATE/ATT/TSM via auto sieve; others need user-supplied sieve | Every built-in estimand + custom user `Estimand`, no sieve needed |
| Loss support | `SquaredLoss` only | `SquaredLoss`, `KLLoss`, `BernoulliLoss`, `BoundedSquaredLoss` |
| `predict_interval` | yes (single-basis fits, honest splits) | not in v1 (augmented training rows are correlated within blocks) |
| Per-fit speed | baseline | ≈1.0× at n≤2000, ≈1.15× at n=5000 (see `test_aug_vs_moment.py`) |
| RMSE on shared estimands | baseline | within ~5% |
| Splits on | covariates only when sieve handles treatment | full feature space (T, X) |
| Backend Protocol | `MomentBackend.fit_rows` | `Backend.fit_augmented` |

Default to `ForestRieszRegressor` for ATE/ATT/TSM when you want CIs. Reach for `AugForestRieszRegressor` when (a) you have a custom estimand, (b) you're working with shift-style estimands, or (c) you want one estimator that just works on everything.

## Bregman losses

`AugForestRieszRegressor` supports all four built-in losses: `SquaredLoss` (default), `KLLoss` (density-ratio targets, enforces α̂ > 0 via the exp link), `BernoulliLoss` (α̂ ∈ (0, 1)), and `BoundedSquaredLoss(lo, hi)` (α̂ ∈ (lo, hi)). Tree structure is chosen by the squared MSE criterion; per-leaf θ is then replaced by the Bregman-optimal value via a Newton iteration on each leaf's augmented rows. For locally constant fits this Newton has a closed form (`α* = -B/(2A)` then apply the link); for sieves it's a small p×p Newton.

```python
from forestriesz import AugForestRieszRegressor, TSM
from rieszreg import KLLoss

# Density-ratio fit on TSM. KLLoss requires non-negative m-coefficients
# (TSM, IPSI, and similar density-ratio estimands satisfy this).
est = AugForestRieszRegressor(estimand=TSM(level=1), loss=KLLoss())
est.fit(df)
alpha_hat = est.predict(df)   # all strictly positive by construction
```

`ForestRieszRegressor` (moment-style) is still squared-only — extending it to Bregman losses needs a different per-leaf gradient than the loss API exposes; planned for v3.

## Known sharp edges

- **`predict_interval` is moment-style + single-basis only.** For multi-basis sieves (e.g. ATE's `[1{T=0}, 1{T=1}]`), CIs need a delta-method on θ' φ(x). For `AugForestRieszRegressor`, CIs need cluster-robust variance with `origin_index` as the cluster id (correlated augmented rows). Both planned for v2.
- **`ForestRieszRegressor` is squared-only.** Use `AugForestRieszRegressor` for the other Bregman losses.
- **Honest splits + inference require `n_estimators % subforest_size == 0`** (EconML constraint). Default `subforest_size=4`, so `n_estimators=100, 200, 500, ...` are safe values.
- **`riesz_feature_fns` callables don't auto-save.** Save persists the forest; load needs the callables repassed for custom sieves. Built-in estimands round-trip fine via `riesz_feature_fns="auto"` (the default).
- **R wrapper exposes `ForestRieszRegressor` only.** `AugForestRieszRegressor` would also work from R but isn't wrapped in v1; call from Python via reticulate if needed.
- **Constant basis with `ForestRieszRegressor` is degenerate for built-in estimands.** Forcing `riesz_feature_fns=None` raises a row-constant check error. The default `"auto"` does the right thing. `AugForestRieszRegressor` doesn't have this issue.
- **`KLLoss` and `BernoulliLoss` require non-negative m-coefficients** (density-ratio-style estimands like TSM and IPSI). They reject ATE / ATT / shift-style data at fit time with a clear error.

## On the roadmap

- v3: Bregman losses for the moment-style `ForestRieszRegressor`.
- v2: delta-method `predict_interval` for multi-basis sieves.
- v2: cluster-robust `predict_interval` for `AugForestRieszRegressor`.
- R wrapper for `AugForestRieszRegressor`.

## Living-doc rule

`README.md` is a living document — update it in the same edit whenever a change touches the public API surface. The user guide is the unified Quarto site at [rieszreg.github.io/rieszreg](https://rieszreg.github.io/rieszreg/); the forest-specific page is [forest backend](https://rieszreg.github.io/rieszreg/backends/forest.html).

## License

MIT.
