# krrr

Kernel ridge Riesz regression, in **Python** and **R**. Sister package to [rieszboost](https://github.com/rieszreg/rieszboost) in the [RieszReg family](https://github.com/rieszreg/rieszreg): same scope (Riesz representers for linear estimands; `ψ = E[m(μ)(Z)]`), same user-facing API, but the learner is kernel ridge regression instead of gradient boosting.

Implements [Singh, *Kernel Ridge Riesz Representers* (arXiv:2102.11076)](https://arxiv.org/abs/2102.11076) for the full set of estimands the rieszreg framework supports — not just TSM1 — by piping rieszreg's augmentation engine into a kernel solve. Includes scalable solvers (Nyström-preconditioned conjugate gradient; random Fourier features; optional Falkon for GPU / very large n).

## Status

v0.0.1 — feature-complete for single-stage Riesz regression. **sklearn-compatible** `KernelRieszRegressor` (composes with `GridSearchCV`, `cross_val_predict`, `clone`, `Pipeline`), six built-in estimand factories (re-exported from rieszreg), eight kernels with algebra (`+`, `*`, `Tensor`), four solvers, R6 wrapper. Numerical parity with the [dml-tmle](https://github.com/alejandroschuler/dml-tmle) R reference at 1e-8.

## Why

Singh (2021) gives a closed-form RKHS estimator for the Riesz representer when the moment functional is `m(α) = α(1, x)` (TSM1). For arbitrary linear functionals — ATE, ATT, additive/local/stochastic shifts, custom user-defined `m()` — rieszreg's augmentation engine reduces the problem to the same linear system, just over a richer index set. `krrr` does that reduction and dispatches to a tiered solver.

## Install

`krrr` depends on the [rieszreg](https://github.com/rieszreg/rieszreg) meta-package. Clone both as siblings:

```sh
git clone https://github.com/rieszreg/krrr.git
git clone https://github.com/rieszreg/rieszreg.git
cd krrr
python3 -m venv .venv
.venv/bin/pip install -e ../rieszreg/python
.venv/bin/pip install -e python/
```

(Once both packages publish to PyPI, this collapses to `pip install krrr`.)

Optional Falkon backend for very large n / GPU:

```sh
.venv/bin/pip install 'krrr[falkon]'
```

## Quickstart (Python) — ATE

```python
import numpy as np
import pandas as pd
from krrr import KernelRieszRegressor, ATE, Gaussian

# Synthetic binary-treatment data
rng = np.random.default_rng(0)
n = 1000
x = rng.uniform(0, 1, n)
pi = 1 / (1 + np.exp(-(-0.02*x - x**2 + 4*np.log(x + 0.3) + 1.5)))
a = rng.binomial(1, pi)
df = pd.DataFrame({"a": a.astype(float), "x": x})

# Configuration baked into the regressor (ngboost / sklearn style)
krr = KernelRieszRegressor(
    estimand=ATE(treatment="a", covariates=("x",)),  # owns its input schema
    kernel=Gaussian(length_scale="median"),
    lambda_grid=np.logspace(-4, 0, 25),
    solver="auto",  # picks "direct" for n_aug ≤ 3000, "nystrom_cg" for ≤ 50k, ...
    validation_fraction=0.25,
)

krr.fit(df)
alpha_hat = krr.predict(df)
# alpha_hat ≈ A/π(X) - (1-A)/(1-π(X)) — without ever estimating π(X)

print(f"selected lambda: {krr.lambda_:.4g}")
print(krr.diagnose(df).summary())
```

## Built-in estimands

Re-exported from rieszreg — same API, same semantics:

| Factory | m(z, α) | Notes |
|---|---|---|
| `ATE(treatment, covariates)` | α(1, x) − α(0, x) | Average treatment effect |
| `ATT(treatment, covariates)` | a · (α(1, x) − α(0, x)) | ATT *partial-estimand* surface |
| `TSM(level, treatment, covariates)` | α(level, x) | Treatment-specific mean |
| `AdditiveShift(delta, ...)` | α(a + δ, x) − α(a, x) | Continuous-treatment shift |
| `LocalShift(delta, threshold, ...)` | 1(a < threshold) · (α(a + δ, x) − α(a, x)) | LASE *partial-estimand* surface |
| `StochasticIntervention(samples_key, ...)` | (1/K) Σₖ α(a'ₖ, x) | Stochastic interventions / IPSI |

Custom `m()` works too (write the functional opaquely; `LinearForm` tracing extracts the points and coefficients):

```python
from krrr import KernelRieszRegressor, Estimand

def m_my_thing(alpha):
    def inner(z):
        return 0.7 * alpha(a=1, x=z["x"]) - 0.3 * alpha(a=0, x=z["x"])
    return inner

est = Estimand(feature_keys=("a", "x"), m=m_my_thing, name="MyMix")
krr = KernelRieszRegressor(estimand=est).fit(df)
```

## Kernels

```python
from krrr import Gaussian, Matern, Linear, Polynomial, Tensor

Gaussian(length_scale="median")        # default; median pairwise distance
Gaussian(length_scale="scott")          # Scott's rule
Gaussian(length_scale=0.5)              # fixed
Matern(nu=2.5, length_scale="median")   # Matern with ν ∈ {0.5, 1.5, 2.5}
Linear()                                # k(x, y) = x · y
Polynomial(degree=3, gamma=1.0, coef0=1.0)

# Algebra
Gaussian() + Linear()                   # Sum
0.5 * Gaussian()                        # Scaled
Gaussian() * Linear()                   # Product (Hadamard on Gram)
Tensor(Gaussian(), [0, 1], Linear(), [2])   # tensor product over disjoint columns
```

The kernel is fit on the augmented training points before any Gram evaluation, so `length_scale="median"` resolves to whatever scale matters for *this* dataset.

## Solver selection

| Solver | When to use | Cost |
|---|---|---|
| `"direct"` | n_aug ≤ 3000 | One eigendecomposition per fit; entire λ-path is O(n²) per λ. Exact. |
| `"nystrom_cg"` | n_aug ≤ 50 000 | Preconditioned CG on the symmetric o-block; m landmarks. |
| `"rff"` | n_aug very large; shift-invariant kernel | Primal D × D solve via random Fourier features. |
| `"falkon"` | n_aug very large; GPU available | Wraps the `falkon` package. Optional dependency. |
| `"auto"` | default | Picks `direct` / `nystrom_cg` / `falkon` by n_aug. |

The solver consumes the augmented dataset directly; you never deal with kernel matrices yourself.

## Tuning with `GridSearchCV`

`KernelRieszRegressor` is a `BaseEstimator`, so it composes with `sklearn.model_selection`:

```python
from sklearn.model_selection import GridSearchCV
from krrr import KernelRieszRegressor, ATE, Gaussian, Matern

grid = GridSearchCV(
    KernelRieszRegressor(estimand=ATE()),
    param_grid={"kernel": [Gaussian(), Matern(nu=2.5)],
                "lambda_grid": [np.logspace(-4, -2, 5),
                                np.logspace(-2, 0, 5)]},
    cv=5,
).fit(df)
```

Score is negative held-out Riesz loss — higher is better, as sklearn expects.

## Cross-fitting

```python
from sklearn.model_selection import cross_val_predict

alpha_oof = cross_val_predict(
    KernelRieszRegressor(estimand=ATE()),
    df, cv=5,
)  # OOF predictions, shape (n,)
```

Each fold's regressor does its own internal validation split for λ selection.

## Diagnostics

```python
print(krr.diagnose(df).summary())
```

The base `Diagnostics` (RMS magnitude, |α| quantiles, extreme-row count, held-out Riesz loss) is shared via rieszreg. Use `krrr.diagnose_kernel(krr, df)` for KRR-specific extras: chosen λ, support size, effective degrees of freedom, condition number of the kernel system.

## Save and load

```python
krr.save("my_alpha_hat/")
loaded = KernelRieszRegressor.load("my_alpha_hat/")
np.array_equal(loaded.predict(df), krr.predict(df))  # True
```

Built-in estimands round-trip from metadata. Custom user-defined estimands require `estimand=` on load.

## Quickstart (R)

R6-style wrapper. Install both Python packages into a venv first, then point R at it:

```r
Sys.setenv(RETICULATE_PYTHON = file.path(getwd(), ".venv/bin/python"))
pkgload::load_all("r/krrr")

df <- data.frame(a = ..., x = ...)
krr <- KernelRieszRegressor$new(
  estimand = ATE("a", "x"),
  kernel = Gaussian(length_scale = "median"),
  lambda_grid = 10^seq(-4, 0, length.out = 25),
  solver = "auto",
  validation_fraction = 0.25
)
krr$fit(df)
alpha_hat <- krr$predict(df)
print(krr$diagnose(df)$summary)
```

R-side and Python-side predictions are bitwise-identical on the same data.

## Tests

```sh
.venv/bin/python -m pytest python/tests -v
```

Includes a numerical-parity test against the dml-tmle krrr.R reference at 1e-8.

## On the roadmap

- **`KLLoss` / `BernoulliLoss` / `BoundedSquaredLoss`** — Newton iteration on the kernel system. v0.2.
- **KeOps lazy kernel ops** — for n_aug > 50k where even materializing the Gram matrix on the o-block is heavy.
- **Marginal-likelihood bandwidth selection** — Gaussian-process interpretation; differentiable.
- **Benchmarks at n = 10⁵, 10⁶** — solver tier comparisons; documented speed-vs-accuracy curves.
- **Custom kernel API** — formal `Kernel` Protocol so user-defined kernels slot into the solver registry.

## Related work

- [Singh, Kernel Ridge Riesz Representers (2102.11076)](https://arxiv.org/abs/2102.11076) — closed-form RKHS estimator for TSM1.
- [Lee & Schuler, RieszBoost (2501.04871)](https://arxiv.org/abs/2501.04871) — gradient-boosted Riesz regression; the sister package.
- [Chernozhukov et al., Auto-DML via Riesz Regression (2104.14737)](https://arxiv.org/abs/2104.14737) — origin of the squared Riesz loss.
- [Rudi-Carratino-Rosasco, FALKON (1705.10958)](https://arxiv.org/abs/1705.10958) — Nyström + preconditioned CG solver, optional backend.
- [Rahimi-Recht, Random Features (NIPS 2007)](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) — RFF solver basis.

## License

TBD.
