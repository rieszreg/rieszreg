# rieszboost

Gradient boosting for Riesz representers, in **Python** and **R** — directly estimate the Riesz representer α₀ of a linear functional θ(P) = E[m(Z, g₀)] without ever deriving or inverting a propensity-style ratio. Implements [Lee & Schuler, *RieszBoost* (arXiv:2501.04871)](https://arxiv.org/abs/2501.04871).

The Python package (xgboost-backed fast path + sklearn-compatible slow path) is the source of truth; the R package wraps it with bitwise-identical predictions. Jump to the [R quickstart](#quickstart-r) below.

## Status

v0.0.1 — feature-complete for single-stage Riesz regression. **sklearn-compatible** `RieszBooster` (composes with `GridSearchCV`, `cross_val_predict`, `clone`, `Pipeline`), six built-in estimand factories, two backends (xgboost / sklearn-compatible), pluggable Bregman-loss framework, R6 wrapper. Verified head-to-head against [Lee-Schuler's reference implementation](https://github.com/kaitlynjlee/boosting_for_rr) (see [examples/lee_schuler/COMPARISON.md](examples/lee_schuler/COMPARISON.md)).

Multi-stage longitudinal LMTP is intentionally out of scope — that belongs in a downstream wrapper that calls `RieszBooster(...).fit(...)` once per time-stage.

## Why

In semiparametric estimation (ATE, treatment-specific means, shift interventions, longitudinal interventions) one-step / TMLE / DML estimators require α̂ — the Riesz representer of the target functional. The classical approach derives α₀'s analytical form (e.g. inverse propensity score for ATE), estimates its components, and substitutes them in. That breaks down under positivity violations and gets unwieldy for non-standard estimands. Riesz regression directly minimizes a loss whose minimum is α₀ regardless of analytical form. RieszBoost does this with gradient boosting — fast, tabular-data-friendly, and easy to tune.

## Install

```sh
git clone https://github.com/alejandroschuler/rieszboost.git
cd rieszboost
python3 -m venv .venv
.venv/bin/pip install -e python/
```

On macOS, `xgboost` requires `libomp`:

```sh
brew install libomp
```

## Quickstart (Python) — ATE

```python
import numpy as np
import pandas as pd
import rieszboost
from rieszboost import RieszBooster, ATE

# Synthetic binary-treatment data
rng = np.random.default_rng(0)
n = 4000
x = rng.uniform(0, 1, n)
pi = 1 / (1 + np.exp(-(-0.02*x - x**2 + 4*np.log(x + 0.3) + 1.5)))
a = rng.binomial(1, pi)
df = pd.DataFrame({"a": a.astype(float), "x": x})

# Configuration baked into the booster (ngboost-style)
booster = RieszBooster(
    estimand=ATE(treatment="a", covariates=("x",)),  # owns its input schema
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=4,
    early_stopping_rounds=20,
    validation_fraction=0.2,
)

# Standard sklearn API
booster.fit(df)
alpha_hat = booster.predict(df)
# alpha_hat ≈ A/π(X) - (1-A)/(1-π(X)) — without ever estimating π(X)

# Diagnostics method on the booster
print(booster.diagnose(df).summary())
```

## Tuning with `GridSearchCV`

`RieszBooster` is a `BaseEstimator`, so it composes directly with `sklearn.model_selection`:

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    RieszBooster(estimand=ATE()),
    param_grid={"learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 4, 5]},
    cv=5,
).fit(df)

best = grid.best_estimator_  # already a fitted RieszBooster
alpha_hat = best.predict(df)
```

Score is negative held-out Riesz loss — higher is better, as sklearn expects.

## Cross-fitting for downstream inference

When plugging α̂ into a TMLE / one-step / DML estimator, use cross-fitting so predictions are out-of-fold. `cross_val_predict` does it for any sklearn estimator:

```python
from sklearn.model_selection import cross_val_predict

booster = RieszBooster(
    estimand=ATE(),
    n_estimators=2000, early_stopping_rounds=20, validation_fraction=0.2,
    learning_rate=0.05, max_depth=4,
)
alpha_oof = cross_val_predict(booster, df, cv=5)  # OOF predictions, shape (n,)
```

Each fold's `RieszBooster` does its own internal validation split for early stopping.

## Custom estimands

The natural API is to write `m(z, alpha)` opaquely. The library traces it to extract the linear-form structure. Wrap the result in an `Estimand` and you're done — `feature_keys` / `extra_keys` belong to the estimand:

```python
from rieszboost import Estimand

def m_my_thing(z, alpha):
    return alpha(a=2, x=z["x"]) - 0.5 * alpha(a=1, x=z["x"])

est = Estimand(feature_keys=("a", "x"), m=m_my_thing, name="MyThing")

booster = RieszBooster(estimand=est, n_estimators=200).fit(df)
```

`alpha(...)` calls record evaluation points; `+`, `-`, and scalar `*` compose them into a `LinearForm`. Anything outside that (e.g. `alpha(...) ** 2`, `alpha(...) + 1.0`) raises — by construction the fast path supports exactly the class of finite linear combinations of point evaluations of α.

## Built-in estimands

| Factory | m(z, α) | Notes |
|---|---|---|
| `ATE(treatment, covariates)` | α(1, x) − α(0, x) | Average treatment effect |
| `ATT(treatment, covariates)` | a · (α(1, x) − α(0, x)) | ATT *partial parameter*. Full ATT divides by P(A=1) and is **not** a Riesz functional — combine α̂_partial with a delta-method EIF (Hubbard 2011) downstream. |
| `TSM(level, treatment, covariates)` | α(level, x) | Treatment-specific mean |
| `AdditiveShift(delta, ...)` | α(a + δ, x) − α(a, x) | Continuous-treatment shift effect |
| `LocalShift(delta, threshold, ...)` | 1(a < threshold) · (α(a + δ, x) − α(a, x)) | LASE *partial parameter*; same caveat as ATT |
| `StochasticIntervention(samples_key, ...)` | (1/K) Σₖ α(a'ₖ, x) | Stochastic interventions / IPSI via Monte Carlo over the intervention density |

For stochastic interventions, attach a list-column of pre-sampled treatment values:

```python
df["shift_samples"] = [
    rng.normal(a_i + delta, sigma, size=20).tolist() for a_i in df["a"]
]
booster = RieszBooster(
    estimand=rieszboost.StochasticIntervention(samples_key="shift_samples"),
    n_estimators=500, learning_rate=0.05,
).fit(df)
```

`extra_keys` on the estimand declares the payload columns; `RieszBooster.fit(df)` pulls them through automatically.

## Backends

The default `XGBoostBackend` uses xgboost's custom-objective interface (fast). Swap to `SklearnBackend` to use any sklearn-compatible base learner:

```python
from sklearn.kernel_ridge import KernelRidge
from rieszboost import SklearnBackend

booster = RieszBooster(
    estimand=ATE(),
    backend=SklearnBackend(lambda: KernelRidge(alpha=1.0, kernel="rbf", gamma=2.0)),
    n_estimators=80, learning_rate=0.05,
)
```

`XGBoostBackend(gradient_only=True)` disables the second-order Newton step and runs first-order gradient boosting (Friedman 2001 / Lee-Schuler Algorithm 2 exactly).

## Bregman losses

Default `SquaredLoss` is the standard Lee-Schuler / Chernozhukov objective. `KLLoss` (φ = t log t with exp link) targets density-ratio representers (TSM, IPSI):

```python
from rieszboost import KLLoss
booster = RieszBooster(estimand=rieszboost.TSM(level=1), loss=KLLoss(), ...)
```

Plug in your own by implementing the `LossSpec` protocol. Follows Hines & Miles ([2510.16127](https://arxiv.org/abs/2510.16127)) and Kato ([2601.07752](https://arxiv.org/abs/2601.07752)).

## Examples

[`examples/lee_schuler/`](examples/) reproduces Section 4 of Lee & Schuler (2025): ATE, ATT, ASE, and LASE under their two simulation DGPs, with EEE estimators and coverage. See [examples/README.md](examples/README.md).

## Quickstart (R)

R6-style wrapper. Install Python rieszboost into a venv first, then point R at it:

```r
Sys.setenv(RETICULATE_PYTHON = file.path(getwd(), ".venv/bin/python"))
pkgload::load_all("r/rieszboost")   # or install from r/rieszboost/

df <- data.frame(a = ..., x = ...)
booster <- RieszBooster$new(
  estimand = ATE("a", "x"),
  n_estimators = 2000L,
  early_stopping_rounds = 20L,
  validation_fraction = 0.2,
  learning_rate = 0.05,
  max_depth = 4L
)
booster$fit(df)
alpha_hat <- booster$predict(df)
print(booster$diagnose(df)$summary)
```

R-side and Python-side predictions are bitwise-identical on the same data.

## On the roadmap

Not yet shipped, sized roughly small → large:

- **Serialization** — `RieszBooster.save(path)` / `load(path)` so fitted models survive a session, with the metadata sidecar (loss spec, estimand name, feature_keys, base_score) written alongside the xgboost binary.
- **More example datasets.** Lalonde (ATE under selection), NHEFS (continuous shift), a two-stage longitudinal example demonstrating how to compose `RieszBooster(...).fit(...)` calls across time-stages. The current `examples/` covers Lee-Schuler's synthetic DGPs only.
- **R-side custom `m()`.** The `LinearForm` tracer is Python-only — R users currently use the built-in factories or write Python via reticulate. Porting the tracer to R is non-trivial; alternative is a JSON-spec syntax that round-trips through Python.
- **lightgbm backend.** Same augmentation trick should work via lightgbm's custom objective. Modest speed/memory tradeoffs vs xgboost; deprioritized.
- **Bregman: more built-in losses.** Currently `SquaredLoss` and `KLLoss`. Logistic / clipped-α losses (for representers known to lie in [0, M]) would round out the toolkit.
- **Packaging.** PyPI release for the Python package, CRAN submission for the R wrapper. Pinned-dependency lockfile, CI, etc.

See `CLAUDE.md` and `~/.claude/plans/i-d-like-to-write-crystalline-raven.md` for design notes.

## Tests

```sh
# Python
.venv/bin/python -m pytest python/tests -v

# R (run from repo root)
Rscript -e '
  Sys.setenv(RETICULATE_PYTHON = file.path(getwd(), ".venv/bin/python"))
  pkgload::load_all("r/rieszboost")
  testthat::test_dir("r/rieszboost/tests/testthat")
'
```

## Related work

- [Chernozhukov et al., RieszNet & ForestRiesz (2110.03031)](https://arxiv.org/abs/2110.03031) — neural-net and random-forest Riesz regression.
- [Chernozhukov et al., Auto-DML via Riesz Regression (2104.14737)](https://arxiv.org/abs/2104.14737) — origin of the squared Riesz loss.
- [Singh, Kernel Ridge Riesz Representers (2102.11076)](https://arxiv.org/abs/2102.11076) — closed-form RKHS estimator.
- [Hines & Miles (2510.16127)](https://arxiv.org/abs/2510.16127) and [Kato (2601.07752)](https://arxiv.org/abs/2601.07752) — Bregman-divergence generalization.
- [van der Laan et al. (2501.11868)](https://arxiv.org/abs/2501.11868) — auto-DML for smooth functionals beyond linear.

## License

TBD.
