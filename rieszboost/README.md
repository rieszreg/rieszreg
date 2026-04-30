# rieszboost

Gradient boosting for Riesz representers, in **Python** and **R** — directly estimate the Riesz representer α of a linear estimand ψ = E[m(μ)(Z)] without ever deriving or inverting a propensity-style ratio. Implements [Lee & Schuler, *RieszBoost* (arXiv:2501.04871)](https://arxiv.org/abs/2501.04871).

The default backend wraps **xgboost's custom-objective interface**, so the inner loop is the same C++ histogram-based, sparsity-aware, GPU-capable booster the world's fastest tabular learners are built on — no Python-level boosting code in the hot path. Drop in `SklearnBackend(...)` to swap in any sklearn-compatible base learner (`KernelRidge`, MLPs, etc.) when you need a smooth learner; the xgboost path stays the default precisely because there's nothing faster for tabular Riesz regression. The R package wraps the Python core with bitwise-identical predictions. Jump to the [R quickstart](#quickstart-r) below.

📖 **[User guide](https://rieszreg.github.io/rieszreg/)** — narrative reference with executable R + Python examples on every page.

## Status

v0.0.1 — feature-complete for single-stage Riesz regression. **sklearn-compatible** `RieszBooster` (composes with `GridSearchCV`, `cross_val_predict`, `clone`, `Pipeline`), six built-in estimand factories, two backends (xgboost / sklearn-compatible), pluggable Bregman-loss framework, R6 wrapper. Verified head-to-head against [Lee-Schuler's reference implementation](https://github.com/kaitlynjlee/boosting_for_rr) (see [examples/lee_schuler/COMPARISON.md](examples/lee_schuler/COMPARISON.md)).

Multi-stage longitudinal LMTP is intentionally out of scope — that belongs in a downstream wrapper that calls `RieszBooster(...).fit(...)` once per time-stage.

## Why

In semiparametric estimation (ATE, treatment-specific means, shift interventions, longitudinal interventions) one-step / TMLE / DML estimators require α̂ — the Riesz representer of the estimand. The classical approach derives α₀'s analytical form (e.g. inverse propensity score for ATE), estimates its components, and substitutes them in. That breaks down under positivity violations and gets unwieldy for non-standard estimands. Riesz regression directly minimizes a loss whose minimum is α₀ regardless of analytical form. RieszBoost does this with gradient boosting — fast, tabular-data-friendly, and easy to tune.

## Install

```sh
git clone https://github.com/rieszreg/rieszboost.git
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

Write `m(alpha)` as an operator that returns a function of `z`. The library traces it to extract the linear-form structure. Wrap the result in an `Estimand` and you're done — `feature_keys` / `extra_keys` belong to the estimand:

```python
from rieszboost import Estimand

def m_my_thing(alpha):
    def inner(z):
        return alpha(a=2, x=z["x"]) - 0.5 * alpha(a=1, x=z["x"])
    return inner

est = Estimand(feature_keys=("a", "x"), m=m_my_thing, name="MyThing")

booster = RieszBooster(estimand=est, n_estimators=200).fit(df)
```

`alpha(...)` calls record evaluation points; `+`, `-`, and scalar `*` compose them into a `LinearForm`. Anything outside that (e.g. `alpha(...) ** 2`, `alpha(...) + 1.0`) raises — by construction the fast path supports exactly the class of finite linear combinations of point evaluations of α.

## Built-in estimands

| Factory | m(α)(z) | Notes |
|---|---|---|
| `ATE(treatment, covariates)` | α(1, x) − α(0, x) | Average treatment effect |
| `ATT(treatment, covariates)` | a · (α(1, x) − α(0, x)) | ATT *partial-estimand* surface. Full ATT divides by P(A=1) and is **not** a Riesz functional — combine α̂_partial with a delta-method EIF (Hubbard 2011) downstream. |
| `TSM(level, treatment, covariates)` | α(level, x) | Treatment-specific mean |
| `AdditiveShift(delta, ...)` | α(a + δ, x) − α(a, x) | Continuous-treatment shift effect |
| `LocalShift(delta, threshold, ...)` | 1(a < threshold) · (α(a + δ, x) − α(a, x)) | LASE *partial-estimand* surface; same caveat as ATT |
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

## Save and load

Two paths, depending on what you need.

**sklearn idiom** — the standard way to persist any `BaseEstimator`:

```python
import joblib
joblib.dump(booster, "alpha.pkl")
loaded = joblib.load("alpha.pkl")
np.array_equal(loaded.predict(df), booster.predict(df))  # True
```

This single-file pickle works with all six built-in estimands, all four losses, and all the sklearn machinery (`clone`, `GridSearchCV`, `cross_val_predict`) post-load.

**Portable directory format** — a folder with the xgboost model in its native binary plus a JSON metadata sidecar. Use this when you need to introspect the metadata, share the model with non-Python tooling, or load it in R:

```python
booster.save("my_alpha_hat/")
loaded = RieszBooster.load("my_alpha_hat/")
```

R-side `RieszBooster$save(path)` and `load_riesz_booster(path)` use the same directory format, so a model saved in one language loads bitwise-identically in the other.

Both formats handle built-in estimands automatically; custom user-defined `Estimand`s with non-importable `m()` callables need `cloudpickle` for the joblib path or `estimand=...` re-passed to `RieszBooster.load(...)` for the directory path.

## Backends

The default `XGBoostBackend` uses xgboost's custom-objective interface (fast). Swap to `SklearnBackend` to use any sklearn-compatible base learner. When you supply `backend=` explicitly, boost-loop knobs (`n_estimators`, `learning_rate`, `early_stopping_rounds`, `validation_fraction`) live on the backend itself — `RieszBooster`'s matching ctor args only apply to the default-XGBoost path.

```python
from sklearn.kernel_ridge import KernelRidge
from rieszboost import SklearnBackend

booster = RieszBooster(
    estimand=ATE(),
    backend=SklearnBackend(
        base_learner_factory=lambda: KernelRidge(alpha=1.0, kernel="rbf", gamma=2.0),
        n_estimators=80,
        learning_rate=0.05,
    ),
)
```

`XGBoostBackend(gradient_only=True)` disables the second-order Newton step and runs first-order gradient boosting (Friedman 2001 / Lee-Schuler Algorithm 2 exactly).

## Bregman losses

Four built-in losses cover the common cases:

| Loss | Domain | Use when |
|---|---|---|
| `SquaredLoss` | ℝ | Default — standard Lee-Schuler / Chernozhukov objective. |
| `KLLoss` | (0, ∞) | Density-ratio representers (TSM, IPSI). Forces α̂ > 0. |
| `BernoulliLoss` | (0, 1) | α₀ known to be a probability by problem structure. |
| `BoundedSquaredLoss(lo, hi)` | (lo, hi) | Squared loss with hard prior bounds (e.g. trimmed propensities). |

Plug in your own by implementing the `LossSpec` protocol. The framework follows Hines & Miles ([2510.16127](https://arxiv.org/abs/2510.16127)) and Kato ([2601.07752](https://arxiv.org/abs/2601.07752)).

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

The active list is empty — all original v0.0.1 items have shipped:

- Six built-in estimand factories with worked examples (synthetic + real-data: Lalonde ATE, NHEFS shift).
- Two backends, four losses, full sklearn integration, R6 wrapper, save / load (cross-language).
- Six-layer test suite (smoke + regression baselines + backend equivalence + edge cases + property + sklearn conformance + perf bench). See [docs/TESTING_PLAN.md](docs/TESTING_PLAN.md).

Future work would focus on CI wiring, real-data examples beyond the two shipped, and additional Bregman losses (e.g. Itakura-Saito for audio-style ratios) as use cases emerge.

Considered and dropped:

- ~~lightgbm backend~~ — xgboost already provides histogram-based splits, GPU support, sparsity-aware finding, and the second-order Newton step. lightgbm's main differentiators (leaf-wise growth, native categorical handling) don't help our augmented dataset, which is mostly continuous and not categorical-heavy. No compelling speed or quality win.
- ~~R-side custom `m()`~~ — porting the `LinearForm` tracer to R is more trouble than it's worth. R users get the full set of built-in estimands; if a user needs a brand-new functional, they write the m() in Python and the R booster reads through reticulate.
- ~~PyPI / CRAN packaging~~ — premature pre-1.0. Pin when the API has stabilized.

See `CLAUDE.md` for design notes and the API design rule.

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
