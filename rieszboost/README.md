# rieszboost

Gradient boosting for Riesz representers, in **Python** and **R** — directly estimate the Riesz representer α₀ of a linear functional θ(P) = E[m(Z, g₀)] without ever deriving or inverting a propensity-style ratio. Implements [Lee & Schuler, *RieszBoost* (arXiv:2501.04871)](https://arxiv.org/abs/2501.04871).

The Python package (xgboost-backed fast path + sklearn-compatible slow path) is the source of truth; the R package wraps it via `reticulate` with bitwise-identical predictions. Jump to the [R quickstart](#quickstart-r) below.

## Status

v0.0.1 — feature-complete for single-stage Riesz regression. Python fast (xgboost) and slow (sklearn-compatible) paths, R wrapper via reticulate, six built-in estimand factories, cross-fitting, early stopping, diagnostics, and a pluggable Bregman-loss framework all ship. Verified head-to-head against [Lee-Schuler's reference implementation](https://github.com/kaitlynjlee/boosting_for_rr) (see [examples/lee_schuler/COMPARISON.md](examples/lee_schuler/COMPARISON.md)).

Multi-stage longitudinal LMTP is intentionally out of scope — that belongs in a downstream wrapper that calls the single-stage `rieszboost.fit(...)` once per time-stage.

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
import rieszboost

# Synthetic binary-treatment data
rng = np.random.default_rng(0)
n = 4000
x = rng.uniform(0, 1, n)
pi = 1 / (1 + np.exp(-(-0.02*x - x**2 + 4*np.log(x + 0.3) + 1.5)))
a = rng.binomial(1, pi)
rows = [{"a": int(ai), "x": float(xi)} for ai, xi in zip(a, x)]

# 80/20 train/valid split for early stopping
n_tr = int(0.8 * n)
train_rows, valid_rows = rows[:n_tr], rows[n_tr:]

booster = rieszboost.fit(
    train_rows,
    rieszboost.ATE(),               # m(z, alpha) = alpha(1, x) - alpha(0, x)
    feature_keys=("a", "x"),
    valid_rows=valid_rows,
    num_boost_round=2000,
    early_stopping_rounds=20,       # halt when held-out Riesz loss stops improving
    learning_rate=0.05,
    max_depth=4,
)

alpha_hat = booster.predict(rows)
# alpha_hat ≈ A/π(X) - (1-A)/(1-π(X)) — without ever estimating π(X)

# Diagnostics — magnitude, tail extremes, near-positivity warnings
print(rieszboost.diagnose(booster=booster, rows=valid_rows, m=rieszboost.ATE()).summary())
```

## Examples

[`examples/lee_schuler/`](examples/) reproduces Section 4 of Lee & Schuler (2025): ATE, ATT, ASE, and LASE under their two simulation DGPs, with EEE estimators and coverage. See [examples/README.md](examples/README.md).

## Cross-fitting for downstream inference

When you'll plug α̂ into a TMLE / one-step / DML estimator, use cross-fitting so the predictions you use are out-of-fold:

```python
result = rieszboost.crossfit(
    rows,
    rieszboost.ATE(),
    feature_keys=("a", "x"),
    n_folds=5,
    early_stopping_inner_split=0.2,  # carve a held-out slice inside each fold
    num_boost_round=2000,
    early_stopping_rounds=20,
    learning_rate=0.05,
    max_depth=3,
    reg_lambda=10.0,                 # keep extrapolation tame
)
alpha_hat_oof = result.alpha_hat   # shape (n,) — out-of-fold predictions
```

> **Note on hyperparameters.** Boosted Riesz representers can extrapolate aggressively at low-overlap boundaries. Shallower trees (`max_depth=3`) and a heavier ridge (`reg_lambda=10`) plus early stopping keep the tails under control. Always run `rieszboost.diagnose(...)` on the fit and inspect the warnings.

## Custom estimands

The natural API is to write `m(z, alpha)` opaquely. The library traces it to extract the linear-form structure:

```python
def m_att(z, alpha):
    """ATT representer: averages over the treated marginal."""
    p_treated = 0.4   # marginal P(A=1), estimated externally
    return (alpha(a=1, x=z["x"]) - alpha(a=0, x=z["x"])) * (z["a"] / p_treated)
```

`alpha(...)` calls record evaluation points; `+`, `-`, and scalar `*` compose them into a `LinearForm`. Anything outside that (e.g. `alpha(...) ** 2`, `alpha(...) + 1.0`) raises — by construction the fast path supports exactly the class of finite linear combinations of point evaluations of α.

## Built-in estimands

| Factory | m(z, α) | Notes |
|---|---|---|
| `rieszboost.ATE(treatment, covariates)` | α(1, x) − α(0, x) | Average treatment effect |
| `rieszboost.ATT(treatment, covariates)` | a · (α(1, x) − α(0, x)) | ATT *partial parameter* `E[A·(μ(1,X)−μ(0,X))]`. The full ATT divides by P(A=1) and is **not** itself a Riesz functional — combine α̂_partial with a delta-method EIF (Hubbard 2011) downstream. |
| `rieszboost.TSM(level, treatment, covariates)` | α(level, x) | Treatment-specific mean |
| `rieszboost.AdditiveShift(delta, treatment, covariates)` | α(a + δ, x) − α(a, x) | Continuous-treatment shift effect |
| `rieszboost.LocalShift(delta, threshold, ...)` | 1(a < threshold) · (α(a + δ, x) − α(a, x)) | LASE *partial parameter*. Like ATT, the full LASE divides by P(A < threshold) and needs a delta-method EIF. |
| `rieszboost.StochasticIntervention(samples_key, ...)` | (1/K) Σₖ α(a'ₖ, x) | Stochastic interventions / IPSI via Monte Carlo over the intervention density |

For stochastic interventions, pre-sample treatment values from g(·\|a, x) per row and attach them under `samples_key`:

```python
rng = np.random.default_rng(0)
for row in rows:
    row["shift_samples"] = rng.normal(row["a"] + delta, sigma, size=20)

booster = rieszboost.fit(rows, rieszboost.StochasticIntervention(),
                         feature_keys=("a", "x"), ...)
```

Full LMTP-style longitudinal interventions with time-varying confounding require multi-stage orchestration (a separate Riesz fit per time-stage); the single-stage `rieszboost.fit(...)` API is the right upstream for an LMTP wrapper to call repeatedly.

## What works today

- Opaque `m(z, alpha)` API with linearity enforced by construction.
- Fast path: data augmentation + xgboost custom objective. Pass `gradient_only=True` to disable the second-order Newton step and use first-order gradient boosting (Friedman 2001 / Lee-Schuler Algorithm 2 exactly).
- Slow general path: first-order gradient boosting (Friedman 2001) on the augmented dataset with any sklearn-compatible base learner — `rieszboost.general_fit(..., base_learner=lambda: KernelRidge(...))`.
- **Bregman-Riesz losses** via `loss_spec=`: `SquaredLoss()` (default — the standard Lee-Schuler / Chernozhukov objective) and `KLLoss()` (φ = t log t with exp link, for density-ratio targets like TSM / IPSI). Plug in your own by implementing the `LossSpec` protocol. Follows Hines & Miles ([2510.16127](https://arxiv.org/abs/2510.16127)) and Kato ([2601.07752](https://arxiv.org/abs/2601.07752)).
- ATE / ATT / TSM / AdditiveShift / LocalShift / StochasticIntervention estimand factories. ATT and LocalShift fit *partial-parameter* representers; the full ATT and LASE require a downstream delta-method correction (see `examples/lee_schuler/binary_dgp.py` and `continuous_dgp.py`).
- R wrapper via reticulate — bitwise-identical predictions across languages.
- `init={float, "m1"}` initialization (in α space; loss spec handles the link transform).
- Early stopping on held-out Riesz loss (`valid_rows=` + `early_stopping_rounds=`).
- K-fold cross-fitting (`rieszboost.crossfit(...)`) with optional inner-split early stopping.
- Diagnostics (`rieszboost.diagnose(...)`): RMS, extremes, |α| quantiles, near-positivity warnings, held-out Riesz loss.

## On the roadmap

Not yet shipped, sized roughly small → large:

- **sklearn-compatible wrapper for `GridSearchCV`.** `rieszboost.fit()` takes a list of row-dicts and an opaque `m()` callable, so vanilla `GridSearchCV` doesn't accept it directly. The right fix is to expose a thin `BaseEstimator` subclass (with `.fit(X) / .predict(X) / .score(X)` returning negative held-out Riesz loss) and a tuning-loop recipe in `examples/`, not a bespoke `tune_riesz()` helper. Lee-Schuler's reference is sklearn-shaped end-to-end and gets tuning from `GridSearchCV` for free; we should match that.
- **Serialization** — `RieszBooster.save(path)` / `load(path)` so fitted models survive a session, with the metadata sidecar (loss spec, feature_keys, base_score) written alongside the xgboost binary.
- **More example datasets.** Lalonde (ATE under selection), NHEFS (continuous shift), a two-stage longitudinal example demonstrating how to compose `rieszboost.fit(...)` calls across time-stages. The current `examples/` covers Lee-Schuler's synthetic DGPs only.
- **R-side custom `m()`.** The `LinearForm` tracer is Python-only — R users currently use the built-in factories or write Python via reticulate. Porting the tracer to R is non-trivial; alternative is a JSON-spec syntax (`m: list(coef=c(1,-1), points=...)`) that round-trips through Python.
- **lightgbm engine adapter.** Same data-augmentation trick should work via lightgbm's custom objective. Modest speed/memory tradeoffs vs xgboost; deprioritized.
- **Bregman: more built-in losses.** Currently `SquaredLoss` and `KLLoss`. Logistic / clipped-α losses (e.g., for representers known to lie in [0, M]) would round out the toolkit.
- **Packaging.** PyPI release for the Python package, CRAN submission for the R wrapper. Pinned-dependency lockfile, CI, etc.

See `CLAUDE.md` and `~/.claude/plans/i-d-like-to-write-crystalline-raven.md` for design notes and the original plan.

## Related work

- [Chernozhukov et al., RieszNet & ForestRiesz (2110.03031)](https://arxiv.org/abs/2110.03031) — neural-net and random-forest Riesz regression.
- [Chernozhukov et al., Auto-DML via Riesz Regression (2104.14737)](https://arxiv.org/abs/2104.14737) — origin of the squared Riesz loss.
- [Singh, Kernel Ridge Riesz Representers (2102.11076)](https://arxiv.org/abs/2102.11076) — closed-form RKHS estimator.
- [Hines & Miles (2510.16127)](https://arxiv.org/abs/2510.16127) and [Kato (2601.07752)](https://arxiv.org/abs/2601.07752) — Bregman-divergence generalization.
- [van der Laan et al. (2501.11868)](https://arxiv.org/abs/2501.11868) — auto-DML for smooth functionals beyond linear.

## Quickstart (R)

Same library, callable from R via reticulate. Install Python rieszboost into a venv first, then point R at it:

```r
# from the repo root
Sys.setenv(RETICULATE_PYTHON = file.path(getwd(), ".venv/bin/python"))
pkgload::load_all("r/rieszboost")   # or install from r/rieszboost/

df <- data.frame(a = ..., x = ...)
n_tr <- floor(0.8 * nrow(df))

fit <- fit_riesz(
  data = df[1:n_tr, ],
  m = ATE(treatment = "a", covariates = "x"),
  feature_keys = c("a", "x"),
  valid_data = df[(n_tr + 1):nrow(df), ],
  num_boost_round = 2000L,
  early_stopping_rounds = 20L,
  learning_rate = 0.05,
  max_depth = 3L,
  reg_lambda = 10
)
alpha_hat <- predict(fit, df)

# Cross-fitting and diagnostics work the same way:
res <- crossfit(df, ATE("a", "x"), c("a", "x"), n_folds = 5L,
                early_stopping_inner_split = 0.2,
                num_boost_round = 1000L, early_stopping_rounds = 20L,
                learning_rate = 0.05, max_depth = 3L, reg_lambda = 10)
print(diagnose_alpha(booster = fit, data = df, m = ATE("a", "x")))
```

R-side and Python-side predictions are bitwise-identical on the same data.

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

## License

TBD.
