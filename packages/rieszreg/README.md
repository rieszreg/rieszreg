# rieszreg

Shared core of the RieszReg family. It holds the estimands, the losses, and the sklearn-compatible `RieszEstimator` that every learner package builds on.

Most users install a learner package and import everything from it. Each one re-exports the names below, so `from rieszboost import ATE` and `from rieszreg import ATE` give the same object.

| Package | Learner class | Method |
|---|---|---|
| `rieszboost` | `RieszBooster` | Gradient boosting |
| `krrr` | `KernelRieszRegressor` | Kernel ridge regression |
| `forestriesz` | `AugForestRieszRegressor`, `ForestRieszRegressor` | Random forest |
| `riesztree` | `RieszTreeRegressor` | A single decision tree |
| `riesznet` | `RieszNet` | Neural network (PyTorch) |

## Key terms

- **Estimand.** The number you want to estimate, such as the average treatment effect (ATE). Pick one with `ATE`, `ATT`, `TSM`, `AdditiveShift`, or `LocalShift`.
- **Riesz representer** $\alpha$. A weight for each row. For the ATE it equals the inverse-propensity weight $A/\pi(X) - (1-A)/(1-\pi(X))$. Averaging $\alpha$ times the outcome residual corrects an outcome model's bias. These packages estimate $\alpha$ directly, without fitting a propensity model first.
- **Learner.** The algorithm that fits $\hat\alpha$: boosting, kernel ridge, forest, tree, or neural network.
- **Loss.** What the learner minimizes. `SquaredLoss()` is the default. `KLLoss()` keeps $\hat\alpha$ positive, which suits estimands like `TSM`.

The [user guide](https://rieszreg.github.io/rieszreg/intro.html) explains the ideas step by step.

## Quickstart

The estimator follows the sklearn pattern: construct, `fit`, `predict`.

```python
import numpy as np
import pandas as pd
from rieszboost import ATE, RieszBooster

rng = np.random.default_rng(0)
n = 2000
age = rng.normal(50, 10, n)
income = rng.normal(0, 1, n)
treated = rng.binomial(1, 1 / (1 + np.exp(-(age - 50) / 10))).astype(float)
outcome = 2.0 * treated + 0.1 * age + income + rng.normal(0, 1, n)

# Z holds the treatment column and the covariates. Keep the outcome out of Z.
Z = pd.DataFrame({"treated": treated, "age": age, "income": income})

booster = RieszBooster(estimand=ATE(treatment="treated"))
booster.fit(Z)
alpha_hat = booster.predict(Z)       # one weight per row
print(booster.diagnose(Z).summary())  # flags extreme weights (poor overlap)
```

`ATE(treatment="treated")` names the treatment column. Every other column of `Z` is a covariate. To use only some columns, pass `covariates=["age", "income"]`. With a NumPy array instead of a DataFrame, column 0 is the treatment.

The built-in estimands do not use the outcome, so `fit(Z)` needs no `y`. Passing `fit(Z, y)` is also fine.

## Cross-fitting and tuning

Every learner is an sklearn estimator. Use the usual sklearn tools:

```python
from sklearn.model_selection import GridSearchCV, cross_val_predict

alpha_cf = cross_val_predict(booster, Z, cv=5)                       # cross-fit
search = GridSearchCV(booster, {"max_depth": [2, 4]}, cv=3).fit(Z)   # tune
```

`score(Z)` returns the negative held-out squared Riesz loss (higher is better), so `GridSearchCV` works with no extra arguments.

## From weights to an estimate

Combine the cross-fit $\hat\alpha$ with any outcome model $\hat\mu$ to get a one-step (DML) estimate and a 95% interval:

```python
from sklearn.ensemble import GradientBoostingRegressor

mu = GradientBoostingRegressor()
mu_hat = cross_val_predict(mu, Z, outcome, cv=5)
mu.fit(Z, outcome)
mu1 = mu.predict(Z.assign(treated=1.0))
mu0 = mu.predict(Z.assign(treated=0.0))

psi_i = mu1 - mu0 + alpha_cf * (outcome - mu_hat)
ate, se = psi_i.mean(), psi_i.std() / np.sqrt(n)
print(f"ATE = {ate:.2f} ± {1.96 * se:.2f}")   # ATE = 2.02 ± 0.10 (truth: 2.0)
```

This short version does not cross-fit `mu1` and `mu0`. The [custom-code estimation page](https://rieszreg.github.io/rieszreg/estimation/custom.html) shows the fully cross-fit version, where each fold's outcome model predicts `mu1` and `mu0` on its held-out rows, plus TMLE.

## Using `RieszEstimator` directly

`RieszEstimator` pairs any estimand with any backend object:

```python
from rieszreg import ATE, RieszEstimator
from rieszboost import XGBoostBackend

est = RieszEstimator(estimand=ATE(treatment="treated"), backend=XGBoostBackend())
```

The learner classes above are thin subclasses that build the backend for you.
