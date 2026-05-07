# Cross-check vs the Lee-Schuler reference implementation

The reference is [`kaitlynjlee/boosting_for_rr`](https://github.com/kaitlynjlee/boosting_for_rr). It contains a sklearn-`DecisionTreeRegressor`-based gradient-boosting fitter implementing exactly Algorithm 2 from Lee & Schuler (2025). This document records a head-to-head comparison against `rieszboost` to verify our augmentation, gradient, and loss formulation are mathematically equivalent.

## Setup

```sh
git clone https://github.com/kaitlynjlee/boosting_for_rr /tmp/lee_ref
PYTHONPATH=/tmp/lee_ref .venv/bin/python examples/lee_schuler/_compare_with_reference.py \
    --n 500 --n_seeds 10 --lr 0.1 --n_estimators 100 --max_depth 3
```

The script generates one dataset, fits both models on it (no early stopping, no subsampling), and reports per-row α̂ disagreement.

| Side | What it does |
|---|---|
| Reference | Lee-Schuler `ATE_ES_stochastic` / `ATT_ES_stochastic` with `early_stopping_rounds=10**9` (effectively off), `sample_prop=1.0`. Each iteration: fit a sklearn `DecisionTreeRegressor` on negative gradients, update F by `learning_rate * predictions`. |
| Ours | `rieszboost.fit(gradient_only=True, learning_rate=lr_ref/2, reg_lambda=0)`. Uses xgboost as the tree backend with `hess=ones_like(grad)` (first-order step) and per-row gradient `2aF + b`. |

## Results

10 seeds, n=500, lr=0.1, n_estimators=100, max_depth=3:

| Estimand | Pearson(ref, ours) | RMSE(ref vs ours) | RMSE(ref vs truth) | RMSE(ours vs truth) |
|---|---|---|---|---|
| ATE | **0.998** | 0.13 | 1.02 | 0.99 |
| ATT | **0.986** | 0.19 | 0.76 | 0.73 |

Disagreement (`RMSE(ref vs ours)`) is an order of magnitude smaller than disagreement of either implementation with the truth. That's "same algorithm, different tree backend" territory — what's left is split-finding differences between sklearn `DecisionTreeRegressor` (exhaustive scan) and xgboost (histogram-based with second-order surrogate even when we send hess=1).

## Why the `learning_rate / 2` matters

Walk through one update at a counterfactual point in the augmented data.

For the ATE Riesz loss, the per-subject contribution is

```
α(z_i)² − 2 α(1, x_i) + 2 α(0, x_i)     # full-strength Riesz loss
```

Lee-Schuler's reference rewrites this as a sum over augmented rows, where each row carries half the linear coefficient:

```
0.5 α² − (2a − 1) α        # original row  (D=1)
(1 − 2a) α                  # counterfactual row  (D=0)
```

Total per-subject contribution: half the natural Riesz loss. Same minimizer (multiplying a loss by ½ doesn't change `argmin`). Their per-row residual is `f − (2a−1)` at original rows, `1 − 2a` at counterfactuals — exactly half the gradient of the natural loss.

`rieszboost` uses the natural Riesz loss directly: per-augmented-row loss is `a F² + b F` with gradient `2aF + b`. So our gradient is twice theirs.

xgboost with `gradient_only=True` and `reg_lambda=0` sets each leaf to `−mean(gradient)` over the leaf rows. With our 2×-larger gradient, each step is 2× theirs. To match dynamics under the same `learning_rate`, halve ours: `learning_rate=lr_ref/2`.

## Bug spotted in the reference

`ATE_ES_stochastic.fit_internal` (the no-early-stopping path) reshapes `A` to `(n, 1)` and then indexes `np.zeros(n)[A == 1]`. After the reshape, `A == 1` is shape `(n, 1)`, and indexing a 1D array with a 2D mask raises `IndexError`. Only `fit_internal_early_stopping` works; the comparison script sets `early_stopping_rounds=10**9` so it never triggers and the working path is exercised.

(PR upstream is in flight.)

## Conclusion

The augmentation, gradient, and loss in `rieszboost` reproduce Lee-Schuler's reference implementation up to numerical differences from the tree backend. With `gradient_only=True` and `learning_rate=lr_ref/2`, you can use `rieszboost` as a drop-in replacement.

For most users, leaving `gradient_only=False` (default) is preferred: xgboost's second-order step with our `hessian_floor=2.0` actually fits α₀ slightly better at matched hyperparameters. The cross-check is here so future engine changes don't silently drift away from the reference.
