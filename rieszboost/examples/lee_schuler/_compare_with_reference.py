"""Head-to-head comparison vs Lee & Schuler's reference implementation
(github.com/kaitlynjlee/boosting_for_rr).

Goal: with gradient_only=True and matched hyperparameters, our engine should
produce α̂ predictions equivalent to their reference fitter on identical data.
This is a bug-check, not a benchmark.

Notes on factor-of-2:
  Their per-row gradient (residual) for the squared Riesz loss is half ours.
  Their loss is `0.5 f² - (2a-1) f` for original rows and `(1-2a) f` for
  counterfactuals. Ours is `f² - 2(2a-1) f` and `2(1-2a) f` (a factor of 2
  on each term, no effect on the minimizer). To match their dynamics under
  the same learning_rate, we use `learning_rate / 2` in our engine.

Run from repo root:
  PYTHONPATH=/tmp/kaitlyn_boosting_<dir>  .venv/bin/python \
      examples/lee_schuler/_compare_with_reference.py
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

# Reference implementation
from rrboost.boosters import ATE_ES_stochastic, ATT_ES_stochastic
from rrboost.dgps import ATE as ate_dgp

# Our implementation
import rieszboost


# ATT partial-parameter m is now in rieszboost.ATT()


def compare_ate(seed: int, n: int, lr_ref: float, n_estimators: int, max_depth: int):
    """Generate one dataset and fit ATE Riesz both ways. Return per-row α̂."""
    Y, A, X = ate_dgp.gen_data(n=n, seed=seed)

    # --- Reference: Lee-Schuler ATE_ES_stochastic with early stopping disabled
    # via huge patience, sample_prop=1.0 (no subsampling), tiny validation split.
    # (Their fit_internal has a shape bug — only fit_internal_early_stopping works.)
    ref = ATE_ES_stochastic(
        learning_rate=lr_ref,
        n_estimators=n_estimators,
        max_depth=max_depth,
        early_stopping=True,
        early_stopping_rounds=10**9,   # never triggers
        validation_fraction=0.01,       # smallest possible
        sample_prop=1.0,
        random_state=0,
    )
    ref.fit(np.column_stack([A, X]))
    alpha_ref = ref.predict(np.column_stack([A, X]))

    # --- Ours: gradient_only=True, lr_ref/2 to match their gradient scale ---
    import pandas as pd
    df = pd.DataFrame({"a": A.astype(float), "x": X[:, 0]})
    booster = rieszboost.RieszBooster(
        estimand=rieszboost.ATE(treatment="a", covariates=("x",)),
        backend=rieszboost.XGBoostBackend(gradient_only=True),
        learning_rate=lr_ref / 2.0,
        n_estimators=n_estimators,
        max_depth=max_depth,
        reg_lambda=0.0,
        random_state=0,
        init=0.0,
    ).fit(df)
    alpha_ours = booster.predict(df)

    # Truth
    alpha_truth = ate_dgp.riesz_rep(A, X)

    return alpha_ref, alpha_ours, alpha_truth


def summarize(name, ref, ours, truth):
    rmse_ref = float(np.sqrt(np.mean((ref - truth) ** 2)))
    rmse_ours = float(np.sqrt(np.mean((ours - truth) ** 2)))
    rmse_diff = float(np.sqrt(np.mean((ref - ours) ** 2)))
    corr = float(np.corrcoef(ref, ours)[0, 1])
    print(f"\n=== {name} ===")
    print(f"  alpha_truth  : min={truth.min():.3f}  max={truth.max():.3f}")
    print(f"  alpha_ref    : min={ref.min():.3f}  max={ref.max():.3f}  RMSE_truth={rmse_ref:.3f}")
    print(f"  alpha_ours   : min={ours.min():.3f}  max={ours.max():.3f}  RMSE_truth={rmse_ours:.3f}")
    print(f"  ref vs ours  : RMSE={rmse_diff:.3f}  Pearson_corr={corr:.4f}")


def compare_att(seed, n, lr_ref, n_estimators, max_depth):
    """Same comparison for ATT (partial-parameter form)."""
    Y, A, X = ate_dgp.gen_data(n=n, seed=seed)

    ref = ATT_ES_stochastic(
        learning_rate=lr_ref,
        n_estimators=n_estimators,
        max_depth=max_depth,
        early_stopping=True,
        early_stopping_rounds=10**9,
        validation_fraction=0.01,
        sample_prop=1.0,
        random_state=0,
    )
    ref.fit(np.column_stack([A, X]))
    alpha_ref = ref.predict(np.column_stack([A, X]))

    import pandas as pd
    df = pd.DataFrame({"a": A.astype(float), "x": X[:, 0]})
    booster = rieszboost.RieszBooster(
        estimand=rieszboost.ATT(),
        backend=rieszboost.XGBoostBackend(gradient_only=True),
        learning_rate=lr_ref / 2.0,
        n_estimators=n_estimators,
        max_depth=max_depth,
        reg_lambda=0.0,
        random_state=0,
        init=0.0,
    ).fit(df)
    alpha_ours = booster.predict(df)

    # Truth: ATT partial representer alpha_0(A, X) = A - (1-A) pi(X)/(1-pi(X))
    pi = ate_dgp.expected_trt(X)
    alpha_truth = A - (1 - A) * pi / (1 - pi)
    return alpha_ref, alpha_ours, alpha_truth


def run_block(name, fn, args):
    print(f"\n## {name} comparison ##")
    print(f"# n={args.n}, lr_ref={args.lr}, n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth}, n_seeds={args.n_seeds}")
    rmse_ref, rmse_ours, rmse_diff, corrs = [], [], [], []
    for seed in range(args.n_seeds):
        ref, ours, truth = fn(
            seed=seed + 411, n=args.n, lr_ref=args.lr,
            n_estimators=args.n_estimators, max_depth=args.max_depth,
        )
        rmse_ref.append(float(np.sqrt(np.mean((ref - truth) ** 2))))
        rmse_ours.append(float(np.sqrt(np.mean((ours - truth) ** 2))))
        rmse_diff.append(float(np.sqrt(np.mean((ref - ours) ** 2))))
        corrs.append(float(np.corrcoef(ref, ours)[0, 1]))
    print(f"  RMSE(ref vs truth)  mean = {np.mean(rmse_ref):.3f}  sd = {np.std(rmse_ref):.3f}")
    print(f"  RMSE(ours vs truth) mean = {np.mean(rmse_ours):.3f}  sd = {np.std(rmse_ours):.3f}")
    print(f"  RMSE(ref vs ours)   mean = {np.mean(rmse_diff):.3f}  sd = {np.std(rmse_diff):.3f}")
    print(f"  Pearson(ref, ours)  mean = {np.mean(corrs):.4f}  min = {np.min(corrs):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    args = parser.parse_args()

    print(f"# Reference: Lee-Schuler [ATE|ATT]_ES_stochastic (early_stopping huge so it never triggers, sample_prop=1.0)")
    print(f"# Ours:      rieszboost.fit(gradient_only=True, learning_rate=lr_ref/2, reg_lambda=0)")
    run_block("ATE", compare_ate, args)
    run_block("ATT", compare_att, args)


if __name__ == "__main__":
    main()
