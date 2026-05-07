"""TSM (treatment-specific mean): estimate E[Y(a*)] for a fixed treatment level a*.

The Riesz representer of m(z, μ) = μ(a*, x) is α₀(A, X) = 1(A=a*) / π(a*|X).
We fit α̂ via rieszboost and plug it into the EEE / one-step estimator
    ψ̂ = (1/n) Σ [μ̂(a*, X) + α̂(O)·(Y − μ̂(O))]

DGP: same binary-treatment setup as Lee-Schuler 4.1. We target a* = 1.
True ψ_TSM = E[μ(1, X)]; with this DGP and X ~ Uniform(0,1),
    μ(1, X) = 5X + 9X + 5 sin(πX) + 25(1−2) = 14X + 5 sin(πX) − 25
hence ψ_TSM_true = 14·0.5 + 5·(2/π) − 25 ≈ −14.82.

Run:
    .venv/bin/python examples/tsm.py --n_reps 50
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from rieszboost import RieszBooster, TSM


def simulate(n: int, rng: np.random.Generator):
    x = rng.uniform(0.0, 1.0, n)
    pi = 1.0 / (1.0 + np.exp(-(-0.02 * x - x**2 + 4.0 * np.log(x + 0.3) + 1.5)))
    a = rng.binomial(1, pi).astype(np.float64)
    mu = 5 * x + 9 * x * a + 5 * np.sin(x * np.pi) + 25 * (a - 2)
    y = mu + rng.normal(0, 1, n)
    return x, a, y, pi


PSI_TRUE = 14 * 0.5 + 5 * (2 / np.pi) - 25  # ≈ −14.82


def fit_outcome_regression(a, x, y) -> xgb.Booster:
    return xgb.train(
        {"objective": "reg:squarederror", "learning_rate": 0.05,
         "max_depth": 4, "reg_lambda": 1.0, "seed": 0, "verbosity": 0},
        xgb.DMatrix(np.column_stack([a, x]), label=y),
        num_boost_round=200,
    )


def predict_mu(mu_hat, a, x):
    return mu_hat.predict(xgb.DMatrix(np.column_stack([a, x])))


def fit_alpha(df_train: pd.DataFrame, level: float) -> RieszBooster:
    return RieszBooster(
        estimand=TSM(level=level, treatment="a", covariates=("x",)),
        n_estimators=2000,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        learning_rate=0.05,
        max_depth=3,
        reg_lambda=1.0,
        random_state=0,
    ).fit(df_train)


def eee_tsm(a, x, y, mu_hat, alpha_hat, level: float):
    """ψ̂ = (1/n) Σ [μ̂(level, X) + α̂(O) (Y − μ̂(O))]."""
    mu_obs = predict_mu(mu_hat, a, x)
    mu_at_level = predict_mu(mu_hat, np.full_like(a, level), x)
    eif = mu_at_level + alpha_hat * (y - mu_obs)
    psi = float(eif.mean())
    se = float(np.sqrt(np.var(eif, ddof=1) / len(eif)))
    return psi, se


def alpha_truth_tsm(a, pi, level: float):
    """For TSM(level=a*), α₀(A, X) = 1(A=a*) / π(a*|X)."""
    indicator = (a == level).astype(float)
    p_at_level = pi if level == 1 else (1 - pi)
    return indicator / p_at_level


def run_one_rep(rng, n=1000, train_frac=0.5, level: float = 1.0):
    n_tr = int(n * train_frac)
    x, a, y, pi = simulate(n, rng)
    a_tr, x_tr, y_tr = a[:n_tr], x[:n_tr], y[:n_tr]
    a_es, x_es, y_es, pi_es = a[n_tr:], x[n_tr:], y[n_tr:], pi[n_tr:]
    df_tr = pd.DataFrame({"a": a_tr, "x": x_tr})
    df_es = pd.DataFrame({"a": a_es, "x": x_es})

    mu_hat = fit_outcome_regression(a_tr, x_tr, y_tr)
    booster = fit_alpha(df_tr, level=level)

    alpha_hat = booster.predict(df_es)
    alpha_true = alpha_truth_tsm(a_es, pi_es, level=level)

    psi, se = eee_tsm(a_es, x_es, y_es, mu_hat, alpha_hat, level=level)
    return {
        "estimate": psi,
        "se": se,
        "alpha_rmse": float(np.sqrt(np.mean((alpha_hat - alpha_true) ** 2))),
        "alpha_mae": float(np.mean(np.abs(alpha_hat - alpha_true))),
    }


def summarize(reps, psi_true: float):
    arr = {k: np.array([r[k] for r in reps]) for k in reps[0]}
    ests = arr["estimate"]
    ses = arr["se"]
    cov = float(np.mean((ests - 1.96 * ses < psi_true) & (psi_true < ests + 1.96 * ses)))

    print("\n=== TSM(level=1) results ===")
    print(f"  truth          : {psi_true:.4f}")
    print(f"  mean estimate  : {ests.mean():.4f}  (bias {ests.mean() - psi_true:+.4f})")
    print(f"  avg SE         : {ses.mean():.4f}")
    print(f"  empirical SD   : {ests.std(ddof=1):.4f}")
    print(f"  RMSE           : {float(np.sqrt(np.mean((ests - psi_true) ** 2))):.4f}")
    print(f"  95% coverage   : {cov:.3f}")
    print(f"  α-RMSE         : {arr['alpha_rmse'].mean():.3f}")
    print(f"  α-MAE          : {arr['alpha_mae'].mean():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=float, default=1.0)
    args = parser.parse_args()

    rng_master = np.random.default_rng(args.seed)
    reps = []
    t0 = time.time()
    for r in range(args.n_reps):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        reps.append(run_one_rep(rng, n=args.n, level=args.level))
        if (r + 1) % max(1, args.n_reps // 10) == 0 or r == args.n_reps - 1:
            print(f"  rep {r + 1:>3}/{args.n_reps}  ({time.time() - t0:.1f}s)")

    summarize(reps, PSI_TRUE)


if __name__ == "__main__":
    main()
