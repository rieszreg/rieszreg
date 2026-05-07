"""Stochastic intervention / IPSI: estimate E[Y under A' ~ g(·|A, X)] for a
soft, distribution-valued intervention. Here the intervention is an additive
Gaussian perturbation: A' = A + δ + ε with ε ~ N(0, σ²).

The Riesz functional is m(z, μ) = ∫ μ(a', x) g(a'|A, X) da'. We approximate
the integral by Monte Carlo: each row carries K pre-sampled shift values
under `df["shift_samples"]`, and `StochasticIntervention(samples_key=…)` makes
m a uniform finite mixture over them. The fast augmentation path applies as
usual.

DGP — same continuous-treatment setup as Lee-Schuler 4.2, with an additive
Gaussian intervention on top:
    X        ~ Uniform(0, 2)
    A | X    ~ N(X² − 1, σ_A² = 2)
    Y | A,X  ~ N(5X + 9A(X+2)² + 5 sin(πX) + 25A, 1)
    A' | A,X ~ A + δ + N(0, σ_shift²)        with δ=1, σ_shift=0.5

Run:
    .venv/bin/python examples/stochastic_intervention.py --n_reps 50
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from rieszboost import RieszBooster, StochasticIntervention


SHIFT = 1.0
SIGMA_SHIFT = 0.5
N_MC_SAMPLES = 20


def expected_outcome(a, x):
    return 5 * x + 9 * a * (x + 2) ** 2 + 5 * np.sin(x * np.pi) + 25 * a


def simulate(n: int, rng: np.random.Generator):
    x = rng.uniform(0, 2, n)
    a = rng.normal(x**2 - 1, np.sqrt(2.0))
    mu = expected_outcome(a, x)
    y = mu + rng.normal(0, 1, n)
    return x, a, y


def true_psi(n_mc: int = 1_000_000, seed: int = 7) -> float:
    """Brute-force ground truth: ψ = E_X E_A E_ε [μ(A + δ + ε, X)].

    Integrand is linear in A so the inner expectation collapses analytically:
      A | X ~ N(X²-1, σ_A²=2),  ε ~ N(0, σ_shift²)  ⇒
      A' | X ~ N(X²-1+δ, σ_A² + σ_shift²)
    μ(a', x) is linear in a', so E[μ(A', X) | X=x] = μ(E[A'|X], x).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2, n_mc)
    mean_a_prime = x**2 - 1 + SHIFT
    return float(np.mean(expected_outcome(mean_a_prime, x)))


def fit_outcome_regression(a, x, y) -> xgb.Booster:
    return xgb.train(
        {"objective": "reg:squarederror", "learning_rate": 0.05, "max_depth": 4,
         "reg_lambda": 1.0, "seed": 0, "verbosity": 0},
        xgb.DMatrix(np.column_stack([a, x]), label=y),
        num_boost_round=200,
    )


def predict_mu(mu_hat, a, x):
    return mu_hat.predict(xgb.DMatrix(np.column_stack([a, x])))


def attach_shift_samples(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    a_arr = df["a"].to_numpy()
    df["shift_samples"] = [
        rng.normal(a_i + SHIFT, SIGMA_SHIFT, N_MC_SAMPLES).tolist()
        for a_i in a_arr
    ]
    return df


def fit_alpha(df_train: pd.DataFrame) -> RieszBooster:
    return RieszBooster(
        estimand=StochasticIntervention(
            samples_key="shift_samples", treatment="a", covariates=("x",)
        ),
        n_estimators=2000,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        learning_rate=0.05,
        max_depth=3,
        reg_lambda=1.0,
        random_state=0,
    ).fit(df_train)


def eee_stochastic(df, mu_hat, alpha_hat):
    """ψ̂ = (1/n) Σ [(1/K) Σ_k μ̂(a'_k, X) + α̂(O)·(Y − μ̂(O))]."""
    a = df["a"].to_numpy()
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    samples = df["shift_samples"].to_numpy()

    mu_obs = predict_mu(mu_hat, a, x)
    # Mean μ̂ over the K shift samples per row.
    K = len(samples[0])
    mu_shift_mean = np.zeros(len(df))
    for j in range(K):
        a_prime = np.array([s[j] for s in samples])
        mu_shift_mean += predict_mu(mu_hat, a_prime, x)
    mu_shift_mean /= K

    eif = mu_shift_mean + alpha_hat * (y - mu_obs)
    psi = float(eif.mean())
    se = float(np.sqrt(np.var(eif, ddof=1) / len(eif)))
    return psi, se


def run_one_rep(rng, n: int = 1000, train_frac: float = 0.5):
    n_tr = int(n * train_frac)
    x, a, y = simulate(n, rng)
    df = pd.DataFrame({"a": a, "x": x, "y": y})
    df = attach_shift_samples(df, rng)

    df_tr = df.iloc[:n_tr].reset_index(drop=True)
    df_es = df.iloc[n_tr:].reset_index(drop=True)

    mu_hat = fit_outcome_regression(
        df_tr["a"].to_numpy(), df_tr["x"].to_numpy(), df_tr["y"].to_numpy()
    )
    # Fit α̂ on training rows (drop y; the booster doesn't see it).
    booster = fit_alpha(df_tr.drop(columns=["y"]))
    alpha_hat = booster.predict(df_es.drop(columns=["y"]))

    psi, se = eee_stochastic(df_es, mu_hat, alpha_hat)
    return {"estimate": psi, "se": se}


def summarize(reps, psi_true: float):
    ests = np.array([r["estimate"] for r in reps])
    ses = np.array([r["se"] for r in reps])
    cov = float(np.mean((ests - 1.96 * ses < psi_true) & (psi_true < ests + 1.96 * ses)))

    print("\n=== StochasticIntervention (additive Gaussian shift) results ===")
    print(f"  intervention   : A' = A + {SHIFT} + N(0, {SIGMA_SHIFT}²),  K={N_MC_SAMPLES} MC samples")
    print(f"  truth (brute)  : {psi_true:.4f}")
    print(f"  mean estimate  : {ests.mean():.4f}  (bias {ests.mean() - psi_true:+.4f})")
    print(f"  avg SE         : {ses.mean():.4f}")
    print(f"  empirical SD   : {ests.std(ddof=1):.4f}")
    print(f"  RMSE           : {float(np.sqrt(np.mean((ests - psi_true) ** 2))):.4f}")
    print(f"  95% coverage   : {cov:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    psi_truth = true_psi()
    print(f"# Ground truth via 1e6-sample integration: ψ = {psi_truth:.4f}")

    rng_master = np.random.default_rng(args.seed)
    reps = []
    t0 = time.time()
    for r in range(args.n_reps):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        reps.append(run_one_rep(rng, n=args.n))
        if (r + 1) % max(1, args.n_reps // 10) == 0 or r == args.n_reps - 1:
            print(f"  rep {r + 1:>3}/{args.n_reps}  ({time.time() - t0:.1f}s)")

    summarize(reps, psi_truth)


if __name__ == "__main__":
    main()
