"""Reproduces Lee & Schuler (arXiv:2501.04871) Section 4.2: ASE and LASE
under the continuous-treatment DGP.

DGP:
    X ~ Uniform(0, 2)
    A | X ~ Normal(X^2 - 1, sigma^2 = 2)
    Y | A, X ~ Normal(5X + 9A(X+2)^2 + 5 sin(X*pi) + 25A, 1)

Shift intervention: A' = A + 1.
Estimands:
  ASE  = E[mu(A+1, X) - mu(A, X)]                   = 108.997
  LASE = E[mu(A+1, X) - mu(A, X) | A < 0]            =  94.814

Closed-form Riesz representers (under the conditional Gaussian density):
  alpha_ASE_0(a, x)  = exp((2(a - x^2) + 1) / 4) - 1
  alpha_LASE_0(a, x) = 1(a < 1) exp((2(a - x^2) + 1) / 4) - 1(a < 0)
                      (this is the *partial* representer; final estimator
                       divides by P(A < 0))

Run:
    .venv/bin/python examples/lee_schuler/continuous_dgp.py --n_reps 50
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import rieszboost


SHIFT = 1.0
THRESHOLD = 0.0  # LASE: shift only for A < THRESHOLD


def simulate(n: int, rng: np.random.Generator):
    x = rng.uniform(0.0, 2.0, n)
    a = rng.normal(x**2 - 1.0, np.sqrt(2.0))
    mu = 5 * x + 9 * a * (x + 2) ** 2 + 5 * np.sin(x * np.pi) + 25 * a
    y = mu + rng.normal(0, 1.0, n)
    return x, a, y, mu


def density_ratio(a, x, delta=SHIFT):
    """p_{A|X}(a-delta, x) / p_{A|X}(a, x) under A|X ~ N(x^2 - 1, 2)."""
    return np.exp((2.0 * (a - x**2) + 1.0) / 4.0)


def alpha_truth_ase(a, x):
    return density_ratio(a, x) - 1.0


def alpha_truth_lase_partial(a, x, t=THRESHOLD, delta=SHIFT):
    """Partial-LASE representer (without the 1/P(A<t) scale)."""
    return (a < (t + delta)).astype(float) * density_ratio(a, x, delta) - (a < t).astype(float)


def fit_outcome_regression(a, x, y):
    X = np.column_stack([a, x])
    return xgb.train(
        {
            "objective": "reg:squarederror",
            "learning_rate": 0.05,
            "max_depth": 4,
            "reg_lambda": 1.0,
            "seed": 0,
            "verbosity": 0,
        },
        xgb.DMatrix(X, label=y),
        num_boost_round=200,
    )


def predict_mu(mu_hat, a, x):
    return mu_hat.predict(xgb.DMatrix(np.column_stack([a, x])))


# m_ase comes from rieszboost.AdditiveShift; m_lase_partial from
# rieszboost.LocalShift (LASE itself isn't a Riesz functional — the partial-
# parameter form here gets a delta-method correction in eee_lase).
m_ase = rieszboost.AdditiveShift(delta=SHIFT, treatment="a", covariates=("x",))
m_lase_partial = rieszboost.LocalShift(
    delta=SHIFT, threshold=THRESHOLD, treatment="a", covariates=("x",)
)


_RIESZ_PARAMS = dict(
    n_estimators=3000,
    early_stopping_rounds=20,
    validation_fraction=0.2,
    learning_rate=0.01,
    max_depth=3,
    reg_lambda=1.0,
    random_state=0,
)


def _df(a, x):
    return pd.DataFrame({"a": a.astype(float), "x": x.astype(float)})


def fit_alpha(df_train, estimand):
    return rieszboost.RieszBooster(estimand=estimand, **_RIESZ_PARAMS).fit(df_train)


def eee_ase(a, x, y, mu_hat, alpha_hat):
    mu_obs = predict_mu(mu_hat, a, x)
    mu_shift = predict_mu(mu_hat, a + SHIFT, x)
    eif = (mu_shift - mu_obs) + alpha_hat * (y - mu_obs)
    psi = float(eif.mean())
    se = float(np.sqrt(np.var(eif, ddof=1) / len(eif)))
    return psi, se


def eee_lase(a, x, y, mu_hat, alpha_hat_partial, t=THRESHOLD):
    """LASE = E[mu(A+1,X) - mu(A,X) | A<t]; uses delta method on 1/P(A<t)."""
    p_t = float(np.mean(a < t))
    mu_obs = predict_mu(mu_hat, a, x)
    mu_shift = predict_mu(mu_hat, a + SHIFT, x)
    indicator = (a < t).astype(float)
    psi_partial = float(np.mean(indicator * (mu_shift - mu_obs) + alpha_hat_partial * (y - mu_obs)))
    psi = psi_partial / p_t
    eif = (1.0 / p_t) * (
        indicator * (mu_shift - mu_obs - psi) + alpha_hat_partial * (y - mu_obs)
    )
    se = float(np.sqrt(np.var(eif, ddof=1) / len(eif)))
    return psi, se


def run_one_rep(rng, n=1000, train_frac=0.5):
    n_tr = int(n * train_frac)
    x, a, y, _ = simulate(n, rng)

    a_tr, x_tr, y_tr = a[:n_tr], x[:n_tr], y[:n_tr]
    a_es, x_es, y_es = a[n_tr:], x[n_tr:], y[n_tr:]
    df_tr = _df(a_tr, x_tr)
    df_es = _df(a_es, x_es)

    mu_hat = fit_outcome_regression(a_tr, x_tr, y_tr)

    booster_ase = fit_alpha(df_tr, m_ase)
    booster_lase = fit_alpha(df_tr, m_lase_partial)

    alpha_hat_ase = booster_ase.predict(df_es)
    alpha_hat_lase = booster_lase.predict(df_es)

    alpha_true_ase = alpha_truth_ase(a_es, x_es)
    alpha_true_lase = alpha_truth_lase_partial(a_es, x_es)

    psi_ase, se_ase = eee_ase(a_es, x_es, y_es, mu_hat, alpha_hat_ase)
    psi_lase, se_lase = eee_lase(a_es, x_es, y_es, mu_hat, alpha_hat_lase)

    return {
        "ase_estimate": psi_ase,
        "ase_se": se_ase,
        "lase_estimate": psi_lase,
        "lase_se": se_lase,
        "ase_alpha_rmse": float(np.sqrt(np.mean((alpha_hat_ase - alpha_true_ase) ** 2))),
        "ase_alpha_mae": float(np.mean(np.abs(alpha_hat_ase - alpha_true_ase))),
        "lase_alpha_rmse": float(np.sqrt(np.mean((alpha_hat_lase - alpha_true_lase) ** 2))),
        "lase_alpha_mae": float(np.mean(np.abs(alpha_hat_lase - alpha_true_lase))),
    }


def summarize(reps, psi_true_ase, psi_true_lase):
    arr = {k: np.array([r[k] for r in reps]) for k in reps[0]}

    def block(name, est, se, truth):
        ests = arr[est]
        ses = arr[se]
        cov = float(np.mean((ests - 1.96 * ses < truth) & (truth < ests + 1.96 * ses)))
        return (
            name,
            float(ests.mean()),
            float(ses.mean()),
            float(np.sqrt(np.mean((ests - truth) ** 2))),
            float(ests.std(ddof=1)),
            cov,
        )

    rows = [
        block("ASE", "ase_estimate", "ase_se", psi_true_ase),
        block("LASE", "lase_estimate", "lase_se", psi_true_lase),
    ]

    print("\n=== Final-parameter results (vs Lee-Schuler Tables 5 & 6) ===")
    print(f"{'Estimand':>8} {'mean':>9} {'avg.SE':>8} {'RMSE':>8} {'emp.SD':>8} {'cov95':>7}")
    for name, mean, se, rmse, sd, cov in rows:
        print(f"{name:>8} {mean:>9.3f} {se:>8.3f} {rmse:>8.3f} {sd:>8.3f} {cov:>7.3f}")
    print("Lee-Schuler Table 5 ASE:  109.672    2.087    2.796    2.713    0.934")
    print("Lee-Schuler Table 6 LASE:  94.921    1.768    1.859    1.855    0.946")

    print("\n=== Riesz representer estimation (vs Lee-Schuler Table 4) ===")
    print(f"{'Estimand':>8} {'RMSE':>8} {'MAE':>8}")
    print(f"{'ASE':>8} {arr['ase_alpha_rmse'].mean():>8.3f} {arr['ase_alpha_mae'].mean():>8.3f}")
    print(f"{'LASE':>8} {arr['lase_alpha_rmse'].mean():>8.3f} {arr['lase_alpha_mae'].mean():>8.3f}")
    print("Lee-Schuler Table 4 ASE:    0.366    0.230")
    print("Lee-Schuler Table 4 LASE:   0.252    0.154")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    psi_true_ase = 108.997
    psi_true_lase = 94.814

    rng_master = np.random.default_rng(args.seed)
    reps = []
    t0 = time.time()
    for r in range(args.n_reps):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        result = run_one_rep(rng, n=args.n)
        reps.append(result)
        if (r + 1) % max(1, args.n_reps // 10) == 0 or r == args.n_reps - 1:
            print(f"  rep {r + 1:>3}/{args.n_reps}  ({time.time() - t0:.1f}s elapsed)")

    summarize(reps, psi_true_ase, psi_true_lase)


if __name__ == "__main__":
    main()
