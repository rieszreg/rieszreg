"""Reproduces Lee & Schuler (arXiv:2501.04871) Section 4.1: ATE and ATT
under the binary-treatment DGP. Reports Riesz-representer RMSE/MAE and EEE
estimates with coverage.

DGP:
    X ~ Uniform(0, 1)
    A | X ~ Binomial(1, logit(-0.02 X - X^2 + 4 log(X + 0.3) + 1.5))
    Y | A, X ~ Normal(5X + 9XA + 5 sin(X*pi) + 25(A - 2), 1)

True parameters: psi_ATE = 29.502, psi_ATT = 30.786.

Run:
    .venv/bin/python examples/lee_schuler/binary_dgp.py --n_reps 50

The paper uses 500 reps with n=1000 (500 train + 500 estimation). Defaults here
are n_reps=50 to keep wall time short; pass --n_reps 500 to match the paper.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import rieszboost


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate(n: int, rng: np.random.Generator):
    x = rng.uniform(0.0, 1.0, n)
    pi = _logit(-0.02 * x - x**2 + 4.0 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi).astype(np.float64)
    mu = 5 * x + 9 * x * a + 5 * np.sin(x * np.pi) + 25 * (a - 2)
    y = mu + rng.normal(0, 1.0, n)
    return x, a, y, pi, mu


def fit_outcome_regression(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> xgb.Booster:
    """xgboost regressor for mu_hat(a, x). Standard squared-error objective."""
    X = np.column_stack([a, x])
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(
        {
            "objective": "reg:squarederror",
            "learning_rate": 0.05,
            "max_depth": 4,
            "reg_lambda": 1.0,
            "seed": 0,
            "verbosity": 0,
        },
        dtrain,
        num_boost_round=200,
    )


def predict_mu(mu_hat: xgb.Booster, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    return mu_hat.predict(xgb.DMatrix(np.column_stack([a, x])))


# Lee-Schuler tune hyperparameters via CV; we use moderate fixed defaults.
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


def fit_alpha_ate(df_train):
    return rieszboost.RieszBooster(
        estimand=rieszboost.ATE(treatment="a", covariates=("x",)),
        **_RIESZ_PARAMS,
    ).fit(df_train)


def fit_alpha_att(df_train):
    """Fits the Riesz representer of the ATT *partial parameter*
    `theta_partial = E[A*(mu(1,X) - mu(0,X))]`. The full ATT is
    `theta_partial / P(A=1)` and is handled by `eee_att` via delta method."""
    return rieszboost.RieszBooster(
        estimand=rieszboost.ATT(treatment="a", covariates=("x",)),
        **_RIESZ_PARAMS,
    ).fit(df_train)


def eee_ate(a, x, y, mu_hat, alpha_hat):
    """EEE estimator for ATE: 1/n sum [mu(1,x) - mu(0,x) + alpha(O)(Y - mu(O))]."""
    mu_obs = predict_mu(mu_hat, a, x)
    mu1 = predict_mu(mu_hat, np.ones_like(a), x)
    mu0 = predict_mu(mu_hat, np.zeros_like(a), x)
    eif = (mu1 - mu0) + alpha_hat * (y - mu_obs)
    psi = float(eif.mean())
    se = float(np.sqrt(np.var(eif, ddof=1) / len(eif)))
    return psi, se


def eee_att(a, x, y, mu_hat, alpha_hat_partial):
    """EEE estimator for ATT using the partial-parameter Riesz representer.
    Following Lee-Schuler appendix A.1 (Hubbard 2011): the EIF involves a delta
    method on 1/P(A=1). We use the empirical mean of A as P̂(A=1)."""
    p_a = float(a.mean())
    mu_obs = predict_mu(mu_hat, a, x)
    mu1 = predict_mu(mu_hat, np.ones_like(a), x)
    mu0 = predict_mu(mu_hat, np.zeros_like(a), x)
    psi_partial = float(np.mean(a * (mu1 - mu0) + alpha_hat_partial * (y - mu_obs)))
    psi = psi_partial / p_a
    eif = (1.0 / p_a) * (
        a * (mu1 - mu0 - psi) + alpha_hat_partial * (y - mu_obs)
    )
    se = float(np.sqrt(np.var(eif, ddof=1) / len(eif)))
    return psi, se


def alpha_truth_ate(a, pi):
    return a / pi - (1 - a) / (1 - pi)


def alpha_truth_att_partial(a, pi):
    return a - (1 - a) * pi / (1 - pi)


def run_one_rep(rng, n=1000, train_frac=0.5):
    """One simulation rep: train/estimation split; on training, fit mu_hat
    and alpha_hat (with an inner 80/20 split for early stopping); compute EEE
    estimates and Riesz RMSE on the estimation half."""
    n_tr = int(n * train_frac)
    x, a, y, pi, _ = simulate(n, rng)

    x_tr, a_tr, y_tr = x[:n_tr], a[:n_tr], y[:n_tr]
    x_es, a_es, y_es, pi_es = x[n_tr:], a[n_tr:], y[n_tr:], pi[n_tr:]

    df_tr = _df(a_tr, x_tr)
    df_es = _df(a_es, x_es)

    mu_hat = fit_outcome_regression(a_tr, x_tr, y_tr)

    booster_ate = fit_alpha_ate(df_tr)
    booster_att = fit_alpha_att(df_tr)

    alpha_hat_ate = booster_ate.predict(df_es)
    alpha_hat_att = booster_att.predict(df_es)

    alpha_true_ate = alpha_truth_ate(a_es, pi_es)
    alpha_true_att = alpha_truth_att_partial(a_es, pi_es)

    psi_ate, se_ate = eee_ate(a_es, x_es, y_es, mu_hat, alpha_hat_ate)
    psi_att, se_att = eee_att(a_es, x_es, y_es, mu_hat, alpha_hat_att)

    return {
        "ate_estimate": psi_ate,
        "ate_se": se_ate,
        "att_estimate": psi_att,
        "att_se": se_att,
        "ate_alpha_rmse": float(np.sqrt(np.mean((alpha_hat_ate - alpha_true_ate) ** 2))),
        "ate_alpha_mae": float(np.mean(np.abs(alpha_hat_ate - alpha_true_ate))),
        "att_alpha_rmse": float(np.sqrt(np.mean((alpha_hat_att - alpha_true_att) ** 2))),
        "att_alpha_mae": float(np.mean(np.abs(alpha_hat_att - alpha_true_att))),
    }


def summarize(reps: list[dict], psi_true_ate: float, psi_true_att: float):
    arr = {k: np.array([r[k] for r in reps]) for k in reps[0]}
    rows = []

    def block(name, est_key, se_key, truth):
        ests = arr[est_key]
        ses = arr[se_key]
        cov = float(np.mean((ests - 1.96 * ses < truth) & (truth < ests + 1.96 * ses)))
        rows.append(
            (
                name,
                float(ests.mean()),
                float(ses.mean()),
                float(np.sqrt(np.mean((ests - truth) ** 2))),
                float(ests.std(ddof=1)),
                cov,
            )
        )

    block("ATE", "ate_estimate", "ate_se", psi_true_ate)
    block("ATT", "att_estimate", "att_se", psi_true_att)

    print("\n=== Final-parameter results (vs Lee-Schuler Tables 2 & 3) ===")
    print(f"{'Estimand':>8} {'mean':>9} {'avg.SE':>8} {'RMSE':>8} {'emp.SD':>8} {'cov95':>7}")
    for name, mean, se, rmse, sd, cov in rows:
        print(f"{name:>8} {mean:>9.3f} {se:>8.3f} {rmse:>8.3f} {sd:>8.3f} {cov:>7.3f}")
    print("Lee-Schuler Table 2 ATE:  29.522    0.175    0.187    0.186    0.940")
    print("Lee-Schuler Table 3 ATT:  30.786    0.173    0.177    0.177    0.950")

    print("\n=== Riesz representer estimation (vs Lee-Schuler Table 1) ===")
    print(f"{'Estimand':>8} {'RMSE':>8} {'MAE':>8}")
    print(f"{'ATE':>8} {arr['ate_alpha_rmse'].mean():>8.3f} {arr['ate_alpha_mae'].mean():>8.3f}")
    print(f"{'ATT':>8} {arr['att_alpha_rmse'].mean():>8.3f} {arr['att_alpha_mae'].mean():>8.3f}")
    print("Lee-Schuler Table 1 ATE:    0.920    0.347")
    print("Lee-Schuler Table 1 ATT:    0.435    0.185")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    psi_true_ate = 29.502
    psi_true_att = 30.786

    rng_master = np.random.default_rng(args.seed)
    reps = []
    t0 = time.time()
    for r in range(args.n_reps):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        result = run_one_rep(rng, n=args.n)
        reps.append(result)
        if (r + 1) % max(1, args.n_reps // 10) == 0 or r == args.n_reps - 1:
            elapsed = time.time() - t0
            print(f"  rep {r + 1:>3}/{args.n_reps}  ({elapsed:.1f}s elapsed)")

    summarize(reps, psi_true_ate, psi_true_att)


if __name__ == "__main__":
    main()
