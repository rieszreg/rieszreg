"""Lalonde NSW (with CPS controls): ATE of a job-training program on 1978
post-program earnings.

This is the classic Dehejia-Wahba / Lalonde dataset for ATE under selection.
The point of the example is not to relitigate the Lalonde controversy — it's
to show a real-data, real-confounding workflow end to end:

  1. Load the data.
  2. Cross-fit μ̂ (outcome regression for `re78`) and α̂ (Riesz representer
     for ATE) — both with `RieszBooster` / xgboost using `cross_val_predict`,
     so all nuisance predictions used in the EEE plug-in are out-of-fold.
  3. Compute the EEE / one-step ATE estimator and a Wald CI.

Comparing NSW-treated vs CPS-controls makes the point estimate dependent on
how well covariates account for the gap between the experimental treated
group and the very different observational comparison group.

Run:
    .venv/bin/python examples/lalonde_ate.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict

from rieszboost import ATE, RieszBooster

try:
    from causaldata import cps_mixtape, nsw_mixtape
except ImportError as e:
    raise SystemExit(
        "This example needs `pip install causaldata`. Run "
        "`.venv/bin/pip install causaldata` and retry."
    ) from e


# Standard Dehejia-Wahba covariates.
COVARIATES = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]


def load_data() -> pd.DataFrame:
    """NSW-treated subjects + CPS-1 controls."""
    treated = nsw_mixtape.load_pandas().data
    treated = treated[treated["treat"] == 1].copy()
    controls = cps_mixtape.load_pandas().data
    df = pd.concat([treated, controls], ignore_index=True)

    # Rename treatment to `a` (rieszboost's default) and drop unused cols.
    df = df.rename(columns={"treat": "a"})
    df["y"] = df["re78"]
    df = df[["a", "y"] + COVARIATES].copy()
    df["a"] = df["a"].astype(float)
    return df


def fit_outcome_regression_oof(df: pd.DataFrame, n_folds: int = 5) -> np.ndarray:
    """OOF μ̂(A, X) via xgboost regression."""
    X = df[["a"] + COVARIATES].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return cross_val_predict(
        xgb.XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.05,
            max_depth=4,
            reg_lambda=1.0,
            n_estimators=300,
            random_state=0,
            verbosity=0,
        ),
        X, y,
        cv=KFold(n_splits=n_folds, shuffle=True, random_state=0),
    )


def fit_outcome_regression_full(df: pd.DataFrame) -> xgb.Booster:
    """Full-data μ̂ — used to predict at counterfactual treatment levels."""
    X = df[["a"] + COVARIATES].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return xgb.train(
        {"objective": "reg:squarederror", "learning_rate": 0.05,
         "max_depth": 4, "reg_lambda": 1.0, "seed": 0, "verbosity": 0},
        xgb.DMatrix(X, label=y),
        num_boost_round=300,
    )


def predict_mu(mu_hat: xgb.Booster, df: pd.DataFrame, a_value: float) -> np.ndarray:
    a = np.full(len(df), float(a_value))
    X = np.column_stack([a, df[COVARIATES].to_numpy(dtype=float)])
    return mu_hat.predict(xgb.DMatrix(X))


def fit_alpha_oof(df: pd.DataFrame, n_folds: int = 5) -> np.ndarray:
    """Out-of-fold Riesz representer α̂ via cross_val_predict."""
    booster = RieszBooster(
        estimand=ATE(treatment="a", covariates=tuple(COVARIATES)),
        n_estimators=2000,
        early_stopping_rounds=20,
        validation_fraction=0.2,
        learning_rate=0.05,
        max_depth=3,
        reg_lambda=10.0,
        random_state=0,
    )
    return cross_val_predict(
        booster,
        df[["a"] + COVARIATES],
        cv=KFold(n_splits=n_folds, shuffle=True, random_state=0),
    )


def eee_ate(df: pd.DataFrame, mu_hat: xgb.Booster, mu_hat_oof: np.ndarray, alpha_hat: np.ndarray):
    """ψ̂ = (1/n) Σ [μ̂(1, X) − μ̂(0, X) + α̂(O)·(Y − μ̂(O))]."""
    mu1 = predict_mu(mu_hat, df, 1.0)
    mu0 = predict_mu(mu_hat, df, 0.0)
    eif = (mu1 - mu0) + alpha_hat * (df["y"].to_numpy() - mu_hat_oof)
    psi = float(eif.mean())
    se = float(np.sqrt(np.var(eif, ddof=1) / len(eif)))
    return psi, se


def main():
    print("# Loading NSW (treated) + CPS (controls)")
    df = load_data()
    print(f"  n = {len(df)}  ({(df['a']==1).sum()} treated, {(df['a']==0).sum()} control)")

    print("# Fitting μ̂ via 5-fold cross-fit (xgboost)")
    mu_hat_oof = fit_outcome_regression_oof(df)
    mu_hat_full = fit_outcome_regression_full(df)

    print("# Fitting α̂ via 5-fold cross_val_predict (rieszboost)")
    alpha_hat = fit_alpha_oof(df)

    print("# Diagnostics on α̂ ...")
    from rieszboost.diagnostics import diagnose
    print(diagnose(alpha_hat).summary())

    psi, se = eee_ate(df, mu_hat_full, mu_hat_oof, alpha_hat)
    ci = (psi - 1.96 * se, psi + 1.96 * se)
    print(f"\n=== Lalonde ATE (NSW-treated vs CPS-controls) ===")
    print(f"  ψ̂_ATE       : {psi:>10.2f}")
    print(f"  SE          : {se:>10.2f}")
    print(f"  95% CI      : [{ci[0]:>9.2f}, {ci[1]:>9.2f}]")
    print(f"  experimental NSW benchmark: ~$1,794 (Dehejia-Wahba 1999)")


if __name__ == "__main__":
    main()
