"""NHEFS: average shift effect of cutting daily cigarette intensity by 5
on 1971→1982 weight change.

NHEFS is the National Health and Nutrition Examination Survey follow-up,
the standard dataset in Hernán & Robins's *Causal Inference: What If*.
It records cigarettes per day at baseline (`smokeintensity`) and weight
change over the follow-up (`wt82_71`).

We estimate the *average shift effect*

    ψ_ASE = E[μ(A − 5, X)] − E[μ(A, X)]

— what would the average weight change be if every smoker reduced by 5
cigarettes per day, all else held constant? (Negative δ = reducing intensity.)

The Riesz representer of `m(O, μ) = μ(A + δ, X) − μ(A, X)` is
α₀(A, X) = p_{A|X}(A − δ | X) / p_{A|X}(A | X) − 1, but rieszboost
estimates it directly without ever needing the conditional density of A.

EEE / one-step:
    ψ̂ = (1/n) Σ [ μ̂(A − 5, X) − μ̂(A, X) + α̂(O)·(Y − μ̂(O)) ]

Run:
    .venv/bin/python examples/nhefs_shift.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict

from rieszboost import AdditiveShift, RieszBooster

try:
    from causaldata import nhefs_complete
except ImportError as e:
    raise SystemExit(
        "This example needs `pip install causaldata`."
    ) from e


# Adjustment set used in the Hernán-Robins continuous-treatment chapter.
COVARIATES = [
    "sex", "age", "race", "education", "smokeyrs", "active",
    "exercise", "wt71",
]
TREATMENT = "smokeintensity"
OUTCOME = "wt82_71"

DELTA = -5.0  # everyone smokes 5 fewer cigarettes per day


def load_data() -> pd.DataFrame:
    df = nhefs_complete.load_pandas().data
    df = df[[TREATMENT, OUTCOME] + COVARIATES].dropna().copy()
    # rieszboost's default treatment column is `a`; keep it conventional.
    df = df.rename(columns={TREATMENT: "a", OUTCOME: "y"})
    df["a"] = df["a"].astype(float)
    df["y"] = df["y"].astype(float)
    return df


def fit_outcome_regression_oof(df: pd.DataFrame, n_folds: int = 5) -> np.ndarray:
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
    X = df[["a"] + COVARIATES].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return xgb.train(
        {"objective": "reg:squarederror", "learning_rate": 0.05,
         "max_depth": 4, "reg_lambda": 1.0, "seed": 0, "verbosity": 0},
        xgb.DMatrix(X, label=y),
        num_boost_round=300,
    )


def predict_mu(mu_hat: xgb.Booster, df: pd.DataFrame, a_value=None) -> np.ndarray:
    a = df["a"].to_numpy(dtype=float) if a_value is None else np.asarray(a_value, dtype=float)
    if np.ndim(a) == 0:
        a = np.full(len(df), float(a))
    X = np.column_stack([a, df[COVARIATES].to_numpy(dtype=float)])
    return mu_hat.predict(xgb.DMatrix(X))


def fit_alpha_oof(df: pd.DataFrame, n_folds: int = 5) -> np.ndarray:
    booster = RieszBooster(
        estimand=AdditiveShift(delta=DELTA, treatment="a", covariates=tuple(COVARIATES)),
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


def eee_shift(df, mu_hat, mu_hat_oof, alpha_hat):
    mu_obs = predict_mu(mu_hat, df)
    mu_shifted = predict_mu(mu_hat, df, a_value=df["a"].to_numpy() + DELTA)
    eif = (mu_shifted - mu_obs) + alpha_hat * (df["y"].to_numpy() - mu_hat_oof)
    psi = float(eif.mean())
    se = float(np.sqrt(np.var(eif, ddof=1) / len(eif)))
    return psi, se


def main():
    print("# Loading NHEFS")
    df = load_data()
    print(f"  n = {len(df)}, smokeintensity range = [{df['a'].min():.0f}, {df['a'].max():.0f}]")
    print(f"  outcome (wt82_71) mean = {df['y'].mean():.2f}, sd = {df['y'].std():.2f}")
    print(f"  intervention: cut daily intensity by {-DELTA:.0f}")

    print("\n# Fitting μ̂ via 5-fold cross-fit (xgboost)")
    mu_oof = fit_outcome_regression_oof(df)
    mu_full = fit_outcome_regression_full(df)

    print("# Fitting α̂ via 5-fold cross_val_predict (rieszboost)")
    alpha_hat = fit_alpha_oof(df)

    from rieszboost.diagnostics import diagnose
    print(diagnose(alpha_hat).summary())

    psi, se = eee_shift(df, mu_full, mu_oof, alpha_hat)
    ci = (psi - 1.96 * se, psi + 1.96 * se)
    print(f"\n=== NHEFS additive shift effect (δ = {DELTA}) ===")
    print(f"  ψ̂_ASE       : {psi:>8.3f} kg")
    print(f"  SE          : {se:>8.3f}")
    print(f"  95% CI      : [{ci[0]:>7.3f}, {ci[1]:>7.3f}]")
    print(f"\n  Interpretation: under a hypothetical intervention reducing")
    print(f"  daily cigarette intensity by {-DELTA:.0f} for everyone, the average")
    print(f"  weight change (1971 → 1982) shifts by {psi:+.2f} kg vs no-change.")


if __name__ == "__main__":
    main()
