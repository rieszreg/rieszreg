"""OutcomeRegNormSq parity vs. stock MSE regressors.

`m(α)(z, y) = α(x) · y` has Riesz representer μ_0(x) = E[Y | X=x]; under the
squared Bregman-Riesz loss the augmented dataset reduces to the original X
with `is_original = 1`, `potential_deriv_coef = -y`, so the empirical loss
equals ∑(α(x_i) − y_i)². Riesz-trained predictions should therefore track
plain MSE-trained predictions on the same data with matched hyperparameters.

Per-backend parity (Pearson > 0.99) is the gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rieszboost import RieszBooster, SklearnBackend, XGBoostBackend
from rieszreg import OutcomeRegNormSq, SquaredLoss
from rieszreg.testing.parity import compare


def _regression_df(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    y = np.sin(x0) + 0.5 * x1 + rng.normal(0.0, 0.3, n)
    df = pd.DataFrame({"x0": x0, "x1": x1})
    return df, y


def test_xgboost_backend_parity_with_xgbregressor():
    """Riesz-loss training of `XGBoostBackend` on `OutcomeRegNormSq` should
    track `xgb.XGBRegressor(objective="reg:squarederror")` with matched
    hyperparameters. `reg_lambda=0` makes the leaf-weight Newton step
    bit-equivalent (custom obj's grad/hess are 2x the default's, so the
    `−G/(H+λ)` ratio agrees only when λ=0)."""
    import xgboost as xgb

    df, y = _regression_df(n=400, seed=0)

    common = dict(n_estimators=100, learning_rate=0.1, random_state=0)

    riesz = RieszBooster(
        estimand=OutcomeRegNormSq(covariates=("x0", "x1")),
        loss=SquaredLoss(),
        backend=XGBoostBackend(),
        max_depth=3,
        reg_lambda=0.0,
        subsample=1.0,
        **common,
    ).fit(df, y)

    ref = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=3,
        reg_lambda=0.0,
        subsample=1.0,
        base_score=float(y.mean()),
        seed=0,
        **common,
    )
    ref.fit(df.values, y)

    rep = compare(riesz.predict(df), ref.predict(df.values))
    assert rep.pearson > 0.99, rep.summary()


def test_sklearn_backend_parity_with_gradient_boosting():
    """Riesz-loss training of `SklearnBackend(DecisionTreeRegressor)` on
    `OutcomeRegNormSq` should track `sklearn.ensemble.GradientBoostingRegressor`
    with matched hyperparameters. Both run Friedman-style first-order gradient
    boosting against the same base learner under squared loss."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor

    df, y = _regression_df(n=400, seed=0)

    common = dict(n_estimators=100, learning_rate=0.1, random_state=0)

    riesz = RieszBooster(
        estimand=OutcomeRegNormSq(covariates=("x0", "x1")),
        loss=SquaredLoss(),
        backend=SklearnBackend(
            lambda: DecisionTreeRegressor(max_depth=3, random_state=0)
        ),
        **common,
    ).fit(df, y)

    ref = GradientBoostingRegressor(
        loss="squared_error",
        max_depth=3,
        **common,
    ).fit(df.values, y)

    rep = compare(riesz.predict(df), ref.predict(df.values))
    assert rep.pearson > 0.99, rep.summary()
