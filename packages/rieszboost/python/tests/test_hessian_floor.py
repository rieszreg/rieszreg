"""``XGBoostBackend(hessian_floor="auto")`` and SklearnBackend's line search
floor each row at the curvature an observed row has at the current prediction
(``Loss.curvature_eta``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from rieszboost import RieszBooster, SklearnBackend, XGBoostBackend
from rieszreg import ATE, TSM, KLLoss
from rieszreg.testing import dgps


def _treatment_data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    pi = np.clip(1 / (1 + np.exp(-(0.8 * x[:, 0] - 0.5 * x[:, 1] + 0.3 * x[:, 0] * x[:, 2]))), 0.05, 0.95)
    a = (rng.uniform(size=n) < pi).astype(float)
    return pd.DataFrame({"a": a, "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2]}), a, pi


def test_auto_floor_equals_fixed_floor_two_under_squared_loss():
    """SquaredLoss curvature is the constant 2, so "auto" reproduces the
    classic hessian_floor=2.0 fit exactly."""
    df, _, _ = _treatment_data(500, 0)
    fits = [
        RieszBooster(ATE(), backend=XGBoostBackend(n_estimators=50, hessian_floor=f)).fit(df)
        for f in ("auto", 2.0)
    ]
    np.testing.assert_array_equal(fits[0].predict(df), fits[1].predict(df))


def test_auto_floor_recovers_kl_tsm_representer_better_than_fixed_floor():
    """With KLLoss the observed-row curvature is α̂, far below 2 for most rows;
    a fixed floor of 2 under-steps, while "auto" recovers α₀ = 1{a=1}/π."""
    df, _, _ = _treatment_data(2000, 0)
    test, a_t, pi_t = _treatment_data(20000, 1)
    truth = a_t / pi_t

    def rmse(floor):
        est = RieszBooster(TSM(level=1), loss=KLLoss(),
                           backend=XGBoostBackend(hessian_floor=floor)).fit(df)
        return float(np.sqrt(np.mean((est.predict(test) - truth) ** 2)))

    auto, fixed = rmse("auto"), rmse(2.0)
    assert auto < 0.6
    assert auto < 0.8 * fixed


@pytest.mark.parametrize("dgp, estimand", [
    (dgps.linear_gaussian_ate(), ATE()),
    (dgps.logistic_tsm(level=1.0), TSM(level=1)),
])
def test_sklearn_backend_recovers_alpha(dgp, estimand):
    """SklearnBackend's line search floors counterfactual rows at the same
    curvature. With a near-zero floor the step blew up wherever a tree leaf
    held only counterfactual rows (relative error 0.36–0.68 here vs ~0.12)."""
    train = dgp.sample(2000, np.random.default_rng(0))
    test = dgp.sample(5000, np.random.default_rng(10))
    alpha0 = dgp.true_alpha(test)
    alpha = RieszBooster(
        estimand, backend=SklearnBackend(lambda: DecisionTreeRegressor(max_depth=3))
    ).fit(train).predict(test)
    assert np.sqrt(np.mean((alpha - alpha0) ** 2)) < 0.25 * np.sqrt(np.mean(alpha0**2))
