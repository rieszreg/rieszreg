"""Loss-aware splits across the four built-in losses."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rieszreg import BernoulliLoss, BoundedSquaredLoss, KLLoss, Loss, SquaredLoss
from riesztree import ATE, RieszTreeRegressor, TSM


def test_squared_default_when_no_loss_passed(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(500, seed=0)
    est = RieszTreeRegressor(estimand=ATE(treatment="a", covariates=covariate_keys), max_depth=3)
    est.fit(df)
    assert isinstance(est._resolved_loss(), SquaredLoss)


def test_kl_loss_yields_nonnegative_alpha_on_tsm(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(800, seed=0)
    est = RieszTreeRegressor(
        estimand=TSM(level=1.0, treatment="a", covariates=covariate_keys),
        loss=KLLoss(),
        max_depth=4,
    ).fit(df)
    a_hat = est.predict(df)
    assert (a_hat >= 0).all()


def test_bounded_squared_clips_alpha():
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(0, 1, (n, 3))
    pi = 1 / (1 + np.exp(-2.0 * x[:, 0]))    # extreme weights to force clipping
    a = (rng.uniform(0, 1, n) < pi).astype(float)
    df = pd.DataFrame(x, columns=["x0", "x1", "x2"])
    df.insert(0, "a", a)
    bs = BoundedSquaredLoss(lo=-3.0, hi=3.0)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=("x0", "x1", "x2")),
        loss=bs,
        max_depth=5,
    ).fit(df)
    a_hat = est.predict(df)
    assert (a_hat >= -3.0).all()
    assert (a_hat <= 3.0).all()


def test_unsupported_loss_raises_at_fit(linear_gaussian_ate, covariate_keys):
    """Inline-constructed Loss subclass with no analytic leaf solver should
    raise NotImplementedError when the splitter dispatcher is consulted."""
    class WeirdLoss(Loss):
        name = "weird"
        def potential(self, alpha):
            return np.exp(alpha)
        def potential_deriv(self, alpha):
            return np.exp(alpha)
        def link_to_alpha(self, eta):
            return eta
        def alpha_to_eta(self, alpha):
            return alpha

    make, _ = linear_gaussian_ate
    df = make(300, seed=0)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        loss=WeirdLoss(),
        max_depth=3,
    )
    with pytest.raises(NotImplementedError, match="leaf solver"):
        est.fit(df)
