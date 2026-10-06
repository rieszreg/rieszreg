"""Both forests recover α₀ at their default hyperparameters."""

from __future__ import annotations

import numpy as np
import pytest

from rieszreg.testing import dgps

from forestriesz import ATE, TSM, AugForestRieszRegressor, ForestRieszRegressor


def test_forest_riesz_tsm_is_zero_off_level():
    """α₀ = 1[a=1]/π is exactly 0 on control rows. The TSM basis 1[a=1]
    carries no intercept, so nothing may be added outside it."""
    dgp = dgps.logistic_tsm(level=1.0)
    df = dgp.sample(1000, np.random.default_rng(0))
    alpha = ForestRieszRegressor(estimand=TSM(level=1), n_estimators=50, random_state=0).fit(df).predict(df)
    np.testing.assert_array_equal(alpha[df["a"] == 0], 0.0)


@pytest.mark.parametrize("dgp, estimand", [
    (dgps.linear_gaussian_ate(), ATE()),
    (dgps.logistic_tsm(level=1.0), TSM(level=1)),
])
def test_aug_forest_defaults_recover_alpha(dgp, estimand):
    """At the defaults, out-of-sample α̂ is far closer to α₀ than α ≡ 0 is.
    (With min_samples_leaf=1 the error was worse than predicting 0.)"""
    train = dgp.sample(2000, np.random.default_rng(0))
    test = dgp.sample(5000, np.random.default_rng(1))
    alpha0 = dgp.true_alpha(test)
    alpha = AugForestRieszRegressor(estimand=estimand, random_state=0).fit(train).predict(test)
    rmse = np.sqrt(np.mean((alpha - alpha0) ** 2))
    assert rmse < 0.3 * np.sqrt(np.mean(alpha0**2))
