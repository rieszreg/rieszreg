"""End-to-end correctness for AugForestRieszRegressor.

The augmentation-style backend should converge to α₀ across the canonical
DGPs without needing a sieve. ATE and TSM both work without
``riesz_feature_fns``; AdditiveShift (which the moment-style path can't
handle without a sieve) also works here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rieszreg.testing import dgps

from forestriesz import (
    AdditiveShift,
    AugForestRieszRegressor,
    ATE,
    TSM,
)


def _fit_predict_factory(estimand):
    def _fit_predict(train, test):
        est = AugForestRieszRegressor(
            estimand=estimand,
            n_estimators=200,
            min_samples_leaf=10,
            random_state=0,
        )
        est.fit(train)
        return est.predict(test)
    return _fit_predict


def test_aug_ate_consistency_grid():
    rmses = dgps.assert_consistency(
        _fit_predict_factory(ATE()),
        dgp=dgps.linear_gaussian_ate(),
        n_grid=(500, 2000),
        rng_seed=0,
        tol_at_max_n=1.0,
        monotonicity_slack=0.5,
    )
    assert rmses[-1] < rmses[0] * 1.5


def test_aug_tsm_consistency_grid():
    rmses = dgps.assert_consistency(
        _fit_predict_factory(TSM(level=1)),
        dgp=dgps.logistic_tsm(level=1.0),
        n_grid=(500, 2000),
        rng_seed=0,
        tol_at_max_n=1.0,
        monotonicity_slack=0.5,
    )
    assert rmses[-1] < rmses[0] * 1.5


@pytest.fixture
def df_continuous():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    a = rng.normal(0.5 * x, 1.0)
    return pd.DataFrame({"a": a, "x": x})


def test_aug_additive_shift_runs(df_continuous):
    """The moment-style backend raises on AdditiveShift; this one shouldn't."""
    est = AugForestRieszRegressor(
        estimand=AdditiveShift(delta=0.5),
        n_estimators=50,
        min_samples_leaf=15,
        random_state=0,
    )
    est.fit(df_continuous)
    pred = est.predict(df_continuous)
    assert pred.shape == (len(df_continuous),)
    assert np.all(np.isfinite(pred))


def test_aug_custom_estimand_runs():
    """Augmentation-style must work on a fully custom Estimand."""
    from rieszreg import FiniteEvalEstimand

    # Custom moment: alpha(a + 1, x) - alpha(a - 1, x).
    def m(alpha):
        def inner(z, y=None):
            return alpha(a=z["a"] + 1.0, x=z["x"]) - alpha(a=z["a"] - 1.0, x=z["x"])
        return inner

    estimand = FiniteEvalEstimand(feature_keys=("a", "x"), m=m)

    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    a = rng.normal(0.0, 1.0, size=n)
    df = pd.DataFrame({"a": a, "x": x})

    est = AugForestRieszRegressor(
        estimand=estimand,
        n_estimators=30,
        min_samples_leaf=15,
        random_state=0,
    )
    est.fit(df)
    pred = est.predict(df)
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))
