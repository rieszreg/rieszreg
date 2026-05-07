"""Estimator-consistency suite from rieszreg.testing.dgps."""
from __future__ import annotations

from rieszreg.testing.dgps import (
    assert_consistency,
    linear_gaussian_ate,
    logistic_tsm,
)
from riesztree import ATE, RieszTreeRegressor, TSM


def test_consistency_ate():
    dgp = linear_gaussian_ate()
    feature_keys = dgp.feature_keys

    def fit_predict(train, test):
        # The DGP's ATE has covariates=("x",); honour it.
        est = RieszTreeRegressor(
            estimand=ATE(treatment="a", covariates=("x",)),
            max_depth=6,
            min_samples_leaf=10,
            min_samples_split=20,
        ).fit(train)
        return est.predict(test)

    assert_consistency(
        fit_predict, dgp=dgp,
        n_grid=(500, 4000),
        tol_at_max_n=0.7,   # single tree is high-variance; relax vs forest/booster
    )


def test_consistency_tsm():
    dgp = logistic_tsm(level=1.0)

    def fit_predict(train, test):
        est = RieszTreeRegressor(
            estimand=TSM(level=1.0, treatment="a", covariates=("x",)),
            max_depth=6,
            min_samples_leaf=10,
            min_samples_split=20,
        ).fit(train)
        return est.predict(test)

    assert_consistency(
        fit_predict, dgp=dgp,
        n_grid=(500, 4000),
        tol_at_max_n=0.7,   # single tree is high-variance; relax vs forest/booster
    )
