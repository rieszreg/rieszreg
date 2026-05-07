"""sklearn conformance: clone preserves params, GridSearchCV runs,
cross_val_predict returns OOF predictions."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_predict

from krrr import ATE, Gaussian, KernelRieszRegressor


def test_clone(binary_ate_data):
    df, _, _ = binary_ate_data
    krr = KernelRieszRegressor(
        estimand=ATE("a", ("x",)),
        kernel=Gaussian(length_scale=0.5),
        lambda_grid=np.logspace(-3, 0, 4),
        validation_fraction=0.2,
    )
    cloned = clone(krr)
    # Params should match
    p_orig = krr.get_params(deep=False)
    p_cloned = cloned.get_params(deep=False)
    assert p_orig.keys() == p_cloned.keys()
    # And cloned is unfitted
    assert not hasattr(cloned, "_booster")
    # Both fit
    krr.fit(df)
    cloned.fit(df)
    np.testing.assert_array_equal(krr.predict(df), cloned.predict(df))


def test_grid_search_cv(binary_ate_data):
    df, _, _ = binary_ate_data
    krr = KernelRieszRegressor(
        estimand=ATE("a", ("x",)),
        lambda_grid=np.logspace(-3, 0, 4),
        validation_fraction=0.0,  # let CV do the work
    )
    grid = GridSearchCV(
        krr,
        param_grid={"lambda_grid": [
            np.array([1e-3]),
            np.array([1e-2]),
            np.array([1e-1]),
        ]},
        cv=3,
    )
    grid.fit(df)
    assert grid.best_estimator_ is not None
    alpha = grid.best_estimator_.predict(df)
    assert alpha.shape == (len(df),)


def test_cross_val_predict(binary_ate_data):
    df, _, _ = binary_ate_data
    krr = KernelRieszRegressor(
        estimand=ATE("a", ("x",)),
        lambda_grid=np.logspace(-3, 0, 4),
        validation_fraction=0.2,
    )
    oof = cross_val_predict(krr, df, cv=3)
    assert oof.shape == (len(df),)
    assert np.all(np.isfinite(oof))
