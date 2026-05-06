"""Cost-complexity pruning behaviour."""
from __future__ import annotations

import numpy as np

from riesztree import ATE, RieszTreeRegressor, n_leaves


def test_pruning_reduces_leaves(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(1500, seed=0)
    estimand = ATE(treatment="a", covariates=covariate_keys)
    base = RieszTreeRegressor(estimand=estimand, max_depth=8, ccp_alpha=0.0).fit(df)
    pruned = RieszTreeRegressor(estimand=estimand, max_depth=8, ccp_alpha=0.5).fit(df)
    assert n_leaves(pruned.predictor_.tree) <= n_leaves(base.predictor_.tree)


def test_pruning_zero_is_identity(linear_gaussian_ate, covariate_keys):
    """ccp_alpha=0 should not change the tree."""
    make, _ = linear_gaussian_ate
    df = make(800, seed=2)
    estimand = ATE(treatment="a", covariates=covariate_keys)
    a = RieszTreeRegressor(estimand=estimand, max_depth=5, ccp_alpha=0.0).fit(df).predict(df)
    b = RieszTreeRegressor(estimand=estimand, max_depth=5, ccp_alpha=0.0).fit(df).predict(df)
    assert np.allclose(a, b)


def test_huge_pruning_collapses_to_root(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(1000, seed=3)
    estimand = ATE(treatment="a", covariates=covariate_keys)
    pruned = RieszTreeRegressor(
        estimand=estimand, max_depth=6, ccp_alpha=1e9
    ).fit(df)
    assert n_leaves(pruned.predictor_.tree) == 1
