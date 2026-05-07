"""Growth policy: depthwise + leafwise both produce reasonable trees."""
from __future__ import annotations

import numpy as np

from riesztree import ATE, RieszTreeRegressor, n_leaves


def test_depthwise_respects_max_depth(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(800, seed=0)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        growth_policy="depthwise",
        max_depth=3,
    )
    est.fit(df)
    # depth ≤ 3 ⇒ leaves ≤ 2^3 = 8.
    assert n_leaves(est.predictor_.tree) <= 8


def test_leafwise_respects_max_leaf_nodes(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(1500, seed=0)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        growth_policy="leafwise",
        max_leaf_nodes=12,
        max_depth=20,  # don't let depth bind
    )
    est.fit(df)
    assert n_leaves(est.predictor_.tree) <= 12


def test_invalid_growth_policy_raises(linear_gaussian_ate, covariate_keys):
    import pytest
    make, _ = linear_gaussian_ate
    df = make(200, seed=0)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        growth_policy="random-forest",
    )
    with pytest.raises(ValueError, match="growth_policy"):
        est.fit(df)


def test_leafwise_yields_lower_train_loss_per_leaf(linear_gaussian_ate, covariate_keys):
    """At equal leaf budget, best-first should achieve ≤ training loss than
    depth-first (it greedily picks the most-impactful split anywhere). At
    least: it should not be dramatically worse."""
    make, _ = linear_gaussian_ate
    df = make(2000, seed=0)
    est_dw = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        growth_policy="depthwise",
        max_depth=4,                # ≤ 16 leaves
    )
    est_lw = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        growth_policy="leafwise",
        max_leaf_nodes=16,
        max_depth=20,
    )
    est_dw.fit(df)
    est_lw.fit(df)
    train_loss_dw = est_dw.riesz_loss(df)
    train_loss_lw = est_lw.riesz_loss(df)
    # leafwise w/ same leaf budget shouldn't be substantially worse on train.
    assert train_loss_lw <= train_loss_dw + 0.05
