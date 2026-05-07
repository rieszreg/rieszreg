"""Per-leaf decoupling: closed-form α* = -C/D matches numerical minimum."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from rieszreg import (
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    SquaredLoss,
)
from riesztree.splitter import make_leaf_solvers


def _features(df, feature_keys):
    return df[list(feature_keys)].to_numpy(dtype=float)


def test_squared_closed_form_matches_numerical(linear_gaussian_ate, covariate_keys):
    from rieszreg import ATE

    make, _ = linear_gaussian_ate
    df = make(2000, seed=1)
    feats = _features(df, ("a",) + covariate_keys)
    aug = ATE(treatment="a", covariates=covariate_keys).augment(feats)
    D, C = aug.is_original, aug.potential_deriv_coef

    leaf_loss, alpha_at = make_leaf_solvers(SquaredLoss())
    # Bin into quadrants by (a, x0).
    a_col = aug.features[:, 0]
    x0_col = aug.features[:, 1]
    leaf = (a_col > 0.5).astype(int) * 2 + (x0_col > 0).astype(int)
    sq = SquaredLoss()

    for k in np.unique(leaf):
        sel = leaf == k
        Dk, Ck = float(D[sel].sum()), float(C[sel].sum())
        a_closed = alpha_at(Dk, Ck)
        assert np.isclose(a_closed, -Ck / Dk)
        # Numerical min of empirical augmented loss in this leaf.
        res = minimize_scalar(
            lambda a: float(np.sum(D[sel] * sq.tilde_potential(a)
                                   + C[sel] * sq.potential_deriv(a)))
        )
        assert np.isclose(res.x, a_closed, atol=1e-4)
        assert np.isclose(leaf_loss(Dk, Ck), -Ck * Ck / Dk)


def test_kl_closed_form_matches_numerical_when_in_domain(linear_gaussian_ate, covariate_keys):
    from rieszreg import TSM

    make, _ = linear_gaussian_ate
    df = make(2000, seed=2)
    feats = _features(df, ("a",) + covariate_keys)
    aug = TSM(level=1.0, treatment="a", covariates=covariate_keys).augment(feats)
    D, C = aug.is_original, aug.potential_deriv_coef

    leaf_loss, alpha_at = make_leaf_solvers(KLLoss())
    kl = KLLoss()

    # Bin by x0; for TSM C ≤ 0 always, so α* = -C/D > 0 is in-domain.
    bins = np.quantile(aug.features[:, 1], [0.5])
    leaf = np.digitize(aug.features[:, 1], bins)

    for k in np.unique(leaf):
        sel = leaf == k
        Dk, Ck = float(D[sel].sum()), float(C[sel].sum())
        a_closed = alpha_at(Dk, Ck)
        # KL has α > 0; bounded numerical search.
        res = minimize_scalar(
            lambda a: float(np.sum(D[sel] * kl.tilde_potential(a)
                                   + C[sel] * kl.potential_deriv(a))),
            bounds=(1e-6, 100.0), method="bounded",
        )
        assert np.isclose(res.x, a_closed, atol=1e-3), (
            f"closed={a_closed}, numeric={res.x}, D={Dk}, C={Ck}"
        )


def test_kl_disqualifies_C_positive_split():
    leaf_loss, _ = make_leaf_solvers(KLLoss())
    # C > 0, D > 0: α* would need to be negative ⇒ infeasible.
    assert leaf_loss(10.0, 5.0) == float("inf")
    # C = 0: boundary, loss = 0.
    assert leaf_loss(10.0, 0.0) == 0.0
    # C < 0: feasible.
    assert np.isfinite(leaf_loss(10.0, -3.0))


def test_bernoulli_disqualifies_out_of_domain_split():
    leaf_loss, _ = make_leaf_solvers(BernoulliLoss())
    # α* must be in (0, 1), so C must be in (-D, 0).
    assert leaf_loss(10.0, 5.0) == float("inf")     # α* < 0
    assert leaf_loss(10.0, -15.0) == float("inf")   # α* > 1
    assert np.isfinite(leaf_loss(10.0, -3.0))       # in (0, 1)


def test_bounded_squared_projects_to_interval():
    leaf_loss, alpha_at = make_leaf_solvers(BoundedSquaredLoss(lo=0.0, hi=1.0))
    # α* = -C/D = -(-5)/10 = 0.5, in (0,1) → unprojected.
    assert np.isclose(alpha_at(10.0, -5.0), 0.5)
    # α* = 2.0 → projects to 1.0
    assert np.isclose(alpha_at(10.0, -20.0), 1.0)
    # α* = -1.0 → projects to 0.0
    assert np.isclose(alpha_at(10.0, 10.0), 0.0)
