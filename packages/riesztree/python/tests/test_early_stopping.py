"""Early stopping via held-out augmented loss."""
from __future__ import annotations

from riesztree import ATE, RieszTreeRegressor


def test_early_stopping_reduces_tree_size(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(1500, seed=0)

    # Without early stopping (deep tree).
    est_no_es = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        max_depth=10,
    ).fit(df)

    # With early stopping (small patience).
    est_es = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        max_depth=10,
        early_stopping_rounds=3,
        validation_fraction=0.2,
    ).fit(df)

    assert est_es.get_n_leaves() < est_no_es.get_n_leaves()


def test_early_stopping_works_with_leafwise(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(1500, seed=0)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        growth_policy="leafwise",
        max_leaf_nodes=200,
        early_stopping_rounds=3,
        validation_fraction=0.2,
    ).fit(df)
    # Should stop well before 200 leaves on this DGP.
    assert est.get_n_leaves() < 200


import pytest


@pytest.mark.parametrize("growth_policy", ["depthwise", "leafwise"])
def test_early_stopping_keeps_best_partial_tree(
    linear_gaussian_ate, covariate_keys, growth_policy, monkeypatch
):
    """After stopping, the tree is rolled back to the partial tree with the
    lowest held-out loss seen during growth, so its held-out loss equals
    that minimum (not the loss after the trailing non-improving splits)."""
    import riesztree.grow as grow

    seen = []
    real = grow._holdout_loss

    def recording(root, valid, loss):
        seen.append(real(root, valid, loss))
        return seen[-1]

    monkeypatch.setattr(grow, "_holdout_loss", recording)
    make, _ = linear_gaussian_ate
    df = make(1500, seed=0)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        max_depth=10, max_leaf_nodes=200, growth_policy=growth_policy,
        early_stopping_rounds=3, validation_fraction=0.2,
    ).fit(df)
    assert len(seen) > 4
    assert est.best_score_ == pytest.approx(min(seen), rel=1e-9)
