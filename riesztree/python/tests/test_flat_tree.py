"""Phase 3 parity: flat-array Tree + Cython predict ≡ Node tree-walk.

Locks the contract that swapping the predictor from a Python ``Node``
tree-walk to a Cython flat-array walk is observably a no-op modulo
floating-point identity. Any drift means the optimisation also changed
the algorithm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riesztree import (
    ATE,
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    RieszTreeRegressor,
    SquaredLoss,
    TSM,
)
from riesztree.fast import (
    FlatTree,
    flat_tree_from_node,
    node_from_flat_tree,
    predict_alpha,
)
from riesztree.fast._tree import _predict_alpha_python
from riesztree.tree import predict_array as node_predict_array


def _make_df(n=600, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    logit = 0.5 * X[:, 0] + 0.4 * X[:, 1]
    pi = 1.0 / (1.0 + np.exp(-logit))
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    cols = {f"x{j}": X[:, j] for j in range(p)}
    cols["a"] = a
    return pd.DataFrame(cols)


def _ate_estimand(p):
    return ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))


# ---------------------------------------------------------------------------
# Adapter round-trip

def test_flat_tree_round_trip_predicts_identically():
    """flat_tree_from_node ∘ node_from_flat_tree is the identity on the
    prediction surface (the lossy reverse adapter is documented as
    sufficient for prediction only)."""
    df = _make_df(p=6, n=400)
    est = RieszTreeRegressor(estimand=_ate_estimand(6), max_depth=4).fit(df)
    flat = flat_tree_from_node(est.predictor_.tree)
    rebuilt = node_from_flat_tree(flat)

    feats = df[list(est.feature_keys_)].to_numpy(dtype=np.float64)
    a_node = node_predict_array(est.predictor_.tree, feats)
    a_round = node_predict_array(rebuilt, feats)
    np.testing.assert_array_equal(a_node, a_round)


# ---------------------------------------------------------------------------
# Cython == Python fallback == Node tree-walk

@pytest.mark.parametrize("max_depth", [2, 4, 8])
def test_cython_predict_matches_node_walk(max_depth):
    """The Cython predict_alpha_c must agree with the Python Node walk
    bit-for-bit on continuous-only trees."""
    df = _make_df(p=8, n=500)
    est = RieszTreeRegressor(estimand=_ate_estimand(8), max_depth=max_depth).fit(df)
    feats = df[list(est.feature_keys_)].to_numpy(dtype=np.float64)

    flat = flat_tree_from_node(est.predictor_.tree)
    a_cython = predict_alpha(flat, feats)            # Cython tight loop
    a_node = node_predict_array(est.predictor_.tree, feats)
    np.testing.assert_array_equal(a_cython, a_node)


def test_python_fallback_matches_cython():
    """The pure-Python fallback (used when the .so is missing) must
    match the compiled path. Guards against algorithmic drift between
    the two implementations."""
    df = _make_df(p=8, n=500)
    est = RieszTreeRegressor(estimand=_ate_estimand(8), max_depth=6).fit(df)
    feats = df[list(est.feature_keys_)].to_numpy(dtype=np.float64)
    flat = flat_tree_from_node(est.predictor_.tree)
    np.testing.assert_array_equal(
        _predict_alpha_python(flat, feats), predict_alpha(flat, feats)
    )


# ---------------------------------------------------------------------------
# All four losses traverse the flat tree correctly

@pytest.mark.parametrize(
    "loss_cls, estimand_factory",
    [
        (SquaredLoss, lambda p: ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))),
        (KLLoss, lambda p: TSM(treatment="a", covariates=tuple(f"x{j}" for j in range(p)), level=1.0)),
        (BernoulliLoss, lambda p: ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))),
        (BoundedSquaredLoss, lambda p: ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))),
    ],
)
def test_flat_predict_matches_node_per_loss(loss_cls, estimand_factory):
    df = _make_df(p=4, n=500)
    estimand = estimand_factory(4)
    loss = (
        BoundedSquaredLoss(lo=-3.0, hi=3.0)
        if loss_cls is BoundedSquaredLoss
        else loss_cls()
    )
    est = RieszTreeRegressor(estimand=estimand, loss=loss, max_depth=3).fit(df)
    feats = df[list(est.feature_keys_)].to_numpy(dtype=np.float64)
    a_pred = est.predict(df)                   # goes through predictor (flat path)
    a_node = node_predict_array(est.predictor_.tree, feats)
    np.testing.assert_allclose(a_pred, a_node, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Categorical splits exercise the fallback path inside the Cython loop

def test_categorical_split_predict_matches_node_walk():
    rng = np.random.default_rng(0)
    n = 800
    cat = rng.integers(0, 6, size=n).astype(float)
    pi_per_level = rng.uniform(0.1, 0.9, size=6)
    pi = pi_per_level[cat.astype(int)]
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    x = rng.normal(0.0, 1.0, size=n)
    df = pd.DataFrame({"a": a, "cat": cat, "x": x})
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=("cat", "x")),
        max_depth=4,
        categorical_features=(0,),
    ).fit(df)
    feats = df[list(est.feature_keys_)].to_numpy(dtype=np.float64)
    a_pred = est.predict(df)
    a_node = node_predict_array(est.predictor_.tree, feats)
    np.testing.assert_array_equal(a_pred, a_node)


# ---------------------------------------------------------------------------
# Pruning invalidates the flat-tree cache

def test_predict_reflects_post_pruning_tree():
    """If pruning happens after a flat tree was cached, predict_alpha
    must rebuild the flat tree before walking. We trigger this by
    invalidate_flat_tree() after mutating in place."""
    from riesztree.pruning import cost_complexity_prune

    df = _make_df(p=4, n=400)
    est = RieszTreeRegressor(estimand=_ate_estimand(4), max_depth=6).fit(df)

    # Force the cache.
    pre = est.predict(df)
    assert est.predictor_._flat_tree is not None

    # Heavy prune in place, invalidate cache.
    cost_complexity_prune(est.predictor_.tree, est.predictor_.loss, ccp_alpha=1e9)
    est.predictor_.invalidate_flat_tree()

    post = est.predict(df)
    # All-leaves-collapsed tree predicts a constant.
    assert np.unique(post).size == 1
    # And differs from pre-pruning predictions on a non-trivial tree.
    if np.unique(pre).size > 1:
        assert not np.array_equal(pre, post)
